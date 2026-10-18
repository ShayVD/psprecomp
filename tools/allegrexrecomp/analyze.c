/* Function discovery. See analyze.h. */

#include "analyze.h"
#include "decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- a growable uint32 list ---------------------------------------------- */

typedef struct { uint32_t *v; int n, cap; } u32list;

static int u32_push(u32list *l, uint32_t x) {
    if (l->n == l->cap) {
        int cap = l->cap ? l->cap * 2 : 256;
        uint32_t *v = (uint32_t *)realloc(l->v, (size_t)cap * sizeof *v);
        if (!v) return -1;
        l->v = v;
        l->cap = cap;
    }
    l->v[l->n++] = x;
    return 0;
}

static int u32_contains(const u32list *l, uint32_t x) {
    for (int i = 0; i < l->n; i++) if (l->v[i] == x) return 1;
    return 0;
}

/* ---- per-word bookkeeping ------------------------------------------------ */

#define SEEN_CODE  0x01   /* decoded as an instruction */
#define SEEN_ENTRY 0x02   /* known function entry */

/* entry_map values: why an address is an entry. */
#define ENTRY_OWN      1  /* a seed, call or tail-call target: entered from anywhere */
#define ENTRY_PROMOTED 2  /* a cross-function branch target (see the rounds) */

static uint32_t word_index(const a_analysis *an, uint32_t addr) {
    return (addr - an->base) >> 2;
}

int a_in_range(const a_analysis *an, uint32_t addr) {
    return addr >= an->base && addr < an->base + an->size && (addr & 3) == 0;
}

static int is_import_stub(const a_analysis *an, uint32_t addr) {
    return an->stub_size && addr >= an->stub_addr &&
           addr < an->stub_addr + an->stub_size;
}

static uint32_t fetch(const a_analysis *an, uint32_t addr) {
    const uint8_t *p = an->code + (addr - an->base);
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* The general-purpose register an instruction writes, or -1 if it writes
 * none. Used by the matchers below to ask "was this register set again in
 * between?", so it must not miss a writer; every GPR-writing op the decoder
 * names is listed.
 *
 * It is deliberately conservative. MOVZ/MOVN write only when their condition
 * holds and LWL/LWR replace only part of the register, but a matcher cannot
 * know which, so they count as writers. JAL and the *AL branches report only
 * $ra, although the callee also clobbers every caller-saved register: a
 * matcher that tracks a register across a call must account for that itself. */
static int gpr_written(const a_insn *in) {
    switch (in->op) {
    case A_SLL: case A_SRL: case A_SRA: case A_SLLV: case A_SRLV: case A_SRAV:
    case A_ROTR: case A_ROTRV: case A_MOVZ: case A_MOVN: case A_MFHI: case A_MFLO:
    case A_CLZ: case A_CLO: case A_ADD: case A_ADDU: case A_SUB: case A_SUBU:
    case A_AND: case A_OR: case A_XOR: case A_NOR: case A_SLT: case A_SLTU:
    case A_MAX: case A_MIN: case A_JALR:
    case A_WSBH: case A_WSBW: case A_SEB: case A_SEH: case A_BITREV:
        return in->rd;
    case A_ADDI: case A_ADDIU: case A_SLTI: case A_SLTIU: case A_ANDI: case A_ORI:
    case A_XORI: case A_LUI: case A_EXT: case A_INS:
    case A_LB: case A_LH: case A_LWL: case A_LW: case A_LBU: case A_LHU: case A_LWR:
    case A_LL: case A_SC:
    case A_MFC0: case A_CFC0: case A_MFIC: case A_MFC1: case A_CFC1:
    case A_MFV: case A_MFVC:
        return in->rt;
    case A_JAL: case A_BLTZAL: case A_BGEZAL: case A_BLTZALL: case A_BGEZALL:
        return PSP_RA_INDEX;
    default:
        return -1;
    }
}

/* ---- jump tables --------------------------------------------------------- */

/* MIPS compiles a `switch` into a bounds check, a scaled index, a table load
 * and an indirect jump. SNC, which built this game, writes it as
 *
 *     sltiu $a2, $a0, N           bounds check -- N is the entry count
 *     beq   $a2, $zero, default
 *     sll   $a0, $a0, 2           scale to a word index
 *     lui   $at, %hi(table)
 *     addu  $at, $at, $a0
 *     lw    $at, %lo(table)($at)  load the target
 *     jr    $at
 *
 * Recovering it means walking backward from the `jr` for the `lw` that fed it,
 * then for the `addu` that formed its address, then for the `lui` (or
 * lui+addiu/ori) that built the table's base and the `sll` that scaled the
 * index, then for the tests that bound the index (switch_bound). The
 * instructions may be interleaved with unrelated ones, so the search is by
 * register rather than by position, along one straight line of code (a
 * jt_path).
 *
 * Two checks make the result safe. Every reachable entry must land inside
 * .text, be instruction-aligned, and decode as a valid instruction; a
 * misidentified table fails those and is rejected wholesale rather than
 * seeding garbage. And once a walk is complete, the match is repeated along
 * every other way into the code it read (check_tables), so a bound that only
 * holds on the fall-through path does not stand.
 */

static uint32_t image_read32(const a_analysis *an, uint32_t addr) {
    if (!an->image) return 0;
    if (addr < an->image_base || addr + 4 > an->image_base + an->image_size) return 0;
    const uint8_t *p = an->image + (addr - an->image_base);
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* How far back to look. A compiler keeps the whole idiom close together; a
 * wider window mostly finds unrelated instructions that happen to match. */
#define JT_WINDOW 24
#define JT_VALUE_WINDOW (JT_WINDOW * 2)
/* Large only to bound the work: every entry is checked anyway. */
#define JT_MAX_ENTRIES 4096
/* The longest backward path a match can use: lw, addu and sll within
 * JT_WINDOW of each other, then the bound and the values it compares within
 * JT_VALUE_WINDOW of the sll; the table base within 2 * JT_WINDOW of the
 * addu stays inside that. */
#define JT_PATH (JT_WINDOW * 3 + JT_VALUE_WINDOW + 8)

/* Does straight-line flow end here? A call (the callee clobbers the
 * caller-saved registers, which gpr_written does not report), an
 * unconditional transfer (j, jr, jalr, and the always-taken branches `b` =
 * beq $x, $x, `bgez $zero`, `blez $zero` and their likely forms: what
 * follows the delay slot is not reached from them) or a trap (break, eret).
 * Walking backward across one would pair the idiom with values that do not
 * reach the jr. */
static int ends_straight_line(const a_insn *in) {
    if (in->is_call || in->is_jump || a_always_taken(in)) return 1;
    return in->op == A_BREAK || in->op == A_ERET;
}

/* One step of a backward walk along the fall-through path: decode `a` into
 * *out and say whether it runs on the way to where the walk started.
 *
 *   STEP_ON    it does
 *   STEP_SKIP  it does not, but the line continues before it: `a` is the
 *              delay slot of a branch-likely, which is nullified on the
 *              fall-through path (`beql $t, $zero, x; lw $a1, ..` only
 *              loads $a1 when the branch is taken). An always-taken likely
 *              branch has no fall-through and is a STEP_STOP instead.
 *   STEP_STOP  the line ends: `a` is a call or an unconditional transfer,
 *              or the delay slot of one (the slot runs before the transfer,
 *              not on the way to what follows), or outside the code */
enum { STEP_STOP, STEP_ON, STEP_SKIP };

static int step_back(const a_analysis *an, uint32_t a, a_insn *out) {
    if (!a_in_range(an, a)) return STEP_STOP;
    a_decode(fetch(an, a), a, out);
    if (ends_straight_line(out)) return STEP_STOP;
    if (a_in_range(an, a - 4)) {
        a_insn prev;
        a_decode(fetch(an, a - 4), a - 4, &prev);
        if (ends_straight_line(&prev)) return STEP_STOP;
        if (prev.is_branch && prev.is_likely) return STEP_SKIP;
    }
    return STEP_ON;
}

/* The code executed before a jr, newest first: step 0 is the instruction
 * before the jr. Normally that is the fall-through path, read backward until
 * the straight line ends. A list of edges (see check_tables) changes it: an
 * edge says that step `step` (at address `at`; -1 is the jr) was arrived at
 * by the branch or jump at `from`, so the steps before it are `from`'s delay
 * slot, `from` itself (taken) and whatever leads to `from` -- possibly
 * through the next edge; or arrived at as an entry, so nothing is known
 * before it. All matchers read the code through a path, so the same match
 * can be re-run along every way into it.
 *
 * Positions on a path are step numbers, not addresses, so "the same
 * instruction" means the same step. */
enum { JT_EDGE_BRANCH, JT_EDGE_ENTRY };

typedef struct { int kind; uint32_t at, from; int step; } jt_edge;

/* The most edges one path is checked through (see check_tables). The
 * deepest chain any of this game's tables needs is exactly 8 (with 4, six
 * of them were rejected). A table with a path needing more is rejected and
 * its targets promoted to entries, like any other rejected table. */
#define JT_EDGE_DEPTH 8

/* The most check_paths calls one table may use. The chains multiply: a state
 * machine whose cases branch back into its own switch has N^depth of them.
 * A table that runs out is rejected and its targets promoted. The most any
 * of this game's tables needs is 1,108 (the switch at 0x0894ABC8). */
#define JT_CHECK_BUDGET 50000

typedef struct {
    const a_analysis *an;
    uint32_t addr[JT_PATH];
    uint8_t  on[JT_PATH];       /* 0: nullified on this path (likely slot) */
    uint8_t  taken[JT_PATH];    /* 1: a branch this path takes */
    int      n;                 /* steps known; the path ends after them */
    int      deepest;           /* highest step a matcher asked for */
    /* Where the last edge's code starts (its delay slot's step), or -1 when
     * the path has no edges; JT_PATH when the last edge ended the path or
     * was never reached. */
    int      edge_from;
} jt_path;

static void push_step(jt_path *p, uint32_t a, int on, int taken) {
    p->addr[p->n] = a; p->on[p->n] = (uint8_t)on; p->taken[p->n] = (uint8_t)taken; p->n++;
}

static void build_path(const a_analysis *an, uint32_t site, const jt_edge *edges, int nedges,
                       jt_path *p) {
    p->an = an;
    p->n = 0;
    p->deepest = -1;
    p->edge_from = nedges ? JT_PATH : -1;
    int ei = 0;                                 /* the next edge to apply */
    uint32_t cur = site;
    while (p->n < JT_PATH) {
        a_insn in;
        const jt_edge *e = ei < nedges ? &edges[ei] : NULL;
        if (e && p->n - 1 == e->step) {
            ei++;
            if (e->kind == JT_EDGE_ENTRY) return;
            const uint32_t slot = e->from + 4;
            if (!a_in_range(an, slot) || !a_in_range(an, e->from)) return;
            a_decode(fetch(an, slot), slot, &in);
            if (ends_straight_line(&in)) return;
            if (ei == nedges) p->edge_from = p->n;
            push_step(p, slot, 1, 0);
            if (p->n == JT_PATH) return;
            a_decode(fetch(an, e->from), e->from, &in);
            if (in.is_call) return;             /* `at` was called, not jumped to */
            push_step(p, e->from, 1, 1);
            cur = e->from;
            continue;
        }
        const uint32_t a = cur - 4;
        int st;
        if (e && p->n == e->step) {
            /* Arrived here by the edge, so it runs whatever precedes it. */
            if (!a_in_range(an, a)) return;
            a_decode(fetch(an, a), a, &in);
            st = ends_straight_line(&in) ? STEP_STOP : STEP_ON;
        } else {
            st = step_back(an, a, &in);
        }
        if (st == STEP_STOP) return;
        push_step(p, a, st == STEP_ON, 0);
        cur = a;
    }
}

/* Step k of the path: -1 past its end, else whether it runs (1) or is
 * nullified (0), with the instruction decoded into *out. */
static int path_get(jt_path *p, int k, a_insn *out) {
    if (k > p->deepest) p->deepest = k;
    if (k < 0 || k >= p->n) return -1;
    a_decode(fetch(p->an, p->addr[k]), p->addr[k], out);
    return p->on[k];
}

/* The nearest step after `from` (-1: the jr) that writes `reg`, within
 * JT_WINDOW. Returns it and decodes it into *out, or returns -1. */
static int last_writer(jt_path *p, int from, int reg, a_insn *out) {
    for (int k = from + 1; k <= from + JT_WINDOW; k++) {
        const int s = path_get(p, k, out);
        if (s < 0) return -1;
        if (s && gpr_written(out) == reg) return k;
    }
    return -1;
}

/* The constant `reg` holds just before step `from`, if it was built by
 * `lui` alone or by `lui` + `addiu`/`ori`. */
static int const_before(jt_path *p, int from, int reg, uint32_t *val) {
    a_insn w;
    const int k = last_writer(p, from, reg, &w);
    if (k < 0) return 0;
    if (w.op == A_LUI) { *val = (uint32_t)w.imm << 16; return 1; }
    if (w.op == A_ADDIU || w.op == A_ORI) {
        a_insn h;
        if (last_writer(p, k, w.rs, &h) < 0 || h.op != A_LUI) return 0;
        *val = ((uint32_t)h.imm << 16) + (uint32_t)w.imm;
        return 1;
    }
    return 0;
}

/* How the result $t of the compare at step `at` is next tested, before
 * step `end`: +1 if the path continues only when $t is set (`beq $t,
 * $zero` not taken, or `bne` taken), -1 if only when it is clear, 0 if it
 * is overwritten or not tested first. */
static int compare_test(jt_path *p, int at, int end, int t) {
    for (int k = at - 1; k > end; k--) {
        a_insn in;
        if (path_get(p, k, &in) <= 0) continue;
        const int eq = in.op == A_BEQ || in.op == A_BEQL;
        if ((eq || in.op == A_BNE || in.op == A_BNEL) &&
            ((in.rs == t && in.rt == 0) || (in.rs == 0 && in.rt == t))) {
            const int sense = eq ? 1 : -1;
            return p->taken[k] ? -sense : sense;
        }
        if (gpr_written(&in) == t) return 0;
    }
    return 0;
}

/* Where the value `reg` holds just before step `from` came from, looking
 * through copies and constant offsets, no further than step `floor`: the
 * value is root + delta, where root is what `root_reg` held just after step
 * `root_at`. `root_at` is the step that computed it, or the step where the
 * walk ended without finding one (the path's end, or past `floor`): the value
 * was live there. Two walks along the same path with the same floor end at
 * the same step, so equal roots mean equal values. When the copies and
 * offsets lead back to $zero (`addiu $a0, $zero, -1`) the root is $zero,
 * at JT_ROOT_ZERO, the same for every walk: the value is the constant delta.
 * The matchers still treat a root as unknown, which is less precise for a
 * constant but never wrong. Other constants (`ori $a0, $zero, 5`, `lui`)
 * are roots like any computed value. Always returns 1. */
#define JT_ROOT_ZERO (JT_PATH + 1)

static int value_root(jt_path *p, int from, int floor, int reg,
                      int *root_at, int *root_reg, int64_t *delta) {
    int64_t d = 0;
    int k = from + 1;
    for (; k <= floor; k++) {
        a_insn in;
        const int s = path_get(p, k, &in);
        if (s < 0) break;
        if (!s || gpr_written(&in) != reg) continue;
        if (in.op == A_ADDIU) { d += in.imm; reg = in.rs; }
        else if ((in.op == A_OR || in.op == A_ADDU) && (in.rs == 0 || in.rt == 0))
            reg = in.rs ? in.rs : in.rt;
        else break;                             /* computed here */
        if (!reg) { k = JT_ROOT_ZERO; break; }
    }
    *root_at = k; *root_reg = reg; *delta = d;
    return 1;
}

/* The reachable indexes of a table indexed by `sll $idx, $key, 2` at step
 * `sll_k`: sets [*lo, *lo + *count) and returns 1, or returns 0.
 *
 * SNC bounds a switch whose cases start at 0 with `sltiu $t, $key, N`, and
 * one whose cases start at LO with a signed range test and a bias:
 *
 *     slti  $a1, $a0, LO          lower bound
 *     bne   $a1, $zero, default
 *     slti  $a1, $a0, HI          upper bound (in the bne's delay slot)
 *     beq   $a1, $zero, default
 *     addiu $a0, $a0, -LO
 *     sll   $a0, $a0, 2
 *
 * or bounds a biased copy (`addiu $a1, $a0, -1; sltiu $a1, $a1, 16; ...;
 * sll $a0, $a0, 2`). So the key and every compared register are traced back
 * to the value they are built from (value_root); a test applies when it is
 * the same value, and its constant is corrected by the two offsets. Every
 * test on the path is a fact about the key, so the tightest of each kind is
 * taken:
 *
 *   upper  slti/sltiu whose result the path requires set (compare_test);
 *          required. sltiu also bounds from below: its operand is >= 0.
 *   lower  slti whose result the path requires clear, and bltz/blez not
 *          taken (bgez/bgtz taken): operand >= 0 / >= 1.
 *
 * With only a signed upper bound and no lower bound in sight, the lowest
 * reachable index is taken to be 0: SNC biases a signed range so its lowest
 * case is index 0 and tests the lower bound on a path before this one. A
 * lowest index below 0 means the code can read before its table: rejected.
 * Bounds that contradict each other mean no index gets through on this
 * path: *count is 0. */
static int switch_bound(jt_path *p, int sll_k, int key, int64_t *lo_out, int64_t *count_out) {
    const int floor = sll_k + JT_VALUE_WINDOW;
    int kat, kreg; int64_t kd;
    if (!value_root(p, sll_k, floor, key, &kat, &kreg, &kd)) return 0;
    int have_hi = 0, have_lo = 0;
    int64_t hi = 0, lo = 0;
    for (int k = sll_k + 1; k <= floor && k < kat; k++) {
        a_insn in;
        const int s = path_get(p, k, &in);
        if (s < 0) break;
        if (!s) continue;
        int bat, breg; int64_t bd;
        if (in.op == A_SLTIU || in.op == A_SLTI) {
            const int test = compare_test(p, k, sll_k, in.rt);
            /* An upper bound needs a positive constant (and sltiu's operand
             * is only known to be >= 0 when it bounds from above); a signed
             * lower bound may be any constant. */
            if (!test || (test > 0 && in.imm <= 0) || (test < 0 && in.op != A_SLTI)) continue;
            if (!value_root(p, k, floor, in.rs, &bat, &breg, &bd) || bat != kat || breg != kreg)
                continue;
            /* key = root + kd, compared = root + bd, so key = compared - bd + kd. */
            const int64_t at = (int64_t)in.imm - bd + kd;
            if (test > 0) {
                if (!have_hi || at < hi) hi = at;
                have_hi = 1;
                if (in.op == A_SLTIU && (!have_lo || kd - bd > lo)) { lo = kd - bd; have_lo = 1; }
            } else {
                if (!have_lo || at > lo) lo = at;
                have_lo = 1;
            }
            continue;
        }
        if (in.is_branch &&
            (in.op == A_BLTZ || in.op == A_BLTZL || in.op == A_BGEZ || in.op == A_BGEZL ||
             in.op == A_BLEZ || in.op == A_BLEZL || in.op == A_BGTZ || in.op == A_BGTZL)) {
            const int taken = p->taken[k];
            int64_t min;
            if ((!taken && (in.op == A_BLTZ || in.op == A_BLTZL)) ||
                (taken && (in.op == A_BGEZ || in.op == A_BGEZL)))      min = 0;
            else if ((!taken && (in.op == A_BLEZ || in.op == A_BLEZL)) ||
                     (taken && (in.op == A_BGTZ || in.op == A_BGTZL))) min = 1;
            else continue;
            if (!value_root(p, k, floor, in.rs, &bat, &breg, &bd) || bat != kat || breg != kreg)
                continue;
            const int64_t at = min - bd + kd;
            if (!have_lo || at > lo) lo = at;
            have_lo = 1;
        }
    }
    if (!have_hi) return 0;
    if (!have_lo) lo = 0;
    if (hi <= lo) { *lo_out = 0; *count_out = 0; return 1; }
    if (lo < 0 || hi - lo > JT_MAX_ENTRIES) return 0;
    *lo_out = lo;
    *count_out = hi - lo;
    return 1;
}

static int in_scan(const a_analysis *an, uint32_t addr) {
    const uint32_t sbase = an->scan_size ? an->scan_base : an->base;
    const uint32_t ssize = an->scan_size ? an->scan_size : an->size;
    return addr >= sbase && addr - sbase < ssize && (addr & 3) == 0 && a_in_range(an, addr);
}

/* Match the table load and bound behind the jr at `site` along path `p`.
 * Fills table/lo/count of *jt (see a_jtable) and returns 1, or returns 0.
 * A count of 0 means no index reaches the jr along this path.
 *
 *   lw   $rN, lo($base)     the jr's register, loaded from the table
 *   addu $base, $x, $y      one of $x/$y is the table's high half (lui),
 *                           or lui+addiu/ori; the other is the scaled index
 *   sll  $idx, $key, 2      the scaling
 *   the bound               see switch_bound
 *
 * SNC writes the table's low half as the lw offset (`lui $at; addu $at,
 * $at, $idx; lw $at, lo($at)`); other compilers fold it into an addiu and
 * use offset 0. Both are covered because the lw offset is always added. */
static int match_table(jt_path *p, uint32_t site, a_jtable *jt) {
    const a_analysis *an = p->an;
    a_insn jr;
    a_decode(fetch(an, site), site, &jr);

    a_insn lw;
    const int lw_k = last_writer(p, -1, jr.rs, &lw);
    if (lw_k < 0 || lw.op != A_LW) return 0;

    a_insn add;
    const int add_k = last_writer(p, lw_k, lw.rs, &add);
    if (add_k < 0 || add.op != A_ADDU) return 0;
    uint32_t base;
    int idx;
    if (const_before(p, add_k, add.rs, &base))      idx = add.rt;
    else if (const_before(p, add_k, add.rt, &base)) idx = add.rs;
    else return 0;

    a_insn sc;
    const int sll_k = last_writer(p, add_k, idx, &sc);
    if (sll_k < 0 || sc.op != A_SLL || sc.sa != 2) return 0;
    int64_t lo, count;
    if (!switch_bound(p, sll_k, sc.rt, &lo, &count)) return 0;

    jt->site = site;
    jt->table = base + (uint32_t)lw.imm + 4u * (uint32_t)lo;
    jt->lo = (uint32_t)lo;
    jt->count = (uint32_t)count;
    return 1;
}

/* Recover the table behind the `jr` at `site` along its fall-through path.
 * On success appends its entries to `out`, fills *jt (all but `first`) and
 * returns the count; otherwise returns 0 and leaves `out` as it was. Every
 * reachable entry must be an instruction in the scan range (.text), or the
 * whole table is rejected. */
static int resolve_jump_table(const a_analysis *an, uint32_t site, u32list *out, a_jtable *jt) {
    if (!an->image) return 0;
    jt_path p;
    build_path(an, site, NULL, 0, &p);
    if (!match_table(&p, site, jt) || jt->count < 2) return 0;

    const int start = out->n;
    for (uint32_t i = 0; i < jt->count; i++) {
        uint32_t entry = image_read32(an, jt->table + i * 4);
        a_insn probe;
        if (!in_scan(an, entry) || !a_decode(fetch(an, entry), entry, &probe) ||
            probe.op == A_INVALID || u32_push(out, entry) != 0) {
            out->n = start;
            return 0;
        }
    }
    return (int)jt->count;
}

/* ---- the walk ------------------------------------------------------------ */

typedef struct jtlist { a_jtable *v; int n, cap; } jtlist;

static int jt_push(jtlist *l, a_jtable t) {
    if (l->n == l->cap) {
        int cap = l->cap ? l->cap * 2 : 64;
        a_jtable *v = (a_jtable *)realloc(l->v, (size_t)cap * sizeof *v);
        if (!v) return -1;
        l->v = v;
        l->cap = cap;
    }
    l->v[l->n++] = t;
    return 0;
}

typedef struct {
    a_analysis *an;
    uint8_t    *seen;
    /* Every address known to be a function entry, marked before any walking
     * starts. Knowing the full entry set up front is what lets a walk tell a
     * tail call from a local jump, and lets it stop when it runs off the end
     * of one function into the start of the next. */
    uint8_t    *entry_map;
    /* Ownership of each word, and the entry of the function currently being
     * traced. Recorded as the walk proceeds so the emitter knows exactly which
     * instructions belong to which function. */
    uint32_t   *owner;
    uint32_t    cur_owner;
    u32list    *func_queue;
    u32list    *imports;
    u32list    *indirects;
    /* Branch targets already claimed by an earlier walk; promoted to entries
     * between rounds so they become reachable through dispatch. */
    u32list    *cross;
    /* Switch tables recovered so far, and their targets. */
    jtlist     *jtables;
    u32list    *jt_targets;
    /* Per word: a jr whose table failed check_tables, left unresolved. */
    uint8_t    *no_table;
} walk_ctx;

static int is_known_entry(const walk_ctx *c, uint32_t addr) {
    if (!a_in_range(c->an, addr)) return 0;
    return c->entry_map[word_index(c->an, addr)] != 0;
}

/* Record a call target. Import thunks are boundaries, not functions to walk
 * into: they get patched at load time and their contents in the file are
 * meaningless. Everything else is a function entry. */
static void note_call(walk_ctx *c, uint32_t target) {
    if (is_import_stub(c->an, target)) {
        if (!u32_contains(c->imports, target)) u32_push(c->imports, target);
        return;
    }
    if (!a_in_range(c->an, target)) return;
    u32_push(c->func_queue, target);
}

/* Walk one function from `entry`, filling in `out`. Returns 0 if anything was
 * decoded, -1 if the entry was unusable. */
static int trace_function(walk_ctx *c, uint32_t entry, a_func *out) {
    a_analysis *an = c->an;
    if (!a_in_range(an, entry)) return -1;

    u32list blocks = { 0 };
    if (u32_push(&blocks, entry) != 0) return -1;

    uint32_t end = entry;
    uint32_t lo  = entry;
    uint32_t insns = 0;
    unsigned has_return = 0, has_indirect = 0, has_vfpu = 0;

    /* Instructions visited by *this* function, so a block shared with an
     * earlier function does not get walked twice here but is still counted
     * once globally. */
    while (blocks.n) {
        uint32_t a = blocks.v[--blocks.n];
        int first = 1;                              /* at the block's start */

        while (a_in_range(an, a)) {
            uint32_t idx = word_index(an, a);
            if (c->seen[idx] & SEEN_CODE) {         /* already walked */
                /* By another function, and not an entry: this one runs on into
                 * that function's code (falling past a branch or a call whose
                 * delay slot is that function's entry, say). The emitter can
                 * only continue there through an entry, so it is promoted,
                 * exactly like a branch into another function's block. */
                if (c->owner[idx] != c->cur_owner && !c->entry_map[idx]) {
                    u32_push(c->cross, a);
                } else if (first && c->owner[idx] == c->cur_owner && a_in_range(an, a - 4)) {
                    /* A branch into one of this function's delay slots, walked
                     * with its transfer. Arriving by the branch, the slot runs
                     * as an ordinary instruction and execution carries on after
                     * it -- which nothing else may reach (after a `b` or a
                     * `j`, say). */
                    a_insn prev;
                    a_decode(fetch(an, a - 4), a - 4, &prev);
                    if (prev.has_delay_slot) { first = 0; a += 4; continue; }
                }
                break;
            }
            first = 0;

            /* Arriving at a different function's entry means we walked off the
             * end of this one — a tail call, or a function that ends without a
             * visible return. Stop; that address is walked as its own function.
             * Without this a single trace swallows every function that follows
             * it in address order. */
            if (a != entry && is_known_entry(c, a)) break;

            a_insn in;
            a_decode(fetch(an, a), a, &in);
            c->seen[idx] |= SEEN_CODE;
            c->owner[idx] = c->cur_owner;
            insns++;
            an->insns++;
            if (a + 4 > end) end = a + 4;
            if (a < lo) lo = a;

            if (in.op == A_INVALID) {
                /* Almost always data reached by a bad path. Stop this trace
                 * rather than manufacturing instructions out of it. */
                an->invalid++;
                break;
            }
            if (in.op == A_VFPU_UNKNOWN) { an->vfpu++; has_vfpu = 1; }

            /* Every branch and jump has a delay slot that executes before
             * control transfers, so it is always part of this function and is
             * always visited — including on paths that leave here. */
            if (in.has_delay_slot) {
                uint32_t d = a + 4;
                if (a_in_range(an, d)) {
                    uint32_t didx = word_index(an, d);
                    /* A slot that is another function's entry is left for that
                     * function's walk: claimed here, it would be seen already
                     * when that walk starts, which then finds nothing. The
                     * emitter runs a slot with its transfer whoever owns it. */
                    if (!(c->seen[didx] & SEEN_CODE) && !(d != entry && is_known_entry(c, d))) {
                        a_insn din;
                        a_decode(fetch(an, d), d, &din);
                        c->seen[didx] |= SEEN_CODE;
                        c->owner[didx] = c->cur_owner;
                        insns++;
                        an->insns++;
                        if (din.op == A_VFPU_UNKNOWN) { an->vfpu++; has_vfpu = 1; }
                        if (din.op == A_INVALID) an->invalid++;
                    }
                    if (d + 4 > end) end = d + 4;
                    if (d < lo) lo = d;
                }
            }

            /* A branch or tail-jump into the import thunks is a call into
             * firmware by another name. It cannot be followed — the thunks are
             * patched at load time and their file contents are meaningless —
             * but it must be *recorded*, because the emitter decides what is an
             * import purely by address range. If the walk does not record it,
             * the generated code references psp_import_<addr> and the imports
             * file never defines it: a link error, and one that only shows up
             * on a module whose imports are not all reached by `jal`.
             *
             * The stub region sits immediately after `.text`, so it is outside
             * the analysed range and the branch/jump paths below skip it
             * silently. Hence the explicit check here. */
            if (in.has_target && !in.is_indirect && is_import_stub(an, in.target)) {
                if (!u32_contains(c->imports, in.target))
                    u32_push(c->imports, in.target);
            }

            if (in.is_call) {
                /* jal / jalr / the *al REGIMM forms. Direct calls give us a
                 * new function; indirect ones we cannot resolve statically. */
                if (in.has_target && !in.is_indirect) note_call(c, in.target);
                else if (in.is_indirect) has_indirect = 1;
                a += 8;                     /* past the delay slot; calls return */
                continue;
            }

            if (in.is_return) { has_return = 1; break; }

            if (in.is_indirect) {
                /* `jr $rN`. A recovered switch table turns it into ordinary
                 * control flow: its targets are blocks of this function,
                 * claimed exactly like conditional-branch targets. */
                const int first = c->jt_targets->n;
                a_jtable jt;
                const int n = c->no_table[idx] ? 0
                            : resolve_jump_table(an, a, c->jt_targets, &jt);
                if (n > 0) {
                    jt.first = (uint32_t)first;
                    const int nblocks = blocks.n, ncross = c->cross->n;
                    const int pushed = jt_push(c->jtables, jt) == 0;
                    int ok = pushed;
                    for (int k = first; ok && k < first + n; k++) {
                        const uint32_t t = c->jt_targets->v[k];
                        const uint32_t ti = word_index(an, t);
                        if ((c->seen[ti] & SEEN_CODE) && !c->entry_map[ti])
                            ok = u32_push(c->cross, t) == 0;
                        else
                            ok = u32_push(&blocks, t) == 0;
                    }
                    if (ok) break;
                    /* Out of memory: a table whose targets are not all walked
                     * is not recorded; the jr stays unresolved. */
                    if (pushed) c->jtables->n--;
                    blocks.n = nblocks;
                    c->cross->n = ncross;
                    c->jt_targets->n = first;
                }
                /* Unresolved: a computed jump we cannot follow. Record the site
                 * and stop; the sites are counted rather than silently
                 * dropped. */
                has_indirect = 1;
                if (!u32_contains(c->indirects, a)) u32_push(c->indirects, a);
                break;
            }

            if (in.is_branch) {
                /* Conditional: the target is another block of this function,
                 * and execution also falls through past the delay slot. */
                if (in.has_target && a_in_range(an, in.target)) {
                    uint32_t ti = word_index(an, in.target);
                    if ((c->seen[ti] & SEEN_CODE) && !c->entry_map[ti]) {
                        /* Already claimed by an earlier walk, and not an entry.
                         * Two functions therefore share this block, which is a
                         * walk-order artefact rather than a property of the
                         * code -- whichever was traced first took it.
                         *
                         * The emitter cannot express a branch into another C
                         * function's middle; it degrades to a dispatch, which
                         * then misses because the address is not an entry.
                         * Promoting it to one makes it reachable. Over-split
                         * is the safe direction; unreachable is not. */
                        u32_push(c->cross, in.target);
                    } else {
                        u32_push(&blocks, in.target);
                    }
                }
                /* An always-taken branch (`b`) has no not-taken path: what
                 * follows its delay slot is only reached through its own
                 * labels, and walking on into it would claim it for this
                 * function -- the case blocks of a switch, typically. */
                if (a_always_taken(&in) && !in.is_call) break;
                a += 8;
                continue;
            }

            if (in.is_jump) {
                /* Unconditional `j`. MIPS compilers emit `b` — which is
                 * `beq $zero, $zero` and decodes as a branch — for local
                 * unconditional flow inside a function. A `j` is therefore
                 * almost always either a tail call or a loop back-edge, and
                 * the direction tells them apart:
                 *
                 *   backward -> a loop back-edge, a block of this function
                 *   forward  -> a tail call, the entry of another function
                 *
                 * Observed in this module as two-instruction thunks:
                 *     j 0x00091E38 ; addiu $a0, $zero, 2
                 * Treating that forward `j` as local flow pulls the callee
                 * into the thunk and, transitively, swallows most of .text
                 * into one 121 KB "function".
                 *
                 * Over-splitting is the safe direction to err: a wrongly split
                 * function is still correct code reached through the dispatch
                 * table, whereas a wrongly merged one has bogus boundaries. */
                if (in.has_target && a_in_range(an, in.target)) {
                    uint32_t bi = word_index(an, in.target);
                    if (in.target <= a && a != entry && !is_known_entry(c, in.target)) {
                        /* Backward `j` -- a loop back-edge into this function's
                         * own blocks, unless another walk already claimed the
                         * target. Then the two functions share a block, exactly
                         * as for a conditional branch, and for the same reason:
                         * walk order, not a property of the code.
                         *
                         * The emitter cannot express a jump into another C
                         * function's middle. It degrades to a dispatch, which
                         * misses because the address was never made an entry --
                         * observed as `j 0x0000E588` inside the allocator,
                         * emitted as a dispatch to an undiscovered address
                         * and missing at run time.
                         *
                         * The branch path above has always promoted these. The
                         * jump path did not, so a whole class of shared block
                         * stayed unreachable. */
                        if ((c->seen[bi] & SEEN_CODE) && !c->entry_map[bi])
                            u32_push(c->cross, in.target);
                        else
                            u32_push(&blocks, in.target);
                    } else {
                        uint32_t ti = word_index(an, in.target);
                        if (!c->entry_map[ti]) {
                            c->entry_map[ti] = 1;
                            u32_push(c->func_queue, in.target); /* tail call */
                        }
                    }
                }
                break;
            }

            a += 4;
        }
    }

    free(blocks.v);

    if (!insns) return -1;

    memset(out, 0, sizeof *out);
    out->addr = entry;
    out->start = lo;
    out->end = end;
    out->insns = insns;
    out->has_return = has_return;
    out->has_indirect = has_indirect;
    out->has_vfpu = has_vfpu;
    return 0;
}

static int cmp_func(const void *a, const void *b) {
    uint32_t x = ((const a_func *)a)->addr, y = ((const a_func *)b)->addr;
    return (x > y) - (x < y);
}

static int cmp_jtable(const void *a, const void *b) {
    uint32_t x = ((const a_jtable *)a)->site, y = ((const a_jtable *)b)->site;
    return (x > y) - (x < y);
}

const a_jtable *a_find_jtable(const a_analysis *an, uint32_t site) {
    int lo = 0, hi = an->njtables - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (an->jtables[mid].site == site) return &an->jtables[mid];
        if (an->jtables[mid].site < site) lo = mid + 1; else hi = mid - 1;
    }
    return NULL;
}

/* ---- every way into a table's code ------------------------------------------ */

typedef struct { jt_edge *v; int n, cap; } edgelist;

static int edge_push(edgelist *l, jt_edge e) {
    if (l->n == l->cap) {
        int cap = l->cap ? l->cap * 2 : 1024;
        jt_edge *v = (jt_edge *)realloc(l->v, (size_t)cap * sizeof *v);
        if (!v) return -1;
        l->v = v;
        l->cap = cap;
    }
    l->v[l->n++] = e;
    return 0;
}

static int cmp_edge(const void *a, const void *b) {
    uint32_t x = ((const jt_edge *)a)->at, y = ((const jt_edge *)b)->at;
    return (x > y) - (x < y);
}

/* Is the match `r` inside the table `ref`: the same table and a subset of
 * its indexes (often the same bound recomputed, sometimes a tighter one),
 * or no index at all (the path's own tests rule every index out)? */
static int within(const a_jtable *r, const a_jtable *ref) {
    if (r->count == 0) return 1;
    return r->table - 4u * r->lo == ref->table - 4u * ref->lo &&
           r->lo >= ref->lo && r->lo + r->count <= ref->lo + ref->count;
}

typedef struct {
    const a_analysis *an;
    const uint8_t    *entry_map;
    const edgelist   *edges;            /* sorted by target */
    long              calls;            /* check_paths calls for this table */
} jt_check;

static int edge_in_chain(const jt_edge *chain, int n, const jt_edge *e) {
    for (int i = 0; i < n; i++)
        if (chain[i].kind == e->kind && chain[i].at == e->at && chain[i].from == e->from)
            return 1;
    return 0;
}

/* Check the paths into the jr at `site` that take the edges `chain` (the
 * first nearest the jr) and then any further edges: does every match stay
 * within `ref`? A further edge can land on any step the match read from the
 * last edge's code on (an edge into step k changes step k + 1). Each is
 * followed in turn, up to JT_EDGE_DEPTH edges in all; a path needing more,
 * or one the match fails on, fails the check. An edge already on the chain
 * is not taken again: the path would run through the same loop twice. A
 * table whose chains need more than JT_CHECK_BUDGET calls fails too. */
static int check_paths(jt_check *ck, uint32_t site, const a_jtable *ref,
                       jt_edge *chain, int nchain) {
    const a_analysis *an = ck->an;
    if (++ck->calls > JT_CHECK_BUDGET) return 0;        /* too many to check */
    jt_path p;
    a_jtable r;
    build_path(an, site, chain, nchain, &p);
    if (!match_table(&p, site, &r)) return 0;
    if (nchain && !within(&r, ref)) return 0;
    if (p.edge_from >= JT_PATH) return 1;       /* nothing past the last edge */

    /* The match read steps 0..deepest (or ran off the path's end). */
    const int last = p.deepest < p.n ? p.deepest : p.n;
    for (int k = p.edge_from; k < last; k++) {
        const uint32_t at = k < 0 ? site : p.addr[k];
        jt_edge own = { JT_EDGE_ENTRY, at, 0, k };
        /* Entering at the step where the path ends anyway changes nothing. */
        const int is_entry = ck->entry_map[word_index(an, at)] == ENTRY_OWN &&
                             !(k == p.n - 1 && p.n < JT_PATH);
        int lo = 0, hi = ck->edges->n;          /* first edge landing at `at` */
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (ck->edges->v[mid].at < at) lo = mid + 1; else hi = mid;
        }
        for (int m = lo - (is_entry ? 1 : 0); m < ck->edges->n; m++) {
            jt_edge e;
            if (m < lo) e = own;                /* the entry, first */
            else if (ck->edges->v[m].at == at) { e = ck->edges->v[m]; e.step = k; }
            else break;
            if (edge_in_chain(chain, nchain, &e)) continue;
            if (nchain == JT_EDGE_DEPTH) return 0;      /* too deep to check */
            chain[nchain] = e;
            if (!check_paths(ck, site, ref, chain, nchain + 1)) return 0;
        }
    }
    return 1;
}

/* A table is matched along the fall-through path into its jr, but code on
 * that path can also be reached part-way: by a branch or jump landing there,
 * as a case of another table, or as a function entry. Each of those starts
 * another path into the jr, and the bound only holds if the match along
 * every path stays within the table -- a merged tail whose incoming branch
 * carries a stale or different $t would let other indexes through. (In this
 * game the edges that land there recompute the identical compare, in their
 * delay slot or a few instructions earlier.)
 *
 * So after each walk, when every edge out of reached code is known, each
 * table is matched again along every path made of up to JT_EDGE_DEPTH edges
 * from reached code (check_paths). What is guaranteed: for every path into
 * the jr that enters the code the match reads through at most JT_EDGE_DEPTH
 * such edges (without repeating one), the match succeeds and its reachable
 * indexes are within the table's; otherwise the table is rejected,
 * including when a path would need more edges or the table's chains exceed
 * JT_CHECK_BUDGET checks. Code that is not reached by discovery contributes
 * no edges. An entry counts as an edge from unknown
 * code only when it is one in its own right; an address promoted from a
 * cross-function branch is an entry only for the emitter's sake, and its
 * branches are edges here already.
 *
 * Rejected sites are marked in `no_table`, and their reachable targets are
 * made entries: the table is real, only its bound is uncertain, so the jr's
 * dispatch must still find them. Returns how many tables were rejected; the
 * caller re-walks, so their jrs become unresolved. */
static int check_tables(const a_analysis *an, const uint8_t *seen, uint8_t *entry_map,
                        const jtlist *jts, const u32list *jtt, uint8_t *no_table) {
    edgelist el = { 0 };
    int ok_mem = 1;
    const uint32_t nwords = an->size >> 2;
    for (uint32_t i = 0; ok_mem && i < nwords; i++) {
        if (!(seen[i] & SEEN_CODE)) continue;
        const uint32_t a = an->base + i * 4;
        a_insn in;
        a_decode(fetch(an, a), a, &in);
        if (!in.has_target || in.is_indirect || !(in.is_branch || in.is_jump)) continue;
        if (!a_in_range(an, in.target)) continue;
        jt_edge e = { in.is_call ? JT_EDGE_ENTRY : JT_EDGE_BRANCH, in.target, a, 0 };
        ok_mem = edge_push(&el, e) == 0;
    }
    for (int j = 0; ok_mem && j < jts->n; j++)
        for (uint32_t k = 0; ok_mem && k < jts->v[j].count; k++) {
            jt_edge e = { JT_EDGE_BRANCH, jtt->v[jts->v[j].first + k], jts->v[j].site, 0 };
            ok_mem = edge_push(&el, e) == 0;
        }
    qsort(el.v, (size_t)el.n, sizeof *el.v, cmp_edge);

    jt_check ck = { an, entry_map, &el, 0 };
    int rejected = 0;
    for (int j = 0; j < jts->n; j++) {
        const a_jtable *t = &jts->v[j];
        jt_path p;
        a_jtable ref;
        jt_edge chain[JT_EDGE_DEPTH];
        build_path(an, t->site, NULL, 0, &p);
        ck.calls = 0;
        if (ok_mem && match_table(&p, t->site, &ref) && check_paths(&ck, t->site, &ref, chain, 0))
            continue;
        no_table[word_index(an, t->site)] = 1;
        for (uint32_t k = 0; k < t->count; k++) {
            const uint32_t ti = word_index(an, jtt->v[t->first + k]);
            if (!entry_map[ti]) entry_map[ti] = ENTRY_OWN;
        }
        rejected++;
    }
    free(el.v);
    return rejected;
}

int a_discover(a_analysis *an, const uint32_t *seeds, int nseeds) {
    an->funcs = NULL; an->nfuncs = 0;
    an->imports = NULL; an->nimports = 0;
    an->indirects = NULL; an->nindirects = 0;
    an->insns = an->vfpu = an->invalid = 0;
    an->ntables = an->ntable_targets = 0;
    an->jtables = NULL; an->njtables = 0;
    an->jt_targets = NULL; an->njt_targets = 0;
    an->bytes_reached = 0;
    /* The import stubs sit at the end of .text, so their extent is the best
     * available executable bound when no section header gives one. */
    if (!an->text_size && an->stub_addr && an->stub_size)
        an->text_size = an->stub_addr + an->stub_size - an->base;

    const uint32_t nwords = an->size >> 2;
    uint8_t *seen = (uint8_t *)calloc(nwords ? nwords : 1, 1);
    if (!seen) return -1;

    uint8_t *entry_map = (uint8_t *)calloc(nwords ? nwords : 1, 1);
    if (!entry_map) { free(seen); return -1; }

    uint32_t *owner = (uint32_t *)malloc((nwords ? nwords : 1) * sizeof *owner);
    if (!owner) { free(seen); free(entry_map); return -1; }

    uint8_t *no_table = (uint8_t *)calloc(nwords ? nwords : 1, 1);
    if (!no_table) { free(seen); free(entry_map); free(owner); return -1; }
    for (uint32_t i = 0; i < nwords; i++) owner[i] = A_NO_OWNER;

    u32list queue = { 0 }, imports = { 0 }, indirects = { 0 }, cross = { 0 };
    u32list jt_targets = { 0 };
    jtlist jtables = { 0 };
    walk_ctx ctx = { an, seen, entry_map, owner, A_NO_OWNER,
                     &queue, &imports, &indirects, &cross,
                     &jtables, &jt_targets, no_table };

    /* Pass 1: establish the complete set of function entries before walking
     * anything. Both the tail-call test and the ran-off-the-end test need to
     * ask "is this somebody's entry?", and neither can answer correctly if
     * entries are still being discovered as the walk proceeds. */
    for (int i = 0; i < nseeds; i++) {
        if (!a_in_range(an, seeds[i])) continue;
        entry_map[word_index(an, seeds[i])] = 1;
        u32_push(&queue, seeds[i]);
    }

    /* Harvest `jal` targets by linear scan. See the note in analyze.h for why
     * this is required rather than a nicety on PSP. Duplicates are harmless —
     * the entry map makes the second sighting a no-op — so no membership test
     * is done here, which keeps this O(n) rather than O(n^2). */
    if (an->scan_calls) {
        /* Restricted to the scan range -- see the note in analyze.h. Falls back
         * to the whole extent if the caller did not set one. */
        uint32_t sbase = an->scan_size ? an->scan_base : an->base;
        uint32_t ssize = an->scan_size ? an->scan_size : an->size;
        for (uint32_t off = 0; off + 4 <= ssize; off += 4) {
            uint32_t a = sbase + off;
            if (!a_in_range(an, a)) continue;
            a_insn in;
            a_decode(fetch(an, a), a, &in);
            if (in.op != A_JAL || !in.has_target) continue;
            if (!a_in_range(an, in.target) || is_import_stub(an, in.target)) continue;

            uint32_t ti = word_index(an, in.target);
            if (entry_map[ti]) continue;
            entry_map[ti] = 1;
            u32_push(&queue, in.target);
        }
    }

    a_func *funcs = NULL;
    int nfuncs = 0, cap = 0;

    /* Iterate to a fixpoint. Switch tables are resolved inside the walk, as
     * it reaches each `jr`, so their targets are already blocks of the right
     * function; what a round adds is cross-function targets (branch, jump or
     * table targets another walk had already claimed), promoted to entries,
     * and tables rejected by check_tables, which needs the whole walk's
     * edges. Each re-walk can expose more cross-function targets that the
     * previous walk order hid, so a fixed two rounds leaves some unpromoted
     * -- which shows up as a dispatch miss at run time. The cap on promoting
     * rounds is a termination guarantee, not an expected limit: entries only
     * ever accumulate, so this converges in a handful of rounds. Rejections
     * always get their re-walk, and terminate because a rejected site never
     * gets a table again. When nothing changes, the last walk's results
     * (functions, unresolved sites, tables) stand as they are. */
    for (int round = 0; ; round++) {
    if (round > 0) {
        int added = 0;
        for (int i = 0; round < 8 && i < cross.n; i++) {
            uint32_t t = cross.v[i];
            if (!a_in_range(an, t)) continue;
            uint32_t ti = word_index(an, t);
            if (entry_map[ti]) continue;
            entry_map[ti] = ENTRY_PROMOTED;
            added++;
        }
        cross.n = 0;
        const int rejected = check_tables(an, seen, entry_map, &jtables, &jt_targets, no_table);
        if (!added && !rejected) break;

        /* Re-walk everything from a clean slate rather than walking only the
         * new entries.
         *
         * A promoted address was, by definition, already claimed in round one
         * -- that is why it showed up as a cross-function target. Walking it
         * again in isolation does nothing: the very first instruction is
         * already marked as code, so the trace stops immediately and produces
         * an empty function. The owning function also still contains the code,
         * so simply stealing it would leave that function with a hole.
         *
         * Discarding the round-one result and re-walking with the complete
         * entry set produces correct, non-overlapping boundaries in one pass,
         * because every split point is now known before any walking starts --
         * which is the same reason the entry map is built up front to begin
         * with. One extra walk is cheap; reconciling overlapping ownership
         * afterwards is not. */
        memset(seen, 0, nwords);
        for (uint32_t i = 0; i < nwords; i++) owner[i] = A_NO_OWNER;
        an->insns = an->vfpu = an->invalid = 0;
        nfuncs = 0;
        indirects.n = 0;
        jtables.n = 0;
        jt_targets.n = 0;

        for (uint32_t i = 0; i < nwords; i++)
            if (entry_map[i]) u32_push(&queue, an->base + i * 4);
        if (!queue.n) break;
    }

    /* The queue grows as calls are found; this terminates because every
     * function entry is marked and never walked twice, and there are finitely
     * many words. */
    while (queue.n) {
        uint32_t entry = queue.v[--queue.n];
        if (!a_in_range(an, entry)) continue;

        uint32_t idx = word_index(an, entry);
        if (seen[idx] & SEEN_ENTRY) continue;
        seen[idx] |= SEEN_ENTRY;

        a_func f;
        ctx.cur_owner = entry;
        if (trace_function(&ctx, entry, &f) != 0) continue;

        if (nfuncs == cap) {
            int ncap = cap ? cap * 2 : 512;
            a_func *nf = (a_func *)realloc(funcs, (size_t)ncap * sizeof *nf);
            if (!nf) break;
            funcs = nf;
            cap = ncap;
        }
        funcs[nfuncs++] = f;
    }
    }   /* rounds */

    for (uint32_t i = 0; i < nwords; i++)
        if (seen[i] & SEEN_CODE) an->bytes_reached += 4;

    /* Sorted by site for a_find_jtable. */
    qsort(jtables.v, (size_t)jtables.n, sizeof *jtables.v, cmp_jtable);
    an->jtables = jtables.v;
    an->njtables = jtables.n;
    an->jt_targets = jt_targets.v;
    an->njt_targets = jt_targets.n;
    an->ntables = jtables.n;
    an->ntable_targets = jt_targets.n;

    qsort(funcs, (size_t)nfuncs, sizeof *funcs, cmp_func);

    an->funcs = funcs;
    an->nfuncs = nfuncs;
    an->imports = imports.v;
    an->nimports = imports.n;
    an->indirects = indirects.v;
    an->nindirects = indirects.n;

    an->owner = owner;
    an->nwords = nwords;

    free(no_table);
    free(cross.v);
    free(queue.v);
    free(seen);
    free(entry_map);
    return 0;
}

static int scan_data(const a_analysis *an, const uint8_t *region, uint32_t region_len,
                     uint32_t region_addr, uint32_t *out_v, a_data_seed *out_s, int max) {
    int found = 0;

    for (uint32_t off = 0; off + 4 <= region_len; off += 4) {
        uint32_t v = (uint32_t)region[off] | ((uint32_t)region[off + 1] << 8) |
                     ((uint32_t)region[off + 2] << 16) | ((uint32_t)region[off + 3] << 24);

        if (!a_in_range(an, v)) continue;        /* inside the code extent */

        /* Point at something that decodes. A pointer into the middle of a data
         * table would usually fail this; a genuine function entry never does. */
        a_insn in;
        if (!a_decode(fetch(an, v), v, &in) || in.op == A_INVALID) continue;

        if (found < max) {
            if (out_v) out_v[found] = v;
            if (out_s) { out_s[found].value = v; out_s[found].source = region_addr + off; }
        }
        found++;
    }
    return found;
}

int a_scan_data_pointers(const a_analysis *an,
                         const uint8_t *region, uint32_t region_len,
                         uint32_t *out, int max) {
    return scan_data(an, region, region_len, 0, out, NULL, out ? max : 0);
}

int a_scan_data_seeds(const a_analysis *an, const uint8_t *region, uint32_t region_len,
                      uint32_t region_addr, a_data_seed *out, int max) {
    return scan_data(an, region, region_len, region_addr, NULL, out, out ? max : 0);
}

/* ---- pruning jump-table words from the data seeds ------------------------- */

#define PRUNE_PASSES 4

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int has_u32(const uint32_t *v, int n, uint32_t x) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (v[mid] == x) return 1;
        if (v[mid] < x) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

static int is_func_entry(const a_analysis *an, uint32_t a) {
    int lo = 0, hi = an->nfuncs - 1;              /* funcs are sorted by addr */
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (an->funcs[mid].addr == a) return 1;
        if (an->funcs[mid].addr < a) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

/* Is the data word at `src` one of a recovered table's reachable entries?
 * (Tables are sorted by site, not by address, so all are searched.) */
static int in_table(const a_analysis *an, uint32_t src) {
    for (int i = 0; i < an->njtables; i++) {
        const a_jtable *t = &an->jtables[i];
        if (src >= t->table && src - t->table < 4u * t->count) return 1;
    }
    return 0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Every (target, owner of the jr) pair of the recovered tables, sorted. */
static uint64_t *local_cases(const a_analysis *an, int *n) {
    uint64_t *v = (uint64_t *)malloc(((size_t)an->njt_targets + 1) * sizeof *v);
    *n = 0;
    if (!v) return NULL;
    for (int i = 0; i < an->njtables; i++) {
        const a_jtable *t = &an->jtables[i];
        const uint32_t own = an->owner[(t->site - an->base) >> 2];
        for (uint32_t k = 0; k < t->count; k++)
            v[(*n)++] = (uint64_t)an->jt_targets[t->first + k] << 32 | own;
    }
    qsort(v, (size_t)*n, sizeof *v, cmp_u64);
    return v;
}

/* Is the code at `v` still reached as something a switch can get to: a
 * function entry, or a block of the function that owns a table listing it? */
static int case_reached(const a_analysis *an, const uint64_t *lc, int nlc, uint32_t v) {
    if (is_func_entry(an, v)) return 1;
    if (!a_in_range(an, v)) return 0;
    const uint32_t own = an->owner[(v - an->base) >> 2];
    if (own == A_NO_OWNER) return 0;
    const uint64_t key = (uint64_t)v << 32 | own;
    int lo = 0, hi = nlc - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (lc[mid] == key) return 1;
        if (lc[mid] < key) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

int a_discover_pruned(a_analysis *an, const uint32_t *seeds, int nseeds,
                      const a_data_seed *ds, int nds, a_prune_stats *ps) {
    memset(ps, 0, sizeof *ps);
    const a_analysis in = *an;                  /* the inputs, for every run */
    uint32_t *other = (uint32_t *)malloc(((size_t)nseeds + 1) * sizeof *other);
    uint32_t *all = (uint32_t *)malloc(((size_t)nseeds + (size_t)nds + 1) * sizeof *all);
    uint8_t *drop = (uint8_t *)calloc((size_t)nds + 1, 1);
    uint8_t *keep = (uint8_t *)calloc((size_t)nds + 1, 1);   /* never drop again */
    int rc = -1;
    if (!other || !all || !drop || !keep) goto out;
    if (nseeds) memcpy(other, seeds, (size_t)nseeds * sizeof *other);
    qsort(other, (size_t)nseeds, sizeof *other, cmp_u32);

    /* By value: the scan finds the same address in several words, and a
     * value is dropped only if every word holding it is a table entry. */
    uint32_t *vals = (uint32_t *)malloc(((size_t)nds + 1) * sizeof *vals);
    if (!vals) goto out;
    for (;;) {
        int n = 0;
        for (int i = 0; i < nseeds; i++) all[n++] = seeds[i];
        for (int i = 0; i < nds; i++) if (!drop[i]) all[n++] = ds[i].value;
        if (ps->passes) a_analysis_free(an);
        *an = in;
        if (a_discover(an, all, n) != 0) { free(vals); goto out; }
        ps->passes++;

        /* Put back every dropped value whose code this run lost. */
        int restored = 0, nlc = 0;
        uint64_t *lc = local_cases(an, &nlc);
        if (!lc) { free(vals); goto out; }
        for (int i = 0; i < nds; i++) {
            if (!drop[i] || case_reached(an, lc, nlc, ds[i].value)) continue;
            for (int j = 0; j < nds; j++)
                if (ds[j].value == ds[i].value && drop[j]) {
                    drop[j] = 0; keep[j] = 1; restored++;
                }
        }
        free(lc);
        ps->restored += restored;

        /* Values this run's tables justify dropping. */
        int added = 0;
        if (ps->passes < PRUNE_PASSES) {
            int nv = 0;
            for (int i = 0; i < nds; i++)
                if (keep[i] || has_u32(other, nseeds, ds[i].value) || !in_table(an, ds[i].source))
                    vals[nv++] = ds[i].value;            /* held somewhere it must stay */
            qsort(vals, (size_t)nv, sizeof *vals, cmp_u32);
            for (int i = 0; i < nds; i++)
                if (!drop[i] && !has_u32(vals, nv, ds[i].value)) { drop[i] = 1; added++; }
        }
        if (!restored && !added) break;
    }
    free(vals);
    for (int i = 0; i < nds; i++) ps->dropped += drop[i];
    rc = 0;
out:
    free(other); free(all); free(drop); free(keep);
    return rc;
}

/* How far a low half may follow its lui. Compilers schedule the pair close
 * together; SNC's crt0 has it two instructions apart. */
#define PAIR_WINDOW 16

int a_scan_address_pairs(const a_analysis *an, uint32_t *out, int max) {
    const uint32_t sbase = an->scan_size ? an->scan_base : an->base;
    const uint32_t ssize = an->scan_size ? an->scan_size : an->size;
    int found = 0;

    for (uint32_t off = 0; off + 4 <= ssize; off += 4) {
        const uint32_t a = sbase + off;
        if (!a_in_range(an, a)) continue;
        a_insn hi;
        a_decode(fetch(an, a), a, &hi);
        if (hi.op != A_LUI || hi.rt == 0) continue;

        /* A lui in the delay slot of an unconditional jump feeds the jump's
         * target, not the code that follows it in memory. (A call's delay
         * slot does fall through, so calls are not skipped.) */
        if (off >= 4) {
            a_insn prev;
            a_decode(fetch(an, a - 4), a - 4, &prev);
            if (prev.is_jump && !prev.is_call) continue;
        }

        int stop_after = -1;          /* instructions left before control leaves */
        for (int i = 1; i <= PAIR_WINDOW && stop_after != 0; i++) {
            const uint32_t b = a + (uint32_t)i * 4;
            if (b - sbase >= ssize || !a_in_range(an, b)) break;
            if (stop_after > 0) stop_after--;
            a_insn lo;
            a_decode(fetch(an, b), b, &lo);
            if (lo.op == A_INVALID) break;               /* data, not code */

            if ((lo.op == A_ADDIU || lo.op == A_ORI) && lo.rs == hi.rt) {
                /* addiu's immediate is sign-extended and ori's zero-extended;
                 * the decoder already did that, so plain addition is right. */
                const uint32_t t = ((uint32_t)hi.imm << 16) + (uint32_t)lo.imm;
                a_insn probe;
                if (t >= sbase && t - sbase < ssize && (t & 3) == 0 &&
                    !is_import_stub(an, t) && a_in_range(an, t) &&
                    a_decode(fetch(an, t), t, &probe) && probe.op != A_INVALID) {
                    if (found < max && out) out[found] = t;
                    found++;
                }
                /* One lui can feed several low halves (lui; addiu a0; addiu
                 * a1), so keep looking unless this one replaced the high half. */
                if (lo.rt == hi.rt) break;
                continue;
            }
            if (gpr_written(&lo) == hi.rt) break;            /* high half gone */
            /* An unconditional jump (j, jr) ends the straight line after its
             * delay slot; a pair does not span into unrelated code. A call
             * returns, so it does not. `b` (beq $zero,$zero) is a branch, not
             * a jump: it is local flow, and a lui before an if/else dominates
             * both arms, so it is deliberately not a stop. */
            if (lo.is_jump && !lo.is_call)
                stop_after = 1;
        }
    }
    return found;
}

void a_analysis_free(a_analysis *an) {
    free(an->funcs);
    free(an->imports);
    free(an->indirects);
    free(an->owner);
    free(an->jtables);
    free(an->jt_targets);
    an->funcs = NULL;
    an->imports = NULL;
    an->indirects = NULL;
    an->owner = NULL;
    an->jtables = NULL;
    an->jt_targets = NULL;
    an->nfuncs = an->nimports = an->nindirects = 0;
    an->njtables = an->njt_targets = 0;
    an->ntables = an->ntable_targets = 0;
}
