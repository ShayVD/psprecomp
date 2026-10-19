/* SNC switch tables: recovered during discovery, case blocks kept inside the
 * function. Synthetic code only, no game data. */
#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BASE  0x08804000u
#define TABLE (BASE + 0x100u)        /* "rodata": in the image, past .text */

static uint32_t R(int op, int rs, int rt, int imm) {
    return (uint32_t)op << 26 | (uint32_t)rs << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
static uint32_t sltiu(int rt, int rs, int imm) { return R(0x0B, rs, rt, imm); }
static uint32_t beq(int rs, int rt, uint32_t pc, uint32_t target) {
    return R(0x04, rs, rt, (int)((target - (pc + 4)) >> 2));
}
static uint32_t sll(int rd, int rt, int sa) { return (uint32_t)rt << 16 | (uint32_t)rd << 11 | (uint32_t)sa << 6; }
static uint32_t lui(int rt, uint32_t imm) { return R(0x0F, 0, rt, (int)imm); }
static uint32_t addu(int rd, int rs, int rt) { return (uint32_t)rs << 21 | (uint32_t)rt << 16 | (uint32_t)rd << 11 | 0x21; }
static uint32_t lw(int rt, int imm, int base) { return R(0x23, base, rt, imm); }
static uint32_t jr(int rs) { return (uint32_t)rs << 21 | 0x08; }
static uint32_t jal(uint32_t target) { return 0x03u << 26 | ((target >> 2) & 0x03FFFFFFu); }
static uint32_t ori(int rt, int rs, int imm) { return R(0x0D, rs, rt, imm); }
static uint32_t slti(int rt, int rs, int imm) { return R(0x0A, rs, rt, imm); }
static uint32_t addiu(int rt, int rs, int imm) { return R(0x09, rs, rt, imm); }
static uint32_t or_(int rd, int rs, int rt) { return (uint32_t)rs << 21 | (uint32_t)rt << 16 | (uint32_t)rd << 11 | 0x25; }
static uint32_t bne(int rs, int rt, uint32_t pc, uint32_t target) {
    return R(0x05, rs, rt, (int)((target - (pc + 4)) >> 2));
}
static uint32_t beql(int rs, int rt, uint32_t pc, uint32_t target) {
    return R(0x14, rs, rt, (int)((target - (pc + 4)) >> 2));
}
static uint32_t blez(int rs, uint32_t pc, uint32_t target) {
    return R(0x06, rs, 0, (int)((target - (pc + 4)) >> 2));
}
static uint32_t bgez(int rs, uint32_t pc, uint32_t target) {   /* REGIMM rt=1 */
    return R(0x01, rs, 1, (int)((target - (pc + 4)) >> 2));
}
static uint32_t j_(uint32_t target) { return 0x02u << 26 | ((target >> 2) & 0x03FFFFFFu); }
#define JR_RA 0x03E00008u
#define AT 1
#define V0 2
#define A0 4
#define A1 5
#define A2 6
#define T0 8
#define T1 9
#define T2 10
#define T3 11
#define A3 7
#define SP 29

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static uint8_t g_code[0x110];
static uint32_t g_w[0x110 / 4];

static void store(void) {
    for (size_t k = 0; k < sizeof g_w / 4; k++)
        for (int b = 0; b < 4; b++) g_code[k * 4 + b] = (uint8_t)(g_w[k] >> (8 * b));
}

/* The SNC idiom, verbatim. */
static void build(uint32_t case2_target) {
    uint32_t *w = g_w;
    memset(g_w, 0, sizeof g_w);
    w[0x00 / 4] = sltiu(A2, A0, 3);
    w[0x04 / 4] = beq(A2, 0, BASE + 0x04, BASE + 0x34);
    w[0x08 / 4] = sll(A0, A0, 2);
    w[0x0C / 4] = lui(AT, (TABLE + 0x8000u) >> 16);
    w[0x10 / 4] = addu(AT, AT, A0);
    w[0x14 / 4] = lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT);
    w[0x18 / 4] = jr(AT);
    w[0x1C / 4] = 0;
    w[0x20 / 4] = ori(V0, 0, 10); w[0x24 / 4] = JR_RA; w[0x28 / 4] = 0;   /* case 0 */
    w[0x2C / 4] = JR_RA;          w[0x30 / 4] = ori(V0, 0, 11);           /* case 1 */
    w[0x34 / 4] = JR_RA;          w[0x38 / 4] = ori(V0, 0, 0);            /* default */
    w[0x3C / 4] = JR_RA;          w[0x40 / 4] = ori(V0, 0, 12);           /* case 2 */
    w[0x100 / 4] = BASE + 0x20;
    w[0x104 / 4] = BASE + 0x2C;
    w[0x108 / 4] = case2_target;
    store();
}

/* The same switch with a call between the lui and the addu. The callee
 * clobbers $at, so the addu does not see the lui's value: not the idiom.
 * The sltiu sits after the call, so only the walk for the table base can
 * reject it. */
static void build_call(void) {
    uint32_t *w = g_w;
    memset(g_w, 0, sizeof g_w);
    w[0x00 / 4] = lui(AT, (TABLE + 0x8000u) >> 16);
    w[0x04 / 4] = jal(BASE + 0x48);
    w[0x08 / 4] = 0;
    w[0x0C / 4] = sltiu(A2, A0, 3);     /* the bound comes after the call */
    w[0x10 / 4] = beq(A2, 0, BASE + 0x10, BASE + 0x40);
    w[0x14 / 4] = sll(A0, A0, 2);
    w[0x18 / 4] = addu(AT, AT, A0);
    w[0x1C / 4] = lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT);
    w[0x20 / 4] = jr(AT);
    w[0x24 / 4] = 0;
    w[0x28 / 4] = JR_RA; w[0x2C / 4] = ori(V0, 0, 10);                   /* case 0 */
    w[0x30 / 4] = JR_RA; w[0x34 / 4] = ori(V0, 0, 11);                   /* case 1 */
    w[0x38 / 4] = JR_RA; w[0x3C / 4] = ori(V0, 0, 12);                   /* case 2 */
    w[0x40 / 4] = JR_RA; w[0x44 / 4] = ori(V0, 0, 0);                    /* default */
    w[0x48 / 4] = JR_RA; w[0x4C / 4] = 0;                                /* callee */
    w[0x100 / 4] = BASE + 0x28;
    w[0x104 / 4] = BASE + 0x30;
    w[0x108 / 4] = BASE + 0x38;
    store();
}

/* A switch whose bound is written by the caller: `seq` emits the code
 * before `lui $at` (it is given the address of the default block), the
 * index is `idx`, scaled already. Cases at 0x60/0x68/0x70, default at 0x80,
 * and at 0x90 a second function that branches into the default block. */
#define SEQ_DEFAULT (BASE + 0x80)
#define SEQ_OTHER   (BASE + 0x90)
#define SEQ_TEXT    0xA0u

typedef struct { uint32_t pc; } asm_t;
static void emit(asm_t *as, uint32_t w) { g_w[(as->pc - BASE) / 4] = w; as->pc += 4; }

/* The table load: SNC's `lui $at; addu $at, $at, idx; lw $at, lo($at)`
 * indexing from `tbase`, or (gcc) `lui $t3; addiu $t2, $t3, lo; addu $t0,
 * idx, $t2; lw $t1, 0($t0)` indexing from TABLE. The table's three entries
 * are always at TABLE; the words either side are code pointers too, so a
 * table read from the wrong place still passes the per-entry check. */
enum { SNC, GCC };

static uint32_t build_seq_at(void (*seq)(asm_t *), int idx, int other, uint32_t tbase, int style) {
    asm_t as = { BASE };
    memset(g_w, 0, sizeof g_w);
    seq(&as);
    int jreg = AT;
    if (style == SNC) {
        emit(&as, lui(AT, (tbase + 0x8000u) >> 16));
        emit(&as, addu(AT, AT, idx));
        emit(&as, lw(AT, (int)(int16_t)(tbase & 0xFFFF), AT));
    } else {
        emit(&as, lui(T3, (TABLE + 0x8000u) >> 16));
        emit(&as, addiu(T2, T3, (int)(int16_t)(TABLE & 0xFFFF)));
        emit(&as, addu(T0, idx, T2));
        emit(&as, lw(T1, 0, T0));
        jreg = T1;
    }
    const uint32_t site = as.pc;
    emit(&as, jr(jreg));
    emit(&as, 0);
    as.pc = BASE + 0x60;
    for (int k = 0; k < 3; k++) { emit(&as, JR_RA); emit(&as, ori(V0, 0, 10 + k)); }
    as.pc = SEQ_DEFAULT;
    emit(&as, JR_RA); emit(&as, ori(V0, 0, 0));
    if (other) {
        as.pc = SEQ_OTHER;
        emit(&as, beq(A0, 0, as.pc, SEQ_DEFAULT)); emit(&as, 0);
        emit(&as, JR_RA); emit(&as, 0);
    }
    for (int k = 0; k < 3; k++) g_w[0x100 / 4 + k] = BASE + 0x60 + 8u * (uint32_t)k;
    g_w[0x10C / 4] = BASE + 0x80;       /* a fourth code pointer: an over-count would take it */
    g_w[0x0FC / 4] = BASE + 0x80;       /* and one before: so would an under-run */
    store();
    return site;
}

static uint32_t build_seq(void (*seq)(asm_t *), int idx, int other) {
    return build_seq_at(seq, idx, other, TABLE, SNC);
}

/* slti a1,a0,5; bne -> default (lower bound); slti a1,a0,8 in its slot;
 * beq -> default (upper bound); addiu a0,a0,-5; sll: cases 5..7. */
static void seq_biased(asm_t *as) {
    emit(as, slti(A1, A0, 5));
    emit(as, bne(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, slti(A1, A0, 8));
    emit(as, beq(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, addiu(A0, A0, -5));
    emit(as, sll(A0, A0, 2));
}
/* A load in a branch-likely's slot runs only when the branch is taken, so
 * it does not clobber the key on the way to the switch. */
static void seq_likely(asm_t *as) {
    emit(as, sltiu(A2, A1, 3));
    emit(as, beql(A2, 0, as->pc, SEQ_DEFAULT));
    emit(as, lw(A1, 0, SP));
    emit(as, sll(A1, A1, 2));
}
/* The bound tests the original, the switch indexes a copy. */
static void seq_copy(asm_t *as) {
    emit(as, sltiu(T1, A1, 3));
    emit(as, or_(A0, A1, 0));
    emit(as, beq(T1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* The bound tests a different register than the one indexed. */
static void seq_other_reg(asm_t *as) {
    emit(as, sltiu(A2, A1, 3));
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* The bias is on the bounded copy, not the key: indexes 1..3 are reachable,
 * so the code indexes from TABLE - 4 and entry 0 is never used. */
static void seq_bound_bias(asm_t *as) {
    emit(as, addiu(A1, A0, -1));
    emit(as, sltiu(A1, A1, 3));
    emit(as, beq(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* A lower bound (blez: $a0 >= 1) with a signed upper bound: indexes 1..3. */
static void seq_blez(asm_t *as) {
    emit(as, blez(A0, as->pc, SEQ_DEFAULT));
    emit(as, 0);
    emit(as, slti(A1, A0, 4));
    emit(as, beq(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* The bound lets index -1 through: the code reads before its table. */
static void seq_negative(asm_t *as) {
    emit(as, addiu(A1, A0, 1));
    emit(as, sltiu(A1, A1, 4));
    emit(as, beq(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* The bound and scaling before a gcc-style table load. */
static void seq_plain(asm_t *as) {
    emit(as, sltiu(A2, A0, 3));
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}
/* A branch from the top lands on the bound's beq, past the bound. Its delay
 * slot computes `slot` into $a2, which is what the beq tests on that path. */
static uint32_t g_slot;
static void seq_merge(asm_t *as) {
    const uint32_t x = as->pc;
    emit(as, bne(A1, 0, x, x + 0x10));
    emit(as, g_slot);
    emit(as, ori(V0, 0, 1));
    emit(as, sltiu(A2, A0, 3));
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));      /* x + 0x10 */
    emit(as, 0);
    emit(as, sll(A0, A0, 2));
}
/* The branch from the top is taken when $a0 < 1, skipping the lower bound
 * the fall-through establishes ($a0 >= 1), but then the index $a0 - 2 is
 * negative and fails the sltiu: no index reaches the jr that way, so the
 * edge cannot widen the table. */
static void seq_infeasible(asm_t *as) {
    emit(as, slti(A3, A0, 1));
    const uint32_t x = as->pc;
    emit(as, bne(A3, 0, x, x + 0xC));
    emit(as, addiu(A0, A0, -2));
    emit(as, ori(V0, 0, 1));
    emit(as, sltiu(A2, A0, 3));                     /* x + 0xC */
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));
    emit(as, 0);
    emit(as, sll(A0, A0, 2));
}
/* The bound's beq is also reached from another function (at SEQ_OTHER),
 * whose branch's delay slot recomputes the bound. */
static void seq_shared(asm_t *as) {
    emit(as, sltiu(A2, A0, 3));
    emit(as, ori(V0, 0, 1));
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));      /* BASE + 8 */
    emit(as, 0);
    emit(as, sll(A0, A0, 2));
}
/* Along the branch from the top the key is a constant (-1, so the sltiu
 * fails and no index gets through); the table stands. */
static void seq_const_key(asm_t *as) {
    const uint32_t x = as->pc;
    emit(as, bne(A1, 0, x, x + 0xC));
    emit(as, addiu(A0, 0, -1));
    emit(as, ori(V0, 0, 1));
    emit(as, sltiu(A2, A0, 3));                     /* x + 0xC */
    emit(as, beq(A2, 0, as->pc, SEQ_DEFAULT));
    emit(as, 0);
    emit(as, sll(A0, A0, 2));
}

/* A lower bound that lets $a0 = -1 through: index -1 reads before the table. */
static void seq_negative_lower(asm_t *as) {
    emit(as, slti(A1, A0, -1));
    emit(as, bne(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, slti(A1, A0, 3));
    emit(as, beq(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}

static void setup(a_analysis *an, uint32_t text_size);

/* Chains of merges. F (at BASE) is the switch: `sltiu a2,a0,3; beq a2,zero,
 * default; sll ...`. Each G_i is `<slot0>; b <past the previous one's
 * bound>; nop`, where G_1 branches to F's beq and G_i to G_(i-1)'s `b`, so a
 * path into the jr can take several edges in a row: G_n -> ... -> G_1 -> F.
 * Cases at 0x30/0x38/0x40, default at 0x48, G_i at 0x50 + 0xC * (i - 1). */
static void check_chain(const char *name, int n, uint32_t last_slot0, int want_table) {
    asm_t as = { BASE };
    memset(g_w, 0, sizeof g_w);
    emit(&as, sltiu(A2, A0, 3));
    emit(&as, beq(A2, 0, as.pc, BASE + 0x48));
    emit(&as, sll(A0, A0, 2));
    emit(&as, lui(AT, (TABLE + 0x8000u) >> 16));
    emit(&as, addu(AT, AT, A0));
    emit(&as, lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT));
    const uint32_t site = as.pc;
    emit(&as, jr(AT)); emit(&as, 0);
    as.pc = BASE + 0x30;
    for (int k = 0; k < 3; k++) { emit(&as, JR_RA); emit(&as, 0); }
    emit(&as, JR_RA); emit(&as, 0);                 /* default, 0x48 */
    uint32_t seeds[16] = { BASE };
    uint32_t into = BASE + 4;
    for (int i = 1; i <= n; i++) {
        const uint32_t g = BASE + 0x50 + 0xCu * (uint32_t)(i - 1);
        as.pc = g;
        emit(&as, i == n ? last_slot0 : sltiu(A2, A0, 3));
        emit(&as, beq(0, 0, as.pc, into));
        emit(&as, 0);
        seeds[i] = g;
        into = g + 4;
    }
    for (int k = 0; k < 3; k++) g_w[0x100 / 4 + k] = BASE + 0x30 + 8u * (uint32_t)k;
    store();

    a_analysis an;
    setup(&an, 0x50 + 0xCu * (uint32_t)n);
    CHECK(a_discover(&an, seeds, n + 1) == 0, "%s: discover", name);
    const a_jtable *jt = a_find_jtable(&an, site);
    if (want_table) {
        CHECK(jt != NULL && an.nindirects == 0, "%s: table expected (%d unresolved)",
              name, an.nindirects);
    } else {
        CHECK(jt == NULL && an.nindirects == 1, "%s: no table expected (%d tables)",
              name, an.njtables);
        /* The table was real, only its bound uncertain: its cases become
         * entries, so a dispatch from the jr still finds them. */
        for (int k = 0; k < 3; k++) {
            const uint32_t t = BASE + 0x30 + 8u * (uint32_t)k;
            CHECK(an.owner[(t - BASE) / 4] == t, "%s: case 0x%08X is a function", name, t);
        }
    }
    a_analysis_free(&an);
}

/* A state machine: every case adjusts the key and branches back to the
 * switch's bound, so paths into the jr can run through the cases in any
 * order -- N^depth chains. Checking them all is too slow; past the per-table
 * budget the table is rejected and its cases promoted. */
#define SM_N 32
static void check_state_machine(void) {
    enum { TEXT = 0x40 + SM_N * 16 + 0x10, SIZE = TEXT + SM_N * 4 };
    static uint32_t w[SIZE / 4];
    static uint8_t code[SIZE];
    const uint32_t table = BASE + TEXT;
    memset(w, 0, sizeof w);
    w[0] = sltiu(A2, A0, SM_N);
    w[1] = beq(A2, 0, BASE + 4, BASE + 0x30);
    w[2] = sll(A1, A0, 2);
    w[3] = lui(AT, (table + 0x8000u) >> 16);
    w[4] = addu(AT, AT, A1);
    w[5] = lw(AT, (int)(int16_t)(table & 0xFFFF), AT);
    w[6] = jr(AT);
    w[0x30 / 4] = JR_RA;                                /* default */
    for (int i = 0; i < SM_N; i++) {
        const uint32_t c = 0x40 + 16u * (uint32_t)i;
        w[c / 4]     = addiu(A0, A0, (i & 1) ? -1 : 0);
        w[c / 4 + 1] = addiu(A0, A0, 0);
        w[c / 4 + 2] = beq(0, 0, BASE + c + 8, BASE);   /* b to the bound */
        w[TEXT / 4 + i] = BASE + c;
    }
    for (size_t k = 0; k < SIZE / 4; k++)
        for (int b = 0; b < 4; b++) code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code; an.base = BASE; an.size = SIZE;
    an.scan_base = BASE; an.scan_size = TEXT;
    an.image = code; an.image_base = BASE; an.image_size = SIZE;
    uint32_t seed = BASE;
    const clock_t t0 = clock();
    CHECK(a_discover(&an, &seed, 1) == 0, "state machine: discover");
    const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    CHECK(an.njtables == 0 && an.nindirects == 1,
          "state machine: table rejected once the budget runs out (%d tables)", an.njtables);
    for (int i = 0; i < SM_N; i++) {
        const uint32_t c = BASE + 0x40 + 16u * (uint32_t)i;
        CHECK(an.owner[(c - BASE) / 4] == c, "state machine: case 0x%08X is a function", c);
    }
    CHECK(secs < 2.0, "state machine: took %.2f s", secs);
    a_analysis_free(&an);
}

/* Only a lower bound (`bne` on the slti): the table's size is unknown. */
static void seq_lower_only(asm_t *as) {
    emit(as, slti(A1, A0, 3));
    emit(as, bne(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}

static void setup(a_analysis *an, uint32_t text_size);

static void check_seq_at(const char *name, void (*seq)(asm_t *), int idx, uint32_t tbase,
                         int style, uint32_t want, uint32_t want_lo) {
    a_analysis an;
    uint32_t seed = BASE;
    const uint32_t site = build_seq_at(seq, idx, 0, tbase, style);
    setup(&an, SEQ_TEXT);
    CHECK(a_discover(&an, &seed, 1) == 0, "%s: discover", name);
    const a_jtable *jt = a_find_jtable(&an, site);
    if (!want) {
        CHECK(jt == NULL && an.njtables == 0, "%s: no table expected", name);
        CHECK(an.nindirects == 1, "%s: jr left as a computed jump, got %d", name, an.nindirects);
    } else {
        CHECK(jt != NULL, "%s: table found", name);
        if (jt) {
            CHECK(jt->count == want && jt->table == TABLE && jt->lo == want_lo,
                  "%s: count %u lo %u table 0x%08X (want %u, %u, 0x%08X)",
                  name, jt->count, jt->lo, jt->table, want, want_lo, TABLE);
            for (uint32_t k = 0; k < jt->count && k < 3; k++)
                CHECK(an.jt_targets[jt->first + k] == BASE + 0x60 + 8 * k,
                      "%s: target %u is 0x%08X", name, k, an.jt_targets[jt->first + k]);
        }
        CHECK(an.nindirects == 0, "%s: no unresolved jump, got %d", name, an.nindirects);
    }
    a_analysis_free(&an);
}

static void check_seq(const char *name, void (*seq)(asm_t *), int idx, uint32_t want) {
    check_seq_at(name, seq, idx, TABLE, SNC, want, 0);
}

/* A transfer between the lui and the addu: the code after it is another
 * function (seeded separately), and the lui does not reach it. With
 * `lui_in_slot` the lui is the transfer's delay slot, which runs before the
 * transfer, not on the way to the addu. With `bound_before` the sltiu sits
 * before the transfer too. */
static void check_stop(const char *name, uint32_t xfer, int lui_in_slot, int bound_before) {
    asm_t as = { BASE };
    memset(g_w, 0, sizeof g_w);
    if (bound_before) emit(&as, sltiu(A2, A0, 3));
    if (!lui_in_slot) emit(&as, lui(AT, (TABLE + 0x8000u) >> 16));
    emit(&as, xfer);
    emit(&as, lui_in_slot ? lui(AT, (TABLE + 0x8000u) >> 16) : 0);
    const uint32_t fn2 = as.pc;
    if (!bound_before) emit(&as, sltiu(A2, A0, 3));
    emit(&as, beq(A2, 0, as.pc, SEQ_DEFAULT));
    emit(&as, sll(A0, A0, 2));
    emit(&as, addu(AT, AT, A0));
    emit(&as, lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT));
    emit(&as, jr(AT));
    emit(&as, 0);
    as.pc = BASE + 0x60;
    for (int k = 0; k < 3; k++) { emit(&as, JR_RA); emit(&as, ori(V0, 0, 10 + k)); }
    as.pc = SEQ_DEFAULT;
    emit(&as, JR_RA); emit(&as, 0);
    emit(&as, JR_RA); emit(&as, 0);                 /* 0x88: the transfers' target */
    for (int k = 0; k < 3; k++) g_w[0x100 / 4 + k] = BASE + 0x60 + 8u * (uint32_t)k;
    store();

    a_analysis an;
    uint32_t seeds[2] = { BASE, fn2 };
    setup(&an, SEQ_TEXT);
    CHECK(a_discover(&an, seeds, 2) == 0, "%s: discover", name);
    CHECK(an.njtables == 0 && an.nindirects == 1, "%s: no table expected, got %d (%d unresolved)",
          name, an.njtables, an.nindirects);
    a_analysis_free(&an);
}

static void setup(a_analysis *an, uint32_t text_size) {
    memset(an, 0, sizeof *an);
    an->code = g_code;
    an->base = BASE;
    an->size = sizeof g_code;
    an->scan_base = BASE;
    an->scan_size = text_size;        /* .text */
    an->image = g_code;
    an->image_base = BASE;
    an->image_size = sizeof g_code;
}

/* Emit the discovered code and return the function source. The files stay
 * on disk: ctest compiles each of them (-fsyntax-only) afterwards, which
 * catches undefined labels and duplicate case values. */
static const char *emit_source(const a_analysis *an, const char *prefix) {
    static char src[1 << 16];
    char path[64];
    emit_opts o;
    memset(&o, 0, sizeof o);
    o.outdir = ".";
    o.prefix = prefix;
    o.module = "synthetic";
    src[0] = '\0';
    CHECK(a_emit(an, &o) == 0, "emit");
    snprintf(path, sizeof path, "./%s_funcs.c", prefix);
    FILE *f = fopen(path, "rb");
    size_t len = f ? fread(src, 1, sizeof src - 1, f) : 0;
    src[len] = '\0';
    if (f) fclose(f);
    return src;
}

/* The SNC idiom with the probe layout: sltiu bound, table at TABLE with
 * entries e0..e2, a one-instruction slot at 0x1C (`ori $v0, 0, 1` when
 * `slot`), blocks at 0x0C (inside the idiom), 0x20 and 0x28. */
static void build_probe(uint32_t e0, uint32_t e1, uint32_t e2, int slot) {
    uint32_t *w = g_w;
    memset(g_w, 0, sizeof g_w);
    w[0x00 / 4] = sltiu(A2, A0, 3);
    w[0x04 / 4] = beq(A2, 0, BASE + 4, BASE + 0x40);
    w[0x08 / 4] = sll(A0, A0, 2);
    w[0x0C / 4] = lui(AT, (TABLE + 0x8000u) >> 16);
    w[0x10 / 4] = addu(AT, AT, A0);
    w[0x14 / 4] = lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT);
    w[0x18 / 4] = jr(AT);
    w[0x1C / 4] = slot ? ori(V0, 0, 1) : 0;
    w[0x20 / 4] = JR_RA; w[0x24 / 4] = 0;
    w[0x28 / 4] = ori(V0, 0, 3); w[0x2C / 4] = JR_RA; w[0x30 / 4] = 0;
    w[0x40 / 4] = JR_RA; w[0x44 / 4] = 0;
    w[0x100 / 4] = e0; w[0x104 / 4] = e1; w[0x108 / 4] = e2;
    store();
}


/* Cases before the switch, one on the function's own entry:
 *   0x00 beq a1 -> 0x30; 0x08 jr ra; 0x10 case 0; 0x1C case 1; 0x30 bound
 *   and idiom, jr at 0x48; 0x50 case 2; 0x60 default. Table: 0x10, 0x1C,
 *   0x50, BASE. */
static void build_back(void) {
    uint32_t *w = g_w;
    memset(g_w, 0, sizeof g_w);
    w[0x00 / 4] = beq(A1, 0, BASE, BASE + 0x30);
    w[0x08 / 4] = JR_RA;
    w[0x10 / 4] = ori(V0, 0, 1); w[0x14 / 4] = JR_RA;
    w[0x1C / 4] = ori(V0, 0, 2); w[0x20 / 4] = JR_RA;
    w[0x30 / 4] = sltiu(A2, A0, 4);
    w[0x34 / 4] = beq(A2, 0, BASE + 0x34, BASE + 0x60);
    w[0x38 / 4] = sll(A0, A0, 2);
    w[0x3C / 4] = lui(AT, (TABLE + 0x8000u) >> 16);
    w[0x40 / 4] = addu(AT, AT, A0);
    w[0x44 / 4] = lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT);
    w[0x48 / 4] = jr(AT);
    w[0x50 / 4] = JR_RA; w[0x54 / 4] = ori(V0, 0, 3);
    w[0x60 / 4] = JR_RA; w[0x64 / 4] = ori(V0, 0, 0);
    w[0x100 / 4] = BASE + 0x10; w[0x104 / 4] = BASE + 0x1C;
    w[0x108 / 4] = BASE + 0x50; w[0x10C / 4] = BASE;
    store();
}

/* `transfer` at BASE, its delay slot at BASE + 4 is also the entry of a
 * second function. The first function's body must still run the slot. */
static void check_foreign_slot_order(const char *name, uint32_t transfer, int falls_on,
                                     int slot_first);
static void check_foreign_slot(const char *name, uint32_t transfer, int falls_on) {
    check_foreign_slot_order(name, transfer, falls_on, 1);
}
static void check_foreign_slot_order(const char *name, uint32_t transfer, int falls_on,
                                     int slot_first) {
    a_analysis an;
    memset(g_w, 0, sizeof g_w);
    g_w[0x00 / 4] = transfer;
    g_w[0x04 / 4] = ori(V0, 0, 5);              /* the slot, and an entry */
    g_w[0x08 / 4] = JR_RA; g_w[0x0C / 4] = 0;
    g_w[0x10 / 4] = JR_RA; g_w[0x14 / 4] = ori(V0, 0, 7);
    store();
    /* The queue walks the last seed first. */
    uint32_t seeds[2] = { slot_first ? BASE : BASE + 4, slot_first ? BASE + 4 : BASE };
    setup(&an, 0x18);
    CHECK(a_discover(&an, seeds, 2) == 0, "%s: discover", name);
    CHECK(an.nfuncs >= 2 && an.owner[1] != BASE, "%s: slot owned by another function", name);
    char prefix[32];
    snprintf(prefix, sizeof prefix, "t_switch_slot_%c%c", name[0], name[strlen(name) - 1]);
    const char *src = emit_source(&an, prefix);
    const char *body = strstr(src, "static void psp_body_08804000");
    const char *end = body ? strstr(body, "static void psp_body_08804004") : NULL;
    const char *slot = body ? strstr(body, "/* 08804004") : NULL;
    CHECK(slot != NULL && (end == NULL || slot < end), "%s: slot runs inline in the first body", name);
    /* The path past the transfer (not taken, or the call's return) runs on
     * at BASE + 8, which the slot's function owns: it is an entry, called. */
    if (body) {
        char first[4096];
        const size_t n = end ? (size_t)(end - body) : strlen(body);
        snprintf(first, sizeof first, "%.*s", (int)(n < sizeof first ? n : sizeof first - 1), body);
        CHECK(strstr(first, "not discovered") == NULL, "%s: nothing undiscovered", name);
        if (falls_on)
            CHECK(strstr(first, "psp_func_08804008()") != NULL, "%s: BASE + 8 is called", name);
    }
    a_analysis_free(&an);
}

/* An always-taken branch at BASE (to 0x18) has no fall-through: the code at
 * 0x08 is only reachable through its own labels, so it is not walked and
 * the body does not run into it. */
static void check_always(const char *name, const char *prefix, uint32_t br) {
    a_analysis an;
    memset(g_w, 0, sizeof g_w);
    g_w[0x00 / 4] = br;
    g_w[0x08 / 4] = ori(V0, 0, 1); g_w[0x0C / 4] = JR_RA;
    g_w[0x18 / 4] = JR_RA; g_w[0x1C / 4] = ori(V0, 0, 2);
    store();
    uint32_t seed = BASE;
    setup(&an, 0x20);
    CHECK(a_discover(&an, &seed, 1) == 0, "%s: discover", name);
    CHECK(an.owner[0x08 / 4] == A_NO_OWNER, "%s: code after it not walked", name);
    CHECK(an.owner[0x18 / 4] == BASE, "%s: target walked", name);
    const char *src = emit_source(&an, prefix);
    CHECK(strstr(src, "falls into") == NULL && strstr(src, "not discovered") == NULL,
          "%s: no fall-through emitted", name);
    CHECK(strstr(src, "goto L_08804018;") != NULL, "%s: branch is a goto", name);
    a_analysis_free(&an);
}

int main(void) {
    a_analysis an;
    uint32_t seed = BASE;

    build(BASE + 0x3C);
    setup(&an, 0x44);
    CHECK(a_discover(&an, &seed, 1) == 0, "discover");
    CHECK(an.nfuncs == 1, "cases stay inside the function, got %d functions", an.nfuncs);
    CHECK(an.njtables == 1, "one table, got %d", an.njtables);
    CHECK(an.nindirects == 0, "no unresolved computed jump, got %d", an.nindirects);
    CHECK(an.ntables == 1 && an.ntable_targets == 3, "report counts %d/%d",
          an.ntables, an.ntable_targets);
    const a_jtable *jt = a_find_jtable(&an, BASE + 0x18);
    CHECK(jt != NULL, "table found at the jr");
    if (jt) {
        CHECK(jt->table == TABLE, "table address 0x%08X", jt->table);
        CHECK(jt->count == 3, "three entries, got %u", jt->count);
        CHECK(an.jt_targets[jt->first] == BASE + 0x20 &&
              an.jt_targets[jt->first + 1] == BASE + 0x2C &&
              an.jt_targets[jt->first + 2] == BASE + 0x3C, "targets in table order");
    }
    CHECK(an.owner[(0x3C) / 4] == BASE, "case 2 owned by the switch's function");
    CHECK(an.nfuncs == 1 && !an.funcs[0].has_indirect, "function not marked indirect");
    CHECK(a_find_jtable(&an, BASE + 0x14) == NULL, "no table at other addresses");
    a_analysis_free(&an);
    CHECK(an.jtables == NULL && an.njtables == 0 && an.jt_targets == NULL &&
          an.njt_targets == 0, "free clears the tables");

    /* The emitter turns the table into a C switch on the loaded target
     * (the address the jr jumps to, so the table's lowest index does not
     * matter). A case the function owns is a goto. */
    {
        uint32_t s1 = BASE;
        build(BASE + 0x3C);
        g_w[0x1C / 4] = addiu(AT, AT, 4);       /* the slot clobbers the switch register */
        store();
        setup(&an, 0x44);
        CHECK(a_discover(&an, &s1, 1) == 0, "discover for emit");
        const char *src = emit_source(&an, "t_switch");
        CHECK(strstr(src, "{ uint32_t _sw = r_at;") != NULL, "target read before the delay slot");
        CHECK(strstr(src, "switch (_sw) {") != NULL, "switch on the loaded target");
        CHECK(strstr(src, "case 0x08804020u: goto L_08804020;") != NULL, "case 0");
        CHECK(strstr(src, "case 0x0880402Cu: goto L_0880402C;") != NULL, "case 1");
        CHECK(strstr(src, "case 0x0880403Cu: goto L_0880403C;") != NULL, "case 2");
        CHECK(strstr(src, "default: psp_dispatch(_sw); return;") != NULL, "default dispatches");
        CHECK(strstr(src, "psp_dispatch(r_at);") == NULL, "no bare computed dispatch left");
        const char *cap = strstr(src, "uint32_t _sw = r_at;");
        const char *clob = strstr(src, "r_at = r_at + 4");
        const char *sw = strstr(src, "switch (_sw) {");
        CHECK(cap && clob && sw && cap < clob && clob < sw,
              "the slot runs after the capture and before the switch");
        CHECK(strstr(src, "L_0880403C: PSP_MARK") != NULL, "case 2 label emitted");
        a_analysis_free(&an);
    }

    /* A case that is another function's entry is a tail call, not a goto
     * (and there is no label for it). Two cases with one target give one
     * case label. */
    {
        uint32_t s2[2] = { BASE, BASE + 0x3C };
        build(BASE + 0x3C);
        g_w[0x104 / 4] = BASE + 0x20;           /* case 1 repeats case 0 */
        store();
        setup(&an, 0x44);
        CHECK(a_discover(&an, s2, 2) == 0, "discover for emit (entry target)");
        const char *src = emit_source(&an, "t_switch_entry");
        const char *c2 = strstr(src, "case 0x0880403Cu:\n");
        CHECK(c2 != NULL, "entry case emitted as a call");
        CHECK(c2 && strstr(c2, "psp_func_0880403C(); PSP_SP_CALL(0x0880403Cu") != NULL &&
              strstr(c2, "psp_func_0880403C(); PSP_SP_CALL(0x0880403Cu") < strstr(c2, "default:"),
              "entry case calls the function");
        CHECK(strstr(src, "goto L_0880403C") == NULL, "no goto into another function");
        const char *lbl = strstr(src, "L_0880403C:");
        CHECK(lbl && strstr(lbl + 1, "L_0880403C:") == NULL,
              "only the function's own entry label");
        CHECK(strstr(src, "case 0x08804020u: goto L_08804020;") != NULL, "owned case still a goto");
        const char *sw = strstr(src, "switch (_sw) {");
        const char *first = sw ? strstr(sw, "case 0x08804020u") : NULL;
        CHECK(first && strstr(first + 1, "case 0x08804020u") == NULL, "duplicate target: one case");
        a_analysis_free(&an);
    }

    /* Cases that jump back: to blocks before the switch and to the
     * function's own entry. All gotos (no recursive call), and a backward
     * goto counts as a loop iteration. */
    {
        uint32_t s3 = BASE;
        build_back();
        setup(&an, 0x68);
        CHECK(a_discover(&an, &s3, 1) == 0, "discover (backward)");
        const a_jtable *jt = a_find_jtable(&an, BASE + 0x48);
        CHECK(jt != NULL && jt->count == 4, "backward table kept");
        CHECK(an.nfuncs == 1, "backward blocks stay in the function, got %d", an.nfuncs);
        const char *src = emit_source(&an, "t_switch_back");
        CHECK(strstr(src, "case 0x08804000u: PSP_LOOP(0x08804000u); goto L_08804000;") != NULL,
              "case on the function's own entry is a looping goto");
        CHECK(strstr(src, "case 0x08804010u: PSP_LOOP(0x08804010u); goto L_08804010;") != NULL,
              "backward case 0 is a looping goto");
        CHECK(strstr(src, "case 0x0880401Cu: PSP_LOOP(0x0880401Cu); goto L_0880401C;") != NULL,
              "backward case 1 is a looping goto");
        CHECK(strstr(src, "case 0x08804050u: goto L_08804050;") != NULL, "forward case has no PSP_LOOP");
        CHECK(strstr(src, "psp_func_08804000();") == NULL, "no recursive call");
        a_analysis_free(&an);
    }

    /* A case that is the jr's own delay slot: discovery makes it an entry,
     * so the case is a tail call, but the slot still runs inline with the
     * jr, before the switch. */
    {
        uint32_t s4 = BASE;
        build_probe(BASE + 0x0C, BASE + 0x1C, BASE + 0x28, 1);
        setup(&an, 0x48);
        CHECK(a_discover(&an, &s4, 1) == 0, "discover (slot target)");
        CHECK(an.njtables == 1, "slot-target table kept, got %d", an.njtables);
        const char *src = emit_source(&an, "t_switch_slot");
        const char *body = strstr(src, "static void psp_body_0880400C");
        const char *sw = body ? strstr(body, "uint32_t _sw") : NULL;
        const char *inl = sw ? strstr(sw, "/* 0880401C") : NULL;
        const char *swi = sw ? strstr(sw, "switch (_sw) {") : NULL;
        CHECK(inl && swi && inl < swi, "slot runs inline before the switch");
        CHECK(strstr(src, "psp_func_0880401C(); PSP_SP_CALL") != NULL, "slot case is a call to its entry");
        a_analysis_free(&an);
    }

    /* A delay slot that is another function's entry still runs with its
     * transfer, whatever the kind. */
    check_foreign_slot("branch", beq(A0, 0, BASE, BASE + 0x10), 1);
    check_foreign_slot("likely branch", beql(A0, 0, BASE, BASE + 0x10), 1);
    check_foreign_slot("jal", jal(BASE + 0x10), 1);
    check_foreign_slot("j", j_(BASE + 0x10), 0);
    check_foreign_slot("jr $ra", JR_RA, 0);
    check_foreign_slot("jr", jr(A0), 0);

    /* Whichever is walked first, the slot stays its own function's: the
     * branch's walk does not claim a delay slot that is an entry. */
    check_foreign_slot_order("branch, branch walked first", beq(A0, 0, BASE, BASE + 0x10), 1, 0);
    check_foreign_slot_order("jal, jal walked first", jal(BASE + 0x10), 1, 0);

    /* A branch into the delay slot of a `b`: the slot runs, then the
     * instruction after it -- the walk follows on, and the body does too,
     * into another function's code if that is where it is. */
    for (int other = 0; other < 2; other++) {
        memset(g_w, 0, sizeof g_w);
        g_w[0x00 / 4] = bne(A0, 0, BASE, BASE + 0x0C);
        g_w[0x08 / 4] = beq(0, 0, BASE + 0x08, BASE + 0x20);
        g_w[0x0C / 4] = ori(V0, 0, 1);          /* b's slot, and a branch target */
        g_w[0x10 / 4] = JR_RA;
        g_w[0x20 / 4] = JR_RA; g_w[0x24 / 4] = ori(V0, 0, 2);
        store();
        uint32_t sd[2] = { BASE, BASE + 0x10 };
        setup(&an, 0x28);
        CHECK(a_discover(&an, sd, other ? 2 : 1) == 0, "discover (slot target %d)", other);
        CHECK(an.owner[0x10 / 4] == (other ? BASE + 0x10 : BASE),
              "the code after the slot is walked (%d): owner 0x%08X", other, an.owner[0x10 / 4]);
        const char *src = emit_source(&an, other ? "t_slot_target_o" : "t_slot_target");
        CHECK(strstr(src, "not discovered") == NULL, "slot target %d: nothing undiscovered", other);
        if (other) {
            const char *body = strstr(src, "static void psp_body_08804000");
            const char *call = body ? strstr(body, "psp_func_08804010()") : NULL;
            const char *next = body ? strstr(body, "static void psp_body_08804010") : NULL;
            CHECK(call && (!next || call < next), "slot target: runs on into 0x08804010");
        }
        a_analysis_free(&an);
    }

    /* The game's strlen (0x08AEA170): its word loop's head is an entry of
     * its own, walked after the parent claimed the byte tail. The loop's
     * fall-through runs on into that tail; returning there instead skipped
     * the `subu $v0` and handed back garbage. The tail is promoted and
     * called. Layout: 0x00 bne a1 -> 0x14 (parent); 0x08 loop head (entry):
     * beql t0, zero -> 0x08, slot; 0x10 or; 0x14 tail: jr ra; subu v0. */
    {
        memset(g_w, 0, sizeof g_w);
        g_w[0x00 / 4] = bne(A1, 0, BASE, BASE + 0x14);
        g_w[0x04 / 4] = or_(A3, A0, 0);
        g_w[0x08 / 4] = lw(T0, 0, A2);                   /* loop head */
        g_w[0x0C / 4] = beql(T0, 0, BASE + 0x0C, BASE + 0x08);
        g_w[0x10 / 4] = addiu(A2, A2, 4);
        g_w[0x14 / 4] = JR_RA;                           /* the tail */
        g_w[0x18 / 4] = 0x00871023u;                     /* subu v0, a0, a3 */
        store();
        uint32_t sd[2] = { BASE + 0x08, BASE };          /* BASE is walked first */
        setup(&an, 0x1C);
        CHECK(a_discover(&an, sd, 2) == 0, "discover (strlen)");
        int tail_fn = 0;
        for (int i = 0; i < an.nfuncs; i++) tail_fn |= an.funcs[i].addr == BASE + 0x14;
        CHECK(tail_fn, "strlen: the tail is promoted to an entry");
        const char *src = emit_source(&an, "t_strlen");
        const char *body = strstr(src, "static void psp_body_08804008");
        const char *close = body ? strstr(body, "\n}\n") : NULL;
        const char *call = body ? strstr(body, "psp_func_08804014()") : NULL;
        CHECK(call && close && call < close, "strlen: the loop runs on into the tail");
        a_analysis_free(&an);
    }

    /* Always-taken branches end the straight line in the walk and in the
     * emitted body. */
    check_always("b", "t_always_b", beq(0, 0, BASE, BASE + 0x18));
    check_always("beq $a1, $a1", "t_always_beq", beq(A1, A1, BASE, BASE + 0x18));
    check_always("bgez $zero", "t_always_bgez", bgez(0, BASE, BASE + 0x18));
    check_always("beql $zero, $zero", "t_always_beql", beql(0, 0, BASE, BASE + 0x18));

    /* A body whose last instruction can fall through, into code that is no
     * function (here an undecodable word stops the walk), transfers there
     * explicitly instead of returning as if it had finished. */
    {
        memset(g_w, 0, sizeof g_w);
        g_w[0x00 / 4] = ori(V0, 0, 1);
        g_w[0x04 / 4] = 0x0000003Fu;            /* not an instruction */
        store();
        uint32_t s0 = BASE;
        setup(&an, 0x10);
        CHECK(a_discover(&an, &s0, 1) == 0, "discover (runs off)");
        const char *src = emit_source(&an, "t_runs_off");
        CHECK(strstr(src, "psp_dispatch(0x08804008u);  /* not discovered */") != NULL,
              "running off the end transfers to the next address");
        a_analysis_free(&an);
    }

    /* An entry inside the bound that is not code in .text rejects the whole
     * table: the jr stays a computed jump. */
    build(TABLE);                       /* points into "rodata" */
    setup(&an, 0x44);
    CHECK(a_discover(&an, &seed, 1) == 0, "discover (bad table)");
    CHECK(an.njtables == 0, "bad table rejected, got %d", an.njtables);
    CHECK(an.nindirects == 1, "jr left as a computed jump, got %d", an.nindirects);
    a_analysis_free(&an);

    /* A call between the lui and the addu: the walk back for the table base
     * stops at the call, and the jr stays a computed jump. */
    build_call();
    setup(&an, 0x50);
    CHECK(a_discover(&an, &seed, 1) == 0, "discover (call)");
    CHECK(an.njtables == 0, "call in the idiom rejected, got %d", an.njtables);
    CHECK(an.nindirects == 1, "jr left as a computed jump (call), got %d", an.nindirects);
    a_analysis_free(&an);

    /* The bound forms SNC uses besides the plain sltiu, and two near misses. */
    check_seq("biased slti", seq_biased, A0, 3);
    check_seq("likely slot", seq_likely, A1, 3);
    check_seq("copy", seq_copy, A0, 3);
    check_seq("other register", seq_other_reg, A0, 0);
    check_seq("lower bound only", seq_lower_only, A0, 0);

    /* Leading entries the bound makes unreachable are not read (the
     * recorded table starts at the first reachable one); a bound that lets a
     * negative index through rejects the table. */
    check_seq_at("bias on the bound", seq_bound_bias, A0, TABLE - 4, SNC, 3, 1);
    check_seq_at("lower bound from blez", seq_blez, A0, TABLE - 4, SNC, 3, 1);
    check_seq_at("negative index", seq_negative, A0, TABLE, SNC, 0, 0);
    check_seq("negative lower bound", seq_negative_lower, A0, 0);
    check_seq_at("gcc-style load", seq_plain, A0, TABLE, GCC, 3, 0);

    /* Every way into the code between the bound and the jr must establish
     * the same bound. */
    g_slot = sltiu(A2, A0, 3);
    check_seq("merge, same bound in the slot", seq_merge, A0, 3);
    g_slot = sltiu(A2, A0, 2);
    check_seq("merge, tighter bound in the slot", seq_merge, A0, 3);
    check_seq_at("merge, no index gets through", seq_infeasible, A0, TABLE, SNC, 3, 0);
    g_slot = 0;
    check_seq("merge, stale bound", seq_merge, A0, 0);
    g_slot = sltiu(A2, A0, 7);
    check_seq("merge, different bound", seq_merge, A0, 0);

    /* Paths through several merges are checked as a whole: the bound must
     * hold however many edges a path takes, up to a depth of 8. */
    check_chain("two merges, same bound", 2, sltiu(A2, A0, 3), 1);
    check_chain("two merges, stale $a2", 2, ori(A2, 0, 1), 0);
    check_chain("eight merges, same bound", 8, sltiu(A2, A0, 3), 1);
    check_chain("nine merges: too deep to check", 9, sltiu(A2, A0, 3), 0);
    check_seq("merge, constant key on the edge", seq_const_key, A0, 3);
    check_state_machine();

    /* The block another function branches into is promoted to an entry, but
     * only so the emitter can reach it: its one way in is that branch, which
     * establishes the same bound, so the table stands. */
    {
        const uint32_t site = build_seq(seq_shared, A0, 0);
        asm_t as = { SEQ_OTHER };
        emit(&as, bne(A1, 0, as.pc, BASE + 8));
        emit(&as, sltiu(A2, A0, 3));
        emit(&as, JR_RA); emit(&as, 0);
        store();
        uint32_t seeds[2] = { SEQ_OTHER, BASE };    /* BASE is walked first */
        setup(&an, SEQ_TEXT);
        CHECK(a_discover(&an, seeds, 2) == 0, "discover (shared)");
        CHECK(an.nfuncs == 3, "shared block promoted, got %d functions", an.nfuncs);
        CHECK(a_find_jtable(&an, site) != NULL && an.nindirects == 0,
              "table behind a promoted block kept (%d unresolved)", an.nindirects);
        a_analysis_free(&an);
    }

    /* Calls and unconditional transfers end the straight line. */
    {
        const uint32_t far = BASE + 0x88, at = BASE + 4;   /* the transfer sits at BASE + 4 */
        check_stop("jr $ra, lui before", JR_RA, 0, 0);
        check_stop("jr $ra, lui in slot", JR_RA, 1, 0);
        check_stop("jr $ra, bound and lui before", JR_RA, 0, 1);
        check_stop("j, lui before", j_(far), 0, 0);
        check_stop("b, lui before", beq(0, 0, at, far), 0, 0);
        check_stop("bgez $zero, lui before", bgez(0, at, far), 0, 0);
        check_stop("beql $zero,$zero, lui before", beql(0, 0, at, far), 0, 0);
        check_stop("beql $zero,$zero, lui in slot", beql(0, 0, BASE, far), 1, 0);
    }

    /* In the real game every case target is already a function entry (the
     * data-pointer scan seeds them): it stays one, is still recorded. */
    {
        uint32_t seeds[2] = { BASE, BASE + 0x3C };
        build(BASE + 0x3C);
        setup(&an, 0x44);
        CHECK(a_discover(&an, seeds, 2) == 0, "discover (entry target)");
        const a_jtable *e = a_find_jtable(&an, BASE + 0x18);
        CHECK(e != NULL && e->count == 3 && an.jt_targets[e->first + 2] == BASE + 0x3C,
              "entry target recorded");
        CHECK(an.nfuncs == 2 && an.owner[0x3C / 4] == BASE + 0x3C,
              "entry target stays its own function (%d functions)", an.nfuncs);
        a_analysis_free(&an);
    }

    /* A re-walk (another function branches into the switch's default block,
     * so discovery promotes it and walks again) records the table once. */
    {
        const uint32_t site = build_seq(seq_likely, A1, 1);
        uint32_t seeds[2] = { BASE, SEQ_OTHER };
        setup(&an, SEQ_TEXT);
        CHECK(a_discover(&an, seeds, 2) == 0, "discover (re-walk)");
        CHECK(an.nfuncs == 3, "default block promoted, got %d functions", an.nfuncs);
        CHECK(an.njtables == 1 && an.njt_targets == 3,
              "table recorded once after the re-walk, got %d/%d", an.njtables, an.njt_targets);
        CHECK(a_find_jtable(&an, site) != NULL, "table found after the re-walk");
        a_analysis_free(&an);
    }

    printf("%s\n", failures ? "test_switch: FAILED" : "test_switch: ok");
    return failures ? 1 : 0;
}
