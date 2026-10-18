/* SNC switch tables: recovered during discovery, case blocks kept inside the
 * function. Synthetic code only, no game data. */
#include "analyze.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
