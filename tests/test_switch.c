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
#define JR_RA 0x03E00008u
#define AT 1
#define V0 2
#define A0 4
#define A1 5
#define A2 6
#define T1 9
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

static uint32_t build_seq(void (*seq)(asm_t *), int idx, int other) {
    asm_t as = { BASE };
    memset(g_w, 0, sizeof g_w);
    seq(&as);
    emit(&as, lui(AT, (TABLE + 0x8000u) >> 16));
    emit(&as, addu(AT, AT, idx));
    emit(&as, lw(AT, (int)(int16_t)(TABLE & 0xFFFF), AT));
    const uint32_t site = as.pc;
    emit(&as, jr(AT));
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
    store();
    return site;
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
/* Only a lower bound (`bne` on the slti): the table's size is unknown. */
static void seq_lower_only(asm_t *as) {
    emit(as, slti(A1, A0, 3));
    emit(as, bne(A1, 0, as->pc, SEQ_DEFAULT));
    emit(as, sll(A0, A0, 2));
}

static void setup(a_analysis *an, uint32_t text_size);

static void check_seq(const char *name, void (*seq)(asm_t *), int idx, uint32_t want) {
    a_analysis an;
    uint32_t seed = BASE;
    const uint32_t site = build_seq(seq, idx, 0);
    setup(&an, SEQ_TEXT);
    CHECK(a_discover(&an, &seed, 1) == 0, "%s: discover", name);
    const a_jtable *jt = a_find_jtable(&an, site);
    if (!want) {
        CHECK(jt == NULL && an.njtables == 0, "%s: no table expected", name);
        CHECK(an.nindirects == 1, "%s: jr left as a computed jump, got %d", name, an.nindirects);
    } else {
        CHECK(jt != NULL, "%s: table found", name);
        if (jt) CHECK(jt->count == want && jt->table == TABLE, "%s: count %u (want %u)",
                      name, jt->count, want);
        CHECK(an.nindirects == 0, "%s: no unresolved jump, got %d", name, an.nindirects);
    }
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
