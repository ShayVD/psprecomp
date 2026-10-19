/* Jump-table words among the heuristic data-pointer seeds: a_discover_pruned
 * drops them once their table is recovered, so the cases are blocks of the
 * switch's function instead of functions of their own. Synthetic code only,
 * no game data. */
#include "analyze.h"

#include <stdio.h>
#include <string.h>

#define BASE  0x08804000u
#define TABLE (BASE + 0x100u)        /* "rodata": in the image, past .text */
#define TEXT  0x44u

static uint32_t R(int op, int rs, int rt, int imm) {
    return (uint32_t)op << 26 | (uint32_t)rs << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
#define JR_RA 0x03E00008u

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static uint8_t g_code[0x110];

/* The SNC idiom at BASE; cases at 0x20, 0x2C, 0x3C, default at 0x34. The
 * table's first entry is `case0`. */
static void build(uint32_t case0) {
    uint32_t w[0x110 / 4];
    memset(w, 0, sizeof w);
    w[0x00 / 4] = R(0x0B, 4, 6, 3);                              /* sltiu a2,a0,3 */
    w[0x04 / 4] = R(0x04, 6, 0, (0x34 - 0x08) / 4);              /* beq a2,zero,default */
    w[0x08 / 4] = 4u << 16 | 4u << 11 | 2u << 6;                 /* sll a0,a0,2 */
    w[0x0C / 4] = R(0x0F, 0, 1, (int)((TABLE + 0x8000u) >> 16)); /* lui at */
    w[0x10 / 4] = 1u << 21 | 4u << 16 | 1u << 11 | 0x21;         /* addu at,at,a0 */
    w[0x14 / 4] = R(0x23, 1, 1, (int)(int16_t)(TABLE & 0xFFFF)); /* lw at,lo(at) */
    w[0x18 / 4] = 1u << 21 | 0x08;                               /* jr at */
    w[0x20 / 4] = R(0x0D, 0, 2, 10); w[0x24 / 4] = JR_RA;        /* case 0 */
    w[0x2C / 4] = JR_RA;             w[0x30 / 4] = R(0x0D, 0, 2, 11);
    w[0x34 / 4] = JR_RA;             w[0x38 / 4] = R(0x0D, 0, 2, 0);
    w[0x3C / 4] = JR_RA;             w[0x40 / 4] = R(0x0D, 0, 2, 12);
    w[0x100 / 4] = case0;
    w[0x104 / 4] = BASE + 0x2C;
    w[0x108 / 4] = BASE + 0x3C;
    w[0x10C / 4] = BASE + 0x3C;       /* the same address again, outside the table */
    for (size_t k = 0; k < sizeof w / 4; k++)
        for (int b = 0; b < 4; b++) g_code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));
}

static void setup(a_analysis *an) {
    memset(an, 0, sizeof *an);
    an->code = g_code;
    an->base = BASE;
    an->size = sizeof g_code;
    an->scan_base = BASE;
    an->scan_size = TEXT;
    an->image = g_code;
    an->image_base = BASE;
    an->image_size = sizeof g_code;
}

/* Two switches with the same idiom, `sltiu a2,a0,3; beq a2,zero,0x80;
 * sll a1,a0,2; lui/addu/lw at; jr at`: S1 at `s1`, S2 at `s2`, tables T1 at
 * 0xD0 and T2 at 0xE0 (entries t1[], t2[]), cases `jr ra; nop` at `cases`. */
static void put(uint32_t *w, uint32_t off, uint32_t v) { w[off / 4] = v; }
static void put_switch(uint32_t *w, uint32_t at, uint32_t table) {
    put(w, at + 0x00, R(0x0B, 4, 6, 3));
    put(w, at + 0x04, R(0x04, 6, 0, (int)((0x80 - (at + 0x08)) / 4)));
    put(w, at + 0x08, 4u << 16 | 5u << 11 | 2u << 6);                  /* sll a1,a0,2 */
    put(w, at + 0x0C, R(0x0F, 0, 1, (int)((table + 0x8000u) >> 16)));
    put(w, at + 0x10, 1u << 21 | 5u << 16 | 1u << 11 | 0x21);          /* addu at,at,a1 */
    put(w, at + 0x14, R(0x23, 1, 1, (int)(int16_t)(table & 0xFFFF)));
    put(w, at + 0x18, 1u << 21 | 0x08);
}
static void build2(uint32_t s1, uint32_t s2, const uint32_t t1[3], const uint32_t t2[3],
                   const uint32_t *cases, int ncases) {
    uint32_t w[0x110 / 4];
    memset(w, 0, sizeof w);
    put_switch(w, s1, BASE + 0xD0);
    put_switch(w, s2, BASE + 0xE0);
    for (int i = 0; i < ncases; i++) put(w, cases[i] - BASE, JR_RA);
    put(w, 0x80, JR_RA);
    for (int k = 0; k < 3; k++) { w[0xD0 / 4 + k] = t1[k]; w[0xE0 / 4 + k] = t2[k]; }
    for (size_t k = 0; k < sizeof w / 4; k++)
        for (int b = 0; b < 4; b++) g_code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));
}

static int is_func(const a_analysis *an, uint32_t a) {
    for (int i = 0; i < an->nfuncs; i++) if (an->funcs[i].addr == a) return 1;
    return 0;
}

int main(void) {
    a_analysis an;
    a_prune_stats ps;
    a_data_seed ds[8];
    int nds;

    /* The scanner reports where each pointer was read. */
    build(BASE + 0x20);
    setup(&an);
    nds = a_scan_data_seeds(&an, g_code + 0x100, 0x10, TABLE, ds, 8);
    CHECK(nds == 4, "four code pointers in the data, got %d", nds);
    CHECK(nds >= 2 && ds[0].value == BASE + 0x20 && ds[0].source == TABLE &&
          ds[1].value == BASE + 0x2C && ds[1].source == TABLE + 4, "values and sources");

    /* The three table words are dropped: the cases are blocks of BASE. */
    {
        uint32_t seed = BASE;
        CHECK(a_discover_pruned(&an, &seed, 1, ds, 3, &ps) == 0, "discover");
        CHECK(an.nfuncs == 1, "cases inside the switch's function, got %d functions", an.nfuncs);
        CHECK(an.njtables == 1, "table recovered, got %d", an.njtables);
        CHECK(ps.dropped == 3 && ps.restored == 0, "dropped %d restored %d", ps.dropped, ps.restored);
        CHECK(ps.passes == 2, "two passes, got %d", ps.passes);
        CHECK(an.owner[0x20 / 4] == BASE && an.owner[0x2C / 4] == BASE && an.owner[0x3C / 4] == BASE,
              "cases owned by BASE");
        a_analysis_free(&an);
    }

    /* A value also read from outside any table (0x3C at TABLE + 0xC) or
     * seeded another way (0x2C) stays an entry. */
    {
        uint32_t seeds[2] = { BASE, BASE + 0x2C };
        setup(&an);
        CHECK(a_discover_pruned(&an, seeds, 2, ds, 4, &ps) == 0, "discover (other sources)");
        CHECK(ps.dropped == 1, "only case 0 dropped, got %d", ps.dropped);
        CHECK(an.nfuncs == 3 && is_func(&an, BASE + 0x2C) && is_func(&an, BASE + 0x3C) &&
              an.owner[0x20 / 4] == BASE, "0x2C and 0x3C stay entries (%d functions)", an.nfuncs);
        CHECK(an.njtables == 1, "table still recovered");
        a_analysis_free(&an);
    }

    /* Case 0 is the switch function's own entry, and nothing else seeds it:
     * dropping the table words leaves the switch unreached. Every dropped
     * seed that is no longer reached is restored, and the result is the
     * unpruned one. */
    {
        build(BASE);
        setup(&an);
        nds = a_scan_data_seeds(&an, g_code + 0x100, 0x0C, TABLE, ds, 8);
        CHECK(a_discover_pruned(&an, NULL, 0, ds, nds, &ps) == 0, "discover (restore)");
        CHECK(ps.dropped == 0 && ps.restored == 3, "dropped %d restored %d", ps.dropped, ps.restored);
        CHECK(ps.passes == 3, "three passes, got %d", ps.passes);
        CHECK(an.njtables == 1 && is_func(&an, BASE) && an.owner[0x2C / 4] != A_NO_OWNER &&
              an.owner[0x3C / 4] != A_NO_OWNER, "switch and cases reached");
        a_analysis_free(&an);
    }

    /* A table that only stands once another table's words are dropped. S1's
     * first case is S2's beq (S2's sltiu sits right after S1's jr). Seeded
     * from the data, that case is an entry: entered from anywhere, S2's beq
     * tests a $a2 nothing bounded, so S2 is rejected. Without the seed the
     * case is reached only through S1's jr, whose path carries S1's bound on
     * the same $a0 -- S2 stands, and its words go in a third pass. */
    {
        const uint32_t t1[3] = { BASE + 0x24, BASE + 0x50, BASE + 0x58 };
        const uint32_t t2[3] = { BASE + 0x60, BASE + 0x68, BASE + 0x70 };
        const uint32_t cases[5] = { BASE + 0x50, BASE + 0x58, BASE + 0x60, BASE + 0x68, BASE + 0x70 };
        build2(0x00, 0x20, t1, t2, cases, 5);
        setup(&an);
        an.scan_size = 0x88;
        nds = a_scan_data_seeds(&an, g_code + 0xD0, 0x20, BASE + 0xD0, ds, 8);
        uint32_t seed = BASE;
        CHECK(a_discover_pruned(&an, &seed, 1, ds, nds, &ps) == 0, "discover (multi-pass)");
        CHECK(ps.passes == 3 && ps.dropped == 6 && ps.restored == 0,
              "multi-pass: passes %d dropped %d restored %d", ps.passes, ps.dropped, ps.restored);
        CHECK(an.njtables == 2 && an.nfuncs == 1, "multi-pass: %d tables, %d functions",
              an.njtables, an.nfuncs);
        a_analysis_free(&an);
    }

    /* One case shared by switches in two functions (0x60 in both tables):
     * dropped, it is walked by one and promoted to an entry by the other. */
    {
        const uint32_t t1[3] = { BASE + 0x40, BASE + 0x48, BASE + 0x60 };
        const uint32_t t2[3] = { BASE + 0x50, BASE + 0x58, BASE + 0x60 };
        const uint32_t cases[5] = { BASE + 0x40, BASE + 0x48, BASE + 0x50, BASE + 0x58, BASE + 0x60 };
        build2(0x00, 0x20, t1, t2, cases, 5);
        setup(&an);
        an.scan_size = 0x88;
        nds = a_scan_data_seeds(&an, g_code + 0xD0, 0x20, BASE + 0xD0, ds, 8);
        uint32_t seeds[2] = { BASE, BASE + 0x20 };
        CHECK(a_discover_pruned(&an, seeds, 2, ds, nds, &ps) == 0, "discover (shared case)");
        CHECK(ps.dropped == 6 && ps.restored == 0, "shared: dropped %d restored %d",
              ps.dropped, ps.restored);
        /* So is the shared default (0x80): BASE, 0x20, 0x60, 0x80. */
        CHECK(an.njtables == 2 && is_func(&an, BASE + 0x60) && an.nfuncs == 4,
              "shared: the shared case is an entry (%d tables, %d functions)", an.njtables, an.nfuncs);
        CHECK(an.owner[0x40 / 4] == BASE && an.owner[0x50 / 4] == BASE + 0x20,
              "shared: the other cases are blocks of their switch");
        a_analysis_free(&an);
    }

    printf("%s\n", failures ? "test_prune: FAILED" : "test_prune: ok");
    return failures ? 1 : 0;
}
