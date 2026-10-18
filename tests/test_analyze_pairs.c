/* Address pairs: lui/addiu and lui/ori that build a code address become seeds.
 * Synthetic code only, no game data. */
#include "analyze.h"

#include <stdio.h>
#include <string.h>

#define BASE 0x08804000u

static uint32_t lui(int rt, uint32_t imm) { return 0x3C000000u | (uint32_t)rt << 16 | (imm & 0xFFFF); }
static uint32_t addiu(int rt, int rs, int imm) {
    return 0x24000000u | (uint32_t)rs << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
static uint32_t ori(int rt, int rs, uint32_t imm) {
    return 0x34000000u | (uint32_t)rs << 21 | (uint32_t)rt << 16 | (imm & 0xFFFF);
}
static uint32_t lw(int rt, int imm, int base) {
    return 0x8C000000u | (uint32_t)base << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
#define HI(a) (((a) + 0x8000u) >> 16)
#define LO(a) ((int)(int16_t)((a) & 0xFFFF))
#define JR_RA 0x03E00008u
#define A0 4
#define A1 5
#define T0 8

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

int main(void) {
    /* f0 builds three addresses: one with addiu, one with ori (a different
     * register for the low half), and one whose lui is clobbered first. A
     * fourth pair points outside the scan range. f1 and f2 are reachable only
     * through those pointers. */
    const uint32_t F1 = BASE + 0x40, F2 = BASE + 0x48, DATA = BASE + 0x80;
    uint32_t w[0x90 / 4];
    memset(w, 0, sizeof w);
    int i = 0;
    w[i++] = lui(A0, HI(F1));            /* 0x00 */
    w[i++] = addiu(A0, A0, LO(F1));      /* 0x04 -> F1 */
    w[i++] = lui(T0, F2 >> 16);          /* 0x08 */
    w[i++] = ori(A1, T0, F2 & 0xFFFF);   /* 0x0C -> F2 (rt != rs) */
    w[i++] = lui(A0, 0x0880);            /* 0x10 */
    w[i++] = lw(A0, 0, A0);              /* 0x14 clobbers $a0: not a pair */
    w[i++] = addiu(A1, A0, 0x4040);      /* 0x18 would be BASE+0x40 if accepted */
    w[i++] = lui(A0, HI(DATA));          /* 0x1C */
    w[i++] = addiu(A0, A0, LO(DATA));    /* 0x20 -> DATA: outside the scan range */
    w[i++] = JR_RA;                      /* 0x24 */
    w[i++] = 0;                          /* 0x28 */
    w[0x40 / 4] = JR_RA; w[0x44 / 4] = 0;   /* f1 */
    w[0x48 / 4] = JR_RA; w[0x4C / 4] = 0;   /* f2 */

    uint8_t code[sizeof w];
    for (size_t k = 0; k < sizeof w / 4; k++)
        for (int b = 0; b < 4; b++) code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = (uint32_t)sizeof code;
    an.scan_base = BASE;
    an.scan_size = 0x50;                 /* ".text"; DATA lies beyond it */

    uint32_t found[8];
    int n = a_scan_address_pairs(&an, found, 8);
    CHECK(n == 2, "expected 2 pair targets, got %d", n);
    int has_f1 = 0, has_f2 = 0;
    for (int k = 0; k < n && k < 8; k++) {
        if (found[k] == F1) has_f1 = 1;
        if (found[k] == F2) has_f2 = 1;
    }
    CHECK(has_f1, "lui/addiu target 0x%08X found", F1);
    CHECK(has_f2, "lui/ori target 0x%08X found", F2);
    CHECK(a_scan_address_pairs(&an, NULL, 0) == 2, "counting pass agrees");

    /* Seeded with the pairs, discovery finds f1 and f2 as functions. */
    uint32_t seeds[3] = { BASE, F1, F2 };
    CHECK(a_discover(&an, seeds, 3) == 0, "discover");
    int got = 0;
    for (int k = 0; k < an.nfuncs; k++)
        if (an.funcs[k].addr == F1 || an.funcs[k].addr == F2) got++;
    CHECK(got == 2, "both pointer-only functions discovered (%d)", got);
    a_analysis_free(&an);

    printf("%s\n", failures ? "test_analyze_pairs: FAILED" : "test_analyze_pairs: ok");
    return failures ? 1 : 0;
}
