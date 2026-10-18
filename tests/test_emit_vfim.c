/* vfim with an integral value must still emit a valid C float expression.
 * Synthetic code only, no game data. */
#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE 0x08804000u

/* vd is the register index the emitter prints for vt = case index; these
 * values were confirmed from the emitted output, not assumed to equal vt.
 * Each case: vfim.s with vt = index, half-float imm16, expected binary32 bits.
 * Followed by jr $ra ; nop. */
static const struct { uint16_t half; uint32_t bits; unsigned vd; } CASES[] = {
    { 0xBC00, 0xBF800000u, 0 },  /* -1 */
    { 0x8000, 0x80000000u, 0 },  /* -0 */
    { 0x3800, 0x3F000000u, 0 },  /* 0.5 */
    { 0x7C00, 0x7F800000u, 0 },  /* inf */
    { 0x7E00, 0x7FC00000u, 0 },  /* NaN */
    { 0x0001, 0x33800000u, 1 },  /* smallest denormal, 2^-24 */
    { 0x03FF, 0x387FC000u, 127 },  /* largest denormal */
};
#define NCASES (sizeof CASES / sizeof CASES[0])

int main(void) {
    uint32_t words[NCASES + 2];
    for (size_t i = 0; i < NCASES; i++)
        words[i] = 0xDF800000u | ((uint32_t)i << 16) | CASES[i].half;
    words[NCASES] = 0x03E00008u;      /* jr $ra */
    words[NCASES + 1] = 0;            /* nop */
    uint8_t code[sizeof words];
    for (size_t i = 0; i < sizeof words / 4; i++)
        for (int b = 0; b < 4; b++) code[i * 4 + b] = (uint8_t)(words[i] >> (8 * b));

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = (uint32_t)sizeof code;
    uint32_t seed = BASE;
    if (a_discover(&an, &seed, 1) != 0 || an.nfuncs != 1) { printf("FAIL discovery\n"); return 1; }

    emit_opts o;
    memset(&o, 0, sizeof o);
    o.outdir = ".";
    o.prefix = "t_vfim";
    o.module = "synthetic";
    if (a_emit(&an, &o) != 0) { printf("FAIL emit\n"); return 1; }

    FILE *f = fopen("./t_vfim_funcs.c", "rb");
    if (!f) { printf("FAIL open\n"); return 1; }
    static char src[1 << 16];
    src[fread(src, 1, sizeof src - 1, f)] = '\0';
    fclose(f);

    int failures = 0;
    for (size_t i = 0; i < NCASES; i++) {
        char want[96];
        snprintf(want, sizeof want, "psp_vimm(%u, psp_bits_to_f32(0x%08Xu));",
                 CASES[i].vd, CASES[i].bits);
        if (!strstr(src, want)) {
            printf("FAIL half 0x%04X: expected %s\n", CASES[i].half, want);
            failures++;
        }
    }
    remove("./t_vfim_funcs.c");
    remove("./t_vfim_funcs.h");
    remove("./t_vfim_imports.c");
    printf("%s\n", failures ? "test_emit_vfim: FAILED" : "test_emit_vfim: ok");
    return failures ? 1 : 0;
}
