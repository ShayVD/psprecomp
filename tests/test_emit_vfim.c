/* vfim with an integral value must still emit a valid C float expression.
 * Synthetic code only, no game data. */
#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE 0x08804000u

/* vfim.s S000, 0xBC00 (half -1.0) ; jr $ra ; nop */
static const uint32_t CODE[] = { 0xDF80BC00u, 0x03E00008u, 0x00000000u };

int main(void) {
    uint8_t code[sizeof CODE];
    for (size_t i = 0; i < sizeof CODE / 4; i++)
        for (int b = 0; b < 4; b++) code[i * 4 + b] = (uint8_t)(CODE[i] >> (8 * b));

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
    if (!strstr(src, "psp_vimm(0, (float)-1);")) {
        printf("FAIL expected psp_vimm(0, (float)-1);\n");
        failures++;
    }
    if (strstr(src, "-1f)")) { printf("FAIL invalid literal -1f\n"); failures++; }
    printf("%s\n", failures ? "test_emit_vfim: FAILED" : "test_emit_vfim: ok");
    return failures ? 1 : 0;
}
