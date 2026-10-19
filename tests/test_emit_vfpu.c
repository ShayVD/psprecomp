/* VFPU and FPU emission: one instruction per function, checked as text.
 * Synthetic code only, no game data. Later tasks add cases here. */
#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE 0x08804000u
static int failures;

/* Emit `word` at BASE, a nop (its delay slot, if it is a branch), then
 * jr $ra; nop twice, and return the generated C. A test branch targets
 * BASE+16, the second jr $ra (offset 3). */
static const char *emit_one(uint32_t word) {
    static char src[1 << 16];
    uint32_t w[8] = { word, 0, 0x03E00008u, 0, 0x03E00008u, 0, 0, 0 };
    uint8_t code[sizeof w];
    for (size_t i = 0; i < 8; i++)
        for (int b = 0; b < 4; b++) code[i * 4 + b] = (uint8_t)(w[i] >> (8 * b));
    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = sizeof code;
    uint32_t seed = BASE;
    src[0] = '\0';
    if (a_discover(&an, &seed, 1) != 0) return src;
    emit_opts o;
    memset(&o, 0, sizeof o);
    o.outdir = ".";
    o.prefix = "t_emit_vfpu";
    o.module = "synthetic";
    if (a_emit(&an, &o) == 0) {
        FILE *f = fopen("./t_emit_vfpu_funcs.c", "rb");
        if (f) { src[fread(src, 1, sizeof src - 1, f)] = '\0'; fclose(f); }
    }
    a_analysis_free(&an);
    return src;
}

static void expect(uint32_t word, const char *needle) {
    const char *src = emit_one(word);
    if (!strstr(src, needle)) { printf("FAIL 0x%08X: expected \"%s\"\n", word, needle); failures++; }
    if (strstr(src, "psp_unimplemented") || strstr(src, "unhandled branch")) {
        printf("FAIL 0x%08X: still untranslated\n", word);
        failures++;
    }
}

int main(void) {
    /* vcst takes its constant index from bits 20..16 (vimm5), not from vs.
     * pspas: vcst.s S000, VFPU_PI = 0xD0690000; the game's 2/pi is 0xD0650024. */
    expect(0xD0690000u, "psp_vcst(0, 9, 1);");
    expect(0xD0650024u, "psp_vcst(36, 5, 1);");

    remove("./t_emit_vfpu_funcs.c");
    remove("./t_emit_vfpu_funcs.h");
    remove("./t_emit_vfpu_imports.c");
    printf("%s\n", failures ? "test_emit_vfpu: FAILED" : "test_emit_vfpu: ok");
    return failures ? 1 : 0;
}
