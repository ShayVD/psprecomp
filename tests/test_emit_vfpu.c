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

/* `needle` must not appear in the C generated for `word`. */
static void reject(uint32_t word, const char *needle) {
    if (strstr(emit_one(word), needle)) {
        printf("FAIL 0x%08X: unexpected \"%s\"\n", word, needle);
        failures++;
    }
}

/* `first` appears, and `then` appears after it. */
static void expect_order(uint32_t word, const char *first, const char *then) {
    const char *src = emit_one(word);
    const char *a = strstr(src, first);
    if (!a || !strstr(a, then)) {
        printf("FAIL 0x%08X: expected \"%s\" before \"%s\"\n", word, first, then);
        failures++;
    }
}

int main(void) {
    /* vcst takes its constant index from bits 20..16 (vimm5), not from vs.
     * pspas: vcst.s S000, VFPU_PI = 0xD0690000; the game's 2/pi is 0xD0650024. */
    expect(0xD0690000u, "psp_vcst(0, 9, 1);");
    expect(0xD0650024u, "psp_vcst(36, 5, 1);");

    /* bvf 0 / bvt 2 / bvtl 5, all to BASE+16 (offset 3). The vfpu_cc bit is
     * bits 20..18 (vimm5 >> 2), the type bits 17..16. pspas, .set noreorder,
     * label four instructions ahead: 0x49000003, 0x49090003, 0x49170003. */
    expect(0x49000003u, "!((psp_cpu.vfpu_cc >> 0) & 1)");
    expect(0x49090003u, "((psp_cpu.vfpu_cc >> 2) & 1)");
    expect(0x49170003u, "((psp_cpu.vfpu_cc >> 5) & 1)");
    reject(0x49090003u, "!((psp_cpu.vfpu_cc");
    reject(0x49170003u, "!((psp_cpu.vfpu_cc");
    expect(0x49020003u, "!((psp_cpu.vfpu_cc >> 0) & 1)");   /* bvfl 0 */

    /* A VFPU op that can trap stores its own address in psp_cpu.pc first, so
     * the trap names the instruction. Loads, stores and prefixes cannot trap. */
    expect_order(0x60008080u, "psp_cpu.pc = 0x08804000u;", "psp_vadd(0, 0, 0, 4);");
    expect_order(0xF3838080u, "psp_cpu.pc = 0x08804000u;", "psp_vmidt(0, 4);");   /* vmidt.q M000 */
    expect_order(0xD0690000u, "psp_cpu.pc = 0x08804000u;", "psp_vcst(0, 9, 1);");
    reject(0xD8000000u, "psp_cpu.pc =");                     /* lv.q */
    reject(0xDC0000E4u, "psp_cpu.pc =");                     /* vpfxs */

    remove("./t_emit_vfpu_funcs.c");
    remove("./t_emit_vfpu_funcs.h");
    remove("./t_emit_vfpu_imports.c");
    printf("%s\n", failures ? "test_emit_vfpu: FAILED" : "test_emit_vfpu: ok");
    return failures ? 1 : 0;
}
