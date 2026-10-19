/* VFPU and FPU emission: one instruction per function, checked as text.
 * Synthetic code only, no game data. Later tasks add cases here. */
#include "analyze.h"
#include "decode.h"
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
     * the trap names the instruction. The table test below covers every case
     * in emit_simple that does it. Loads, stores and prefixes cannot trap. */
    reject(0xD8000000u, "psp_cpu.pc =");                     /* lv.q */
    reject(0xDC0000E4u, "psp_cpu.pc =");                     /* vpfxs */

    /* mfv/mtv move one lane as bits (pspas: mfv $a0, S010; mtv $a0, S123) and
     * cannot trap, so they store no pc. */
    expect(0x48640001u, "r_a0 = psp_mfv(1);");
    expect(0x48E40066u, "psp_mtv(102, r_a0);");
    reject(0x48640001u, "psp_cpu.pc =");
    reject(0x48E40066u, "psp_cpu.pc =");

    /* vrot (code 0x11 = [-s,c,0,0]), vcmovf cc 6 (vimm5 0x0E), vcrsp.t, vqmul.q,
     * vsge, vslt. pspas words. */
    expect(0xF3B18180u, "psp_vrot(0, 1, 17, 4);");
    expect(0xD2AE8180u, "psp_vcmov(0, 1, 14, 4);");
    expect(0xF2828100u, "psp_vcrsp(0, 1, 2);");
    expect(0xF2828180u, "psp_vqmul(0, 1, 2);");
    expect(0x6F028180u, "psp_vsge(0, 1, 2, 4);");
    expect(0x6F828180u, "psp_vslt(0, 1, 2, 4);");

    /* Pipeline control. Only the canonical vsync immediate is a no-op: the
     * decoder turns every other 0xFFFFxxxx into vsync, and a data word
     * reached as code must stay loud. */
    expect(0xFFFF0000u, ";  /* vnop */");
    expect(0xFFFF0320u, ";  /* vsync */");
    expect(0xFFFF040Du, ";  /* vflush */");
    {
        const char *src = emit_one(0xFFFF0001u);
        if (!strstr(src, "psp_unimplemented(0x08804000u, \"vsync with an unexpected immediate\");")) {
            printf("FAIL 0xFFFF0001: expected psp_unimplemented for vsync\n");
            failures++;
        }
        if (strstr(src, "/* vsync */")) { printf("FAIL 0xFFFF0001: emitted as a no-op\n"); failures++; }
    }

    /* One representative word per emit_simple case that calls emit_vcall_pc.
     * The store must precede the runtime call. Each word is checked to decode
     * to the intended op, so a wrong word fails here and not by silently
     * testing something else. Words are built from decode.c's field layouts
     * (not run through pspas): major opcode, then bits 25..23 (VFPU0/1/5/7
     * families), or rs (VFPU4, bits 25..21) and rt (bits 20..16), or the
     * VFPU6 rs group; 0x8080 is the quad size. */
    {
        static const struct { uint32_t word; a_op op; const char *call; } T[] = {
            { 0x60008080u, A_VADD,  "psp_vadd("   },
            { 0x60808080u, A_VSUB,  "psp_vsub("   },
            { 0x64008080u, A_VMUL,  "psp_vmul("   },
            { 0x63808080u, A_VDIV,  "psp_vdiv("   },
            { 0x6D008080u, A_VMIN,  "psp_vmin("   },
            { 0x6D808080u, A_VMAX,  "psp_vmax("   },
            { 0x64808080u, A_VDOT,  "psp_vdot("   },
            { 0x65008080u, A_VSCL,  "psp_vscl("   },
            { 0x6C008080u, A_VCMP,  "psp_vcmp("   },
            { 0xD0008080u, A_VMOV,  "psp_vunary(" },
            { 0xD0038080u, A_VIDT,  "psp_vidt("   },
            { 0xD0690000u, A_VCST,  "psp_vcst("   },
            { 0xDF000001u, A_VIIM,  "psp_vimm("   },
            { 0xDF800001u, A_VFIM,  "psp_vimm("   },
            { 0xF0008080u, A_VMMUL, "psp_vmmul("  },
            { 0xF1808080u, A_VTFM4, "psp_vtfm("   },
            { 0xF2008080u, A_VMSCL, "psp_vmscl("  },
            { 0xF3838080u, A_VMIDT, "psp_vmidt("  },
            { 0xF3868080u, A_VMZERO,"psp_vmzero(" },
            { 0xF3878080u, A_VMONE, "psp_vmone("  },
            { 0xF3808080u, A_VMMOV, "psp_vmmov("  },
            { 0xF3B18180u, A_VROT,  "psp_vrot("   },
            { 0xD2AE8180u, A_VCMOV, "psp_vcmov("  },
            { 0xF2828100u, A_VCRSP, "psp_vcrsp("  },
            { 0xF2828180u, A_VQMUL, "psp_vqmul("  },
            { 0x6F028180u, A_VSGE,  "psp_vsge("   },
            { 0x6F828180u, A_VSLT,  "psp_vslt("   },
        };
        for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
            a_insn d;
            memset(&d, 0, sizeof d);
            a_decode(T[i].word, BASE, &d);
            if (d.op != T[i].op) {
                printf("FAIL 0x%08X: decodes to op %d, test expects %d\n",
                       T[i].word, (int)d.op, (int)T[i].op);
                failures++;
                continue;
            }
            expect_order(T[i].word, "psp_cpu.pc = 0x08804000u;", T[i].call);
        }
    }

    /* mfv to $zero is discarded; mfvc/mtvc (bit 7) are different ops and stay
     * unimplemented. */
    expect(0x48600001u, "*/\n    ;\n");   /* nothing but the disassembly comment */
    reject(0x48600001u, "psp_mfv");
    reject(0x48600001u, "psp_unimplemented");
    {
        const uint32_t w[2] = { 0x48640080u, 0x48E40080u };   /* mfvc, mtvc $a0 */
        for (int i = 0; i < 2; i++)
            if (!strstr(emit_one(w[i]), "psp_unimplemented")) {
                printf("FAIL 0x%08X: mfvc/mtvc should be unimplemented\n", w[i]);
                failures++;
            }
    }

    remove("./t_emit_vfpu_funcs.c");
    remove("./t_emit_vfpu_funcs.h");
    remove("./t_emit_vfpu_imports.c");
    printf("%s\n", failures ? "test_emit_vfpu: FAILED" : "test_emit_vfpu: ok");
    return failures ? 1 : 0;
}
