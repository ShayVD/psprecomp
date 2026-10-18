/* --split: three tiny functions into two files plus a registration file.
 * Synthetic code only, no game data. */
#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE 0x08804000u

/* f0: jal f1 ; nop ; jr ra ; nop     f1: jal f2 ; nop ; jr ra ; nop     f2: jr ra ; nop */
static const uint32_t CODE[] = {
    0x0C000000u | ((BASE + 16) >> 2 & 0x3FFFFFF), 0, 0x03E00008u, 0,
    0x0C000000u | ((BASE + 32) >> 2 & 0x3FFFFFF), 0, 0x03E00008u, 0,
    0x03E00008u, 0,
};

static int failures;

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    static char buf[4][1 << 16];
    static int k;
    char *b = buf[k++ & 3];
    b[fread(b, 1, (1 << 16) - 1, f)] = '\0';
    fclose(f);
    return b;
}

static void expect(int cond, const char *what) {
    if (!cond) { printf("FAIL %s\n", what); failures++; }
}

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
    expect(a_discover(&an, &seed, 1) == 0 && an.nfuncs == 3, "three functions discovered");

    emit_opts o;
    memset(&o, 0, sizeof o);
    o.outdir = ".";
    o.prefix = "t_split";
    o.module = "synthetic";
    o.split = 2;
    expect(a_emit(&an, &o) == 0, "emit succeeds");

    char *p0 = slurp("./t_split_funcs_000.c");
    char *p1 = slurp("./t_split_funcs_001.c");
    char *reg = slurp("./t_split_register.c");
    expect(p0 && p1 && reg, "all three files exist");
    if (p0 && p1 && reg) {
        expect(strstr(p0, "void psp_func_08804000(void)") != NULL, "f0 in part 0");
        expect(strstr(p0, "void psp_func_08804010(void)") != NULL, "f1 in part 0");
        expect(strstr(p1, "void psp_func_08804020(void)") != NULL, "f2 in part 1");
        expect(strstr(p1, "void psp_func_08804000(void) {") == NULL, "f0 not repeated in part 1");
        expect(strstr(reg, "void psp_recomp_register(void)") != NULL, "register function");
        expect(strstr(reg, "{ 0x08804020u, psp_func_08804020 },") != NULL, "f2 registered");
        expect(strstr(reg, "{ 0u, 0 }") != NULL, "sentinel in register tables");
        expect(strstr(p0, "psp_recomp_register") == NULL && strstr(p0, "psp_recomp_funcs") == NULL, "no registration in parts");
    }
    /* split larger than the function count: empty parts still exist */
    o.prefix = "t_split5";
    o.split = 5;
    expect(a_emit(&an, &o) == 0, "emit split=5 succeeds");
    for (int i = 3; i < 5; i++) {
        char nm[64];
        snprintf(nm, sizeof nm, "./t_split5_funcs_%03d.c", i);
        char *p = slurp(nm);
        expect(p != NULL, "empty part exists");
        if (p) {
            expect(strstr(p, "#include \"t_split5_funcs.h\"") != NULL, "empty part has include");
            expect(strstr(p, "psp_func_") == NULL, "empty part has no functions");
        }
        remove(nm);
    }
    remove("./t_split5_funcs_000.c");
    remove("./t_split5_funcs_001.c");
    remove("./t_split5_funcs_002.c");
    remove("./t_split5_register.c");
    remove("./t_split5_funcs.h");
    remove("./t_split5_imports.c");

    /* out-of-range split is rejected */
    o.prefix = "t_split1000";
    o.split = 1000;
    expect(a_emit(&an, &o) == -1, "split=1000 rejected");
    remove("./t_split1000_funcs.h");
    remove("./t_split1000_imports.c");

    remove("./t_split_funcs_000.c");
    remove("./t_split_funcs_001.c");
    remove("./t_split_register.c");
    remove("./t_split_funcs.h");
    remove("./t_split_imports.c");
    printf("%s\n", failures ? "test_emit_split: FAILED" : "test_emit_split: ok");
    return failures ? 1 : 0;
}
