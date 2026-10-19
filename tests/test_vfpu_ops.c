/* VFPU ops added in P3. Synthetic values only. Tasks 11-13 add cases. */
#include "psprecomp/vfpu.h"
#include "psprecomp/recomp_rt.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                           printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

static void set4(uint32_t vreg, int n, const float *v) {
    int r[4];
    psp_vfpu_regs(vreg, n, r);
    for (int i = 0; i < n; i++) psp_cpu.v[r[i]] = v[i];
}
static void get(uint32_t vreg, int n, float *out) {
    int r[4];
    psp_vfpu_regs(vreg, n, r);
    for (int i = 0; i < n; i++) out[i] = psp_cpu.v[r[i]];
}
static uint32_t bits(float f) { return psp_f32_to_bits(f); }

static void test_moves(void) {
    /* mtv/mfv move bits, not values: a NaN payload and -0 survive. */
    psp_mtv(0x66u, 0x7FC00001u);
    CHECK(psp_mfv(0x66u) == 0x7FC00001u, "NaN payload round trip: 0x%08X", psp_mfv(0x66u));
    psp_mtv(0x01u, 0x80000000u);
    CHECK(psp_mfv(0x01u) == 0x80000000u, "-0 round trip");
    /* A signalling NaN keeps its quiet bit clear. */
    psp_mtv(0x02u, 0x7F800001u);
    CHECK(psp_mfv(0x02u) == 0x7F800001u, "sNaN round trip: 0x%08X", psp_mfv(0x02u));
    float s;
    get(0x01u, 1, &s);
    CHECK(bits(s) == 0x80000000u, "mtv wrote S010");
}

/* mfv/mtv do not consult or consume a pending prefix. */
static void test_prefix_untouched(void) {
    float o[4];
    const float in[4] = { 1, 2, 3, 4 };
    set4(0x00u, 4, in);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);                 /* [W, Z, Y, X] */
    psp_mtv(0x66u, 0x3F800000u);
    (void)psp_mfv(0x66u);
    CHECK(psp_vfpu_prefix_pending(), "mfv/mtv consumed the pending prefix");
    psp_vunary(PSP_VU_MOV, 0x02u, 0x00u, 4);
    get(0x02u, 4, o);
    CHECK(o[0] == 4 && o[1] == 3 && o[2] == 2 && o[3] == 1,
          "prefix applied after mtv: %g %g %g %g", o[0], o[1], o[2], o[3]);
    CHECK(!psp_vfpu_prefix_pending(), "prefix consumed by the vmov");
}

int main(void) {
    psp_vfpu_reset();
    test_moves();
    test_prefix_untouched();
    printf("%s\n", failures ? "test_vfpu_ops: FAILED" : "test_vfpu_ops: ok");
    return failures ? 1 : 0;
}
