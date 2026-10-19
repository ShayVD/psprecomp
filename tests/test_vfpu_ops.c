/* VFPU ops added in P3. Synthetic values only. Tasks 11-13 add cases. */
#include "psprecomp/vfpu.h"
#include "psprecomp/recomp_rt.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                           printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

static inline void set4(uint32_t vreg, int n, const float *v) {
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
    float s;
    get(0x01u, 1, &s);
    CHECK(bits(s) == 0x80000000u, "mtv wrote S010");
}

int main(void) {
    (void)set4;
    psp_vfpu_reset();
    test_moves();
    printf("%s\n", failures ? "test_vfpu_ops: FAILED" : "test_vfpu_ops: ok");
    return failures ? 1 : 0;
}
