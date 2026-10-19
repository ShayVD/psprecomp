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

static void test_rot_cmov_crsp(void) {
    float o[4];
    const float one = 1.0f, half = 0.5f;

    set4(0x01u, 1, &one);                        /* S010 = 1 quarter turn */
    psp_vrot(0x00u, 0x01u, 0x11u, 4);            /* [-s, c, 0, 0] */
    get(0x00u, 4, o);
    CHECK(o[0] == -1.0f && o[1] == 0.0f && o[2] == 0.0f && o[3] == 0.0f,
          "vrot.q [-s,c,0,0] at 1: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vrot(0x00u, 0x01u, 0x00u, 3);            /* [c, s, s] */
    get(0x00u, 3, o);
    CHECK(o[0] == 0.0f && o[1] == 1.0f && o[2] == 1.0f, "vrot.t [c,s,s] at 1: %g %g %g", o[0], o[1], o[2]);
    set4(0x01u, 1, &half);
    psp_vrot(0x00u, 0x01u, 0x04u, 2);            /* [c, s] at 45 degrees */
    get(0x00u, 2, o);
    CHECK(fabsf(o[0] - 0.70710678f) < 1e-6f && fabsf(o[1] - 0.70710678f) < 1e-6f, "vrot.p at 0.5");

    float v;
    const float two = 2.0f, three = 3.0f, neg = -1.0f;
    set4(0x01u, 1, &two); psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == -1.0f, "vcos(2) = -1 exactly, got %g", v);
    set4(0x01u, 1, &three); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == -1.0f, "vsin(3) = -1 exactly, got %g", v);
    set4(0x01u, 1, &neg); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == -1.0f, "vsin(-1) = -1, got %g", v);
    set4(0x01u, 1, &one); psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == 0.0f, "vcos(1) = 0 exactly, got %g", v);
    set4(0x01u, 1, &one); psp_vunary(PSP_VU_NSIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == -1.0f, "vnsin(1) = -1, got %g", v);

    /* vcmov: cc bit 2 set, bit 0 clear. */
    const float src[4] = { 1, 2, 3, 4 }, dst[4] = { 9, 9, 9, 9 };
    psp_cpu.vfpu_cc = 0x04u;
    set4(0x01u, 4, src); set4(0x00u, 4, dst);
    psp_vcmov(0x00u, 0x01u, 0x2u, 4);            /* vcmovt cc2: moves */
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[3] == 4, "vcmovt on a set bit moves");
    set4(0x00u, 4, dst);
    psp_vcmov(0x00u, 0x01u, 0x0u, 4);            /* vcmovt cc0: does not */
    get(0x00u, 4, o);
    CHECK(o[0] == 9 && o[3] == 9, "vcmovt on a clear bit keeps");
    psp_vcmov(0x00u, 0x01u, 0x8u, 4);            /* vcmovf cc0: moves */
    get(0x00u, 4, o);
    CHECK(o[0] == 1, "vcmovf on a clear bit moves");
    set4(0x00u, 4, dst);
    psp_cpu.vfpu_cc = 0x05u;                     /* lanes 0 and 2 */
    psp_vcmov(0x00u, 0x01u, 0x6u, 4);            /* vcmovt per lane */
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[1] == 9 && o[2] == 3 && o[3] == 9, "per-lane vcmovt: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(0x00u, 4, dst);
    psp_vcmov(0x00u, 0x01u, 0xEu, 4);            /* vcmovf per lane: the other two */
    get(0x00u, 4, o);
    CHECK(o[0] == 9 && o[1] == 2 && o[2] == 9 && o[3] == 4, "per-lane vcmovf: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(0x00u, 4, dst);
    psp_vcmov(0x00u, 0x01u, 0x7u, 4);            /* cc 7: never */
    get(0x00u, 4, o);
    CHECK(o[0] == 9, "cc 7 never moves");

    /* vcrsp: x cross y = z. vqmul: i * j = k. */
    const float x[3] = { 1, 0, 0 }, y[3] = { 0, 1, 0 };
    set4(0x01u, 3, x); set4(0x02u, 3, y);
    psp_vcrsp(0x00u, 0x01u, 0x02u);
    get(0x00u, 3, o);
    CHECK(o[0] == 0 && o[1] == 0 && o[2] == 1, "x cross y = z: %g %g %g", o[0], o[1], o[2]);
    const float qi[4] = { 1, 0, 0, 0 }, qj[4] = { 0, 1, 0, 0 }, q1[4] = { 0, 0, 0, 1 };
    set4(0x01u, 4, qi); set4(0x02u, 4, qj);
    psp_vqmul(0x00u, 0x01u, 0x02u);
    get(0x00u, 4, o);
    CHECK(o[0] == 0 && o[1] == 0 && o[2] == 1 && o[3] == 0, "i*j = k: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(0x02u, 4, q1);
    psp_vqmul(0x00u, 0x01u, 0x02u);
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[3] == 0, "i*1 = i");

    /* vsge / vslt, with a NaN lane. */
    const float a[4] = { 1, 2, NAN, 4 }, b[4] = { 2, 2, 0, 3 };
    set4(0x01u, 4, a); set4(0x02u, 4, b);
    psp_vsge(0x00u, 0x01u, 0x02u, 4);
    get(0x00u, 4, o);
    CHECK(o[0] == 0 && o[1] == 1 && o[2] == 0 && o[3] == 1, "vsge: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vslt(0x00u, 0x01u, 0x02u, 4);
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[1] == 0 && o[2] == 0 && o[3] == 0, "vslt: %g %g %g %g", o[0], o[1], o[2], o[3]);
}

int main(void) {
    psp_vfpu_reset();
    test_moves();
    test_prefix_untouched();
    test_rot_cmov_crsp();
    printf("%s\n", failures ? "test_vfpu_ops: FAILED" : "test_vfpu_ops: ok");
    return failures ? 1 : 0;
}
