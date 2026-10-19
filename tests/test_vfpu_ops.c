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

static int g_traps;
static const char *g_trap_what;
static void count_trap(uint32_t addr, const char *what) { (void)addr; g_traps++; g_trap_what = what; }

/* vcmov has no vt operand, so a pending vpfxt is not ours to interpret:
 * trap, leave the destination alone, consume the prefixes. */
static void test_vcmov_prefixes(void) {
    float o[4];
    const float src[4] = { 1, 2, 3, 4 }, dst[4] = { 9, 9, 9, 9 };
    psp_vfpu_reset();
    psp_vfpu_set_trap_handler(count_trap);
    g_traps = 0;
    psp_cpu.vfpu_cc = 0x01u;
    set4(0x01u, 4, src); set4(0x00u, 4, dst);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x1B);
    psp_vcmov(0x00u, 0x01u, 0x0u, 4);                 /* would move every lane */
    get(0x00u, 4, o);
    CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vcmov with vpfxt"),
          "vcmov with vpfxt traps (%d)", g_traps);
    CHECK(o[0] == 9 && o[3] == 9, "trapped vcmov wrote nothing: %g %g", o[0], o[3]);
    CHECK(!psp_vfpu_prefix_pending(), "trapped vcmov consumed the prefixes");

    /* vpfxs applies to the source. */
    set4(0x00u, 4, dst);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);            /* [W, Z, Y, X] */
    psp_vcmov(0x00u, 0x01u, 0x0u, 4);
    get(0x00u, 4, o);
    CHECK(g_traps == 1 && o[0] == 4 && o[1] == 3 && o[2] == 2 && o[3] == 1,
          "vcmov with vpfxs: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* The destination prefix covers the lanes that do not move too: they are
     * rewritten through it (saturated, or skipped when masked). UNVERIFIED on
     * hardware. Lanes 0, 2, 3 move; lane 1 keeps 9, which sat0 turns into 1;
     * lane 2 is masked and keeps its 7. */
    const float dst2[4] = { 9, 9, 7, 9 }, src2[4] = { 2, 2, 3, 4 };
    psp_cpu.vfpu_cc = 0x0Du;
    set4(0x01u, 4, src2); set4(0x00u, 4, dst2);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x01u | (0x01u << 2) | (1u << 10));
    psp_vcmov(0x00u, 0x01u, 0x6u, 4);
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[1] == 1 && o[2] == 7 && o[3] == 4,
          "vcmov under vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vfpu_set_trap_handler(NULL);
    psp_vfpu_reset();
}

/* Operands with no symmetry, so a swapped term or a wrong sign shows. */
static void test_cross_quat_general(void) {
    float o[4];
    const float a3[3] = { 1, 2, 3 }, b3[3] = { 4, 5, 6 };
    set4(0x01u, 3, a3); set4(0x02u, 3, b3);
    psp_vcrsp(0x00u, 0x01u, 0x02u);
    get(0x00u, 3, o);
    CHECK(o[0] == -3 && o[1] == 6 && o[2] == -3, "(1,2,3)x(4,5,6): %g %g %g", o[0], o[1], o[2]);
    set4(0x01u, 3, a3); set4(0x02u, 3, b3);
    psp_vcrsp(0x01u, 0x01u, 0x02u);                   /* vd = vs */
    get(0x01u, 3, o);
    CHECK(o[0] == -3 && o[1] == 6 && o[2] == -3, "vcrsp vd=vs: %g %g %g", o[0], o[1], o[2]);
    set4(0x01u, 3, a3); set4(0x02u, 3, b3);
    psp_vcrsp(0x02u, 0x01u, 0x02u);                   /* vd = vt */
    get(0x02u, 3, o);
    CHECK(o[0] == -3 && o[1] == 6 && o[2] == -3, "vcrsp vd=vt: %g %g %g", o[0], o[1], o[2]);

    const float a4[4] = { 1, 2, 3, 4 }, b4[4] = { 5, 6, 7, 8 };
    set4(0x01u, 4, a4); set4(0x02u, 4, b4);
    psp_vqmul(0x00u, 0x01u, 0x02u);
    get(0x00u, 4, o);
    CHECK(o[0] == 24 && o[1] == 48 && o[2] == 48 && o[3] == -6,
          "(1,2,3,4)*(5,6,7,8): %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(0x01u, 4, a4); set4(0x02u, 4, b4);
    psp_vqmul(0x01u, 0x01u, 0x02u);
    get(0x01u, 4, o);
    CHECK(o[0] == 24 && o[1] == 48 && o[2] == 48 && o[3] == -6, "vqmul vd=vs: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(0x01u, 4, a4); set4(0x02u, 4, b4);
    psp_vqmul(0x02u, 0x01u, 0x02u);
    get(0x02u, 4, o);
    CHECK(o[0] == 24 && o[1] == 48 && o[2] == 48 && o[3] == -6, "vqmul vd=vt: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* The prefix semantics of these two are unverified and the game never
     * prefixes them: a pending vpfxs or vpfxt traps and writes nothing. The
     * destination prefix still applies. */
    psp_vfpu_reset();
    psp_vfpu_set_trap_handler(count_trap);
    const float zero[4] = { 0, 0, 0, 0 };
    for (int which = 0; which < 2; which++) {
        g_traps = 0;
        set4(0x01u, 3, a3); set4(0x02u, 3, b3); set4(0x00u, 4, zero);
        psp_vfpu_set_prefix(which, 0x1B);
        psp_vcrsp(0x00u, 0x01u, 0x02u);
        get(0x00u, 3, o);
        CHECK(g_traps == 1 && o[0] == 0 && o[1] == 0 && o[2] == 0 && !psp_vfpu_prefix_pending(),
              "vcrsp with prefix %d traps (%d): %g", which, g_traps, o[0]);
        g_traps = 0;
        set4(0x01u, 4, a4); set4(0x02u, 4, b4); set4(0x00u, 4, zero);
        psp_vfpu_set_prefix(which, 0x1B);
        psp_vqmul(0x00u, 0x01u, 0x02u);
        get(0x00u, 4, o);
        CHECK(g_traps == 1 && o[0] == 0 && o[3] == 0 && !psp_vfpu_prefix_pending(),
              "vqmul with prefix %d traps (%d): %g", which, g_traps, o[0]);
    }
    g_traps = 0;
    set4(0x01u, 3, a3); set4(0x02u, 3, b3); set4(0x00u, 4, zero);
    psp_vfpu_set_prefix(PSP_VPFX_D, 1u << 8);          /* mask lane 0 */
    psp_vcrsp(0x00u, 0x01u, 0x02u);
    get(0x00u, 3, o);
    CHECK(g_traps == 0 && o[0] == 0 && o[1] == 6 && o[2] == -3, "vcrsp under vpfxd: %g %g %g", o[0], o[1], o[2]);
    psp_vfpu_set_trap_handler(NULL);
    psp_vfpu_reset();
}

static void test_sincos_edges(void) {
    float v;
    const float tiny = -1e-30f, nz = -0.0f;
    set4(0x01u, 1, &tiny); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v / -1.5707963e-30f - 1.0f) < 1e-5f, "vsin(-1e-30) = %g", v);
    const float m10 = -1e-10f, p10 = 1e-10f;
    float vn, vp;
    set4(0x01u, 1, &m10); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &vn);
    set4(0x01u, 1, &p10); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &vp);
    CHECK(vn == -vp && vp > 0, "vsin(-1e-10) = -vsin(1e-10): %g %g", vn, vp);
    set4(0x01u, 1, &nz); psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == 0.0f && signbit(v), "vsin(-0) is -0: %g", v);
    psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == 1.0f, "vcos(-0) = 1: %g", v);
    const float big = 16777218.0f;                    /* 2 mod 4 */
    set4(0x01u, 1, &big);
    psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == -1.0f, "vcos(16777218) = -1: %g", v);
    psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(v == 0.0f, "vsin(16777218) = 0: %g", v);
    /* Negative non-integers: -0.5 is -45 degrees, -1.5 is -135. */
    const float h = -0.5f, h3 = -1.5f, m3 = -3.5f;
    set4(0x01u, 1, &h);
    psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v + 0.70710678f) < 1e-6f, "vsin(-0.5) = %g", v);
    psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v - 0.70710678f) < 1e-6f, "vcos(-0.5) = %g", v);
    set4(0x01u, 1, &h3);
    psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v + 0.70710678f) < 1e-6f, "vsin(-1.5) = %g", v);
    psp_vunary(PSP_VU_COS, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v + 0.70710678f) < 1e-6f, "vcos(-1.5) = %g", v);
    set4(0x01u, 1, &m3);                              /* -315 degrees = +45 */
    psp_vunary(PSP_VU_SIN, 0x02u, 0x01u, 1); get(0x02u, 1, &v);
    CHECK(fabsf(v - 0.70710678f) < 1e-6f, "vsin(-3.5) = %g", v);
}

static void test_rot_lanes(void) {
    float o[4];
    const float qt = 0.25f, half = 0.5f;               /* sin 22.5 = .38268, cos = .92388 */
    const float S = 0.38268343f, C = 0.92387953f;
    const float nine[4] = { 9, 9, 9, 9 };

    set4(0x01u, 1, &qt);
    psp_vrot(0x00u, 0x01u, 0x08u, 4);                  /* sine lane 2, cosine lane 0 */
    get(0x00u, 4, o);
    CHECK(fabsf(o[0] - C) < 1e-6f && o[1] == 0 && fabsf(o[2] - S) < 1e-6f && o[3] == 0,
          "vrot.q [c,0,s,0]: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vrot(0x00u, 0x01u, 0x0Du, 4);                  /* sine lane 3, cosine lane 1 */
    get(0x00u, 4, o);
    CHECK(o[0] == 0 && fabsf(o[1] - C) < 1e-6f && o[2] == 0 && fabsf(o[3] - S) < 1e-6f,
          "vrot.q [0,c,0,s]: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vrot(0x00u, 0x01u, 0x04u, 2);                  /* sine lane 1, cosine lane 0: [c, s] */
    get(0x00u, 2, o);
    CHECK(fabsf(o[0] - C) < 1e-6f && fabsf(o[1] - S) < 1e-6f, "vrot.p [c,s]: %g %g", o[0], o[1]);
    psp_vrot(0x00u, 0x01u, 0x01u, 2);                  /* sine lane 0, cosine lane 1: [s, c] */
    get(0x00u, 2, o);
    CHECK(fabsf(o[0] - S) < 1e-6f && fabsf(o[1] - C) < 1e-6f, "vrot.p [s,c]: %g %g", o[0], o[1]);

    set4(0x01u, 1, &half);
    psp_vrot(0x00u, 0x01u, 0x15u, 3);                  /* -s in lane 1, c in lane 1 too: [-s, c, -s] */
    get(0x00u, 3, o);
    CHECK(fabsf(o[0] + 0.70710678f) < 1e-6f && fabsf(o[1] - 0.70710678f) < 1e-6f &&
          fabsf(o[2] + 0.70710678f) < 1e-6f, "vrot.t 0x15: %g %g %g", o[0], o[1], o[2]);

    /* Lanes past the width are not written. */
    set4(0x00u, 4, nine);
    set4(0x01u, 1, &qt);
    psp_vrot(0x00u, 0x01u, 0x04u, 2);
    get(0x00u, 4, o);
    CHECK(o[2] == 9 && o[3] == 9, "vrot.p leaves lanes 2, 3: %g %g", o[2], o[3]);
    /* A lane code beyond the width writes nothing for it. */
    set4(0x00u, 4, nine);
    psp_vrot(0x00u, 0x01u, 0x0Cu, 3);                  /* sine lane 3 (beyond .t), cosine lane 0 */
    get(0x00u, 4, o);
    CHECK(fabsf(o[0] - C) < 1e-6f && o[1] == 0 && o[2] == 0 && o[3] == 9,
          "vrot.t sine lane beyond width: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* vd aliases the source: C000 = S000's column. */
    set4(0x00u, 1, &qt);
    psp_vrot(0x00u, 0x00u, 0x11u, 4);                  /* [-s, c, 0, 0] */
    get(0x00u, 4, o);
    CHECK(fabsf(o[0] + S) < 1e-6f && fabsf(o[1] - C) < 1e-6f && o[2] == 0 && o[3] == 0,
          "vrot C000,S000: %g %g %g %g", o[0], o[1], o[2], o[3]);
}

static void test_cmp_signed_zero(void) {
    float o[4];
    const float a[4] = { 0.0f, -0.0f, 0.0f, -0.0f }, b[4] = { -0.0f, 0.0f, 0.0f, -0.0f };
    set4(0x01u, 4, a); set4(0x02u, 4, b);
    psp_vsge(0x00u, 0x01u, 0x02u, 4);
    get(0x00u, 4, o);
    CHECK(o[0] == 1 && o[1] == 1 && o[2] == 1 && o[3] == 1, "vsge treats +-0 as equal: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vslt(0x00u, 0x01u, 0x02u, 4);
    get(0x00u, 4, o);
    CHECK(o[0] == 0 && o[1] == 0 && o[2] == 0 && o[3] == 0, "vslt +-0: %g %g %g %g", o[0], o[1], o[2], o[3]);
}

static void set_bits(uint32_t vreg, int n, const uint32_t *b) {
    int r[4];
    psp_vfpu_regs(vreg, n, r);
    for (int i = 0; i < n; i++) psp_cpu.v[r[i]] = psp_bits_to_f32(b[i]);
}
static void get_bits(uint32_t vreg, int n, uint32_t *b) {
    int r[4];
    psp_vfpu_regs(vreg, n, r);
    for (int i = 0; i < n; i++) b[i] = psp_f32_to_bits(psp_cpu.v[r[i]]);
}

/* psp_cvt_w is shared with the FPU: all four rounding modes (MIPS FCR31
 * encoding), ties, signs, and the saturating edges. */
static void test_cvt_w(void) {
    static const double  x[] = { -2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 2.75, -2.75 };
    static const int32_t n[] = { -2,   -2,    0,    0,     0,    0,   2,   2,   3,    -3 };   /* nearest-even */
    static const int32_t z[] = { -2,   -1,    0,    0,     0,    0,   1,   2,   2,    -2 };   /* toward zero */
    static const int32_t u[] = { -2,   -1,    0,    0,     1,    1,   2,   3,   3,    -2 };   /* toward +inf */
    static const int32_t d[] = { -3,   -2,   -1,   -1,     0,    0,   1,   2,   2,    -3 };   /* toward -inf */
    for (size_t i = 0; i < sizeof x / sizeof x[0]; i++) {
        CHECK(psp_cvt_w(x[i], 0) == n[i], "cvt_w nearest %g -> %d", x[i], (int)psp_cvt_w(x[i], 0));
        CHECK(psp_cvt_w(x[i], 1) == z[i], "cvt_w trunc %g -> %d", x[i], (int)psp_cvt_w(x[i], 1));
        CHECK(psp_cvt_w(x[i], 2) == u[i], "cvt_w ceil %g -> %d", x[i], (int)psp_cvt_w(x[i], 2));
        CHECK(psp_cvt_w(x[i], 3) == d[i], "cvt_w floor %g -> %d", x[i], (int)psp_cvt_w(x[i], 3));
    }
    for (int rm = 0; rm < 4; rm++) {
        CHECK(psp_cvt_w(NAN, rm) == INT32_MAX, "cvt_w NaN rm %d", rm);
        CHECK(psp_cvt_w(INFINITY, rm) == INT32_MAX, "cvt_w +inf rm %d", rm);
        CHECK(psp_cvt_w(-INFINITY, rm) == INT32_MIN, "cvt_w -inf rm %d", rm);
        CHECK(psp_cvt_w(3e9, rm) == INT32_MAX && psp_cvt_w(-3e9, rm) == INT32_MIN, "cvt_w range rm %d", rm);
        CHECK(psp_cvt_w(2147483647.0, rm) == INT32_MAX && psp_cvt_w(-2147483648.0, rm) == INT32_MIN,
              "cvt_w exact limits rm %d", rm);
    }
    CHECK(psp_cvt_w(2147483646.5, 0) == 2147483646, "cvt_w tie near the top rounds to even");
    CHECK(psp_cvt_w(-2147483648.5, 0) == INT32_MIN && psp_cvt_w(-2147483648.5, 2) == INT32_MIN, "cvt_w below min");
    CHECK(psp_cvt_w(1.0, 5) == 1 && psp_cvt_w(1.5, 4) == 2, "cvt_w uses only the low two mode bits");
}

static void test_conversions(void) {
    uint32_t o[4];
    const float f[4] = { 1.3f, -1.3f, 2.5f, -2.5f };
    set4(0x01u, 4, f);
    psp_vf2i(0, 0x00u, 0x01u, 0, 4);             /* vf2in: nearest-even */
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 1 && (int32_t)o[1] == -1 && o[2] == 2 && (int32_t)o[3] == -2, "vf2in");
    psp_vf2i(1, 0x00u, 0x01u, 3, 4);             /* vf2iz, scale 3: trunc(x * 8) */
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 10 && (int32_t)o[1] == -10 && o[2] == 20 && (int32_t)o[3] == -20, "vf2iz scale 3");
    psp_vf2i(2, 0x00u, 0x01u, 0, 2);             /* vf2iu: up */
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 2 && (int32_t)o[1] == -1, "vf2iu");
    psp_vf2i(3, 0x00u, 0x01u, 0, 2);             /* vf2id: down */
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 1 && (int32_t)o[1] == -2, "vf2id");
    psp_vf2i(0, 0x00u, 0x01u, 1, 4);             /* scale 1: 2.6 -> 3, 5.0 -> 5 */
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 3 && (int32_t)o[1] == -3 && o[2] == 5 && (int32_t)o[3] == -5, "vf2in scale 1: %d %d %d %d",
          (int)o[0], (int)o[1], (int)o[2], (int)o[3]);
    const float big[4] = { 3e9f, -3e9f, NAN, INFINITY };
    set4(0x01u, 4, big);
    psp_vf2i(1, 0x00u, 0x01u, 0, 4);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 0x7FFFFFFFu && o[1] == 0x80000000u && o[2] == 0x7FFFFFFFu && o[3] == 0x7FFFFFFFu,
          "saturation: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);
    const float sat[2] = { 1.0e9f, -1.0e9f };    /* in range unscaled, out of range at scale 2 */
    set4(0x01u, 2, sat);
    psp_vf2i(1, 0x00u, 0x01u, 2, 2);
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 0x7FFFFFFFu && o[1] == 0x80000000u, "scaled saturation: %08X %08X", o[0], o[1]);

    const uint32_t ints[2] = { 0x00010000u, 0xFFFF0000u };
    set_bits(0x01u, 2, ints);
    psp_vi2f(0x00u, 0x01u, 16, 2);               /* vi2f scale 16 */
    float g[2];
    get(0x00u, 2, g);
    CHECK(g[0] == 1.0f && g[1] == -1.0f, "vi2f scale 16: %g %g", g[0], g[1]);
    psp_vi2f(0x00u, 0x01u, 0, 2);
    get(0x00u, 2, g);
    CHECK(g[0] == 65536.0f && g[1] == -65536.0f, "vi2f scale 0: %g %g", g[0], g[1]);
    const uint32_t imin = 0x80000000u;
    set_bits(0x01u, 1, &imin);
    psp_vi2f(0x00u, 0x01u, 31, 1);
    get(0x00u, 1, g);
    CHECK(g[0] == -1.0f, "vi2f of INT_MIN at scale 31: %g", g[0]);

    /* Packing. Lane 2 of the first source is negative with bits 23..30 set, so
     * the unsigned narrowings only give 0 for it through the negative clamp. */
    const uint32_t q[4] = { 0x7F800000u, 0x00800000u, 0xFF800000u, 0x3FFFFFFFu };
    set_bits(0x01u, 4, q);
    psp_vpack(PSP_VP_I2UC, 0x00u, 0x01u, 4);
    get_bits(0x00u, 1, o);
    CHECK(o[0] == 0x7F0001FFu, "vi2uc: 0x%08X", o[0]);
    psp_vpack(PSP_VP_I2C, 0x00u, 0x01u, 4);
    get_bits(0x00u, 1, o);
    CHECK(o[0] == 0x3FFF007Fu, "vi2c: 0x%08X", o[0]);

    const uint32_t packed = 0x80FF0001u;
    set_bits(0x01u, 1, &packed);
    psp_vpack(PSP_VP_UC2I, 0x00u, 0x01u, 1);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 0x00808080u && o[1] == 0 && o[2] == 0x7FFFFFFFu && o[3] == 0x40404040u,
          "vuc2i: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);
    psp_vpack(PSP_VP_C2I, 0x00u, 0x01u, 1);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 0x01000000u && o[1] == 0 && o[2] == 0xFF000000u && o[3] == 0x80000000u,
          "vc2i: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);

    const uint32_t halves = 0x8000FFFFu;
    set_bits(0x01u, 1, &halves);
    psp_vpack(PSP_VP_S2I, 0x00u, 0x01u, 1);
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 0xFFFF0000u && o[1] == 0x80000000u, "vs2i: %08X %08X", o[0], o[1]);
    psp_vpack(PSP_VP_US2I, 0x00u, 0x01u, 1);
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 0x7FFF8000u && o[1] == 0x40000000u, "vus2i: %08X %08X", o[0], o[1]);

    /* The pair forms widen to a quad (the game's vs2i has two source lanes). */
    const uint32_t hp[2] = { 0x00018002u, 0x7FFF0004u };
    set_bits(0x01u, 2, hp);
    psp_vpack(PSP_VP_S2I, 0x00u, 0x01u, 2);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 0x80020000u && o[1] == 0x00010000u && o[2] == 0x00040000u && o[3] == 0x7FFF0000u,
          "vs2i.p: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);
    psp_vpack(PSP_VP_US2I, 0x00u, 0x01u, 2);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 0x40010000u && o[1] == 0x00008000u && o[2] == 0x00020000u && o[3] == 0x3FFF8000u,
          "vus2i.p: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);

    const uint32_t two[2] = { 0x12345678u, 0xFEDCBA98u };
    set_bits(0x01u, 2, two);
    psp_vpack(PSP_VP_I2S, 0x00u, 0x01u, 2);
    get_bits(0x00u, 1, o);
    CHECK(o[0] == 0xFEDC1234u, "vi2s.p: 0x%08X", o[0]);
    psp_vpack(PSP_VP_I2US, 0x00u, 0x01u, 2);
    get_bits(0x00u, 1, o);
    CHECK(o[0] == 0x00002468u, "vi2us.p (negative lane clamps to 0): 0x%08X", o[0]);
    const uint32_t four[4] = { 0x12345678u, 0xFEDCBA98u, 0x7FFFFFFFu, 0x00010000u };
    set_bits(0x01u, 4, four);
    psp_vpack(PSP_VP_I2S, 0x00u, 0x01u, 4);
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 0xFEDC1234u && o[1] == 0x00017FFFu, "vi2s.q: %08X %08X", o[0], o[1]);
    psp_vpack(PSP_VP_I2US, 0x00u, 0x01u, 4);
    get_bits(0x00u, 2, o);
    CHECK(o[0] == 0x00002468u && o[1] == 0x0002FFFFu, "vi2us.q: %08X %08X", o[0], o[1]);
}

/* Prefix policy for the conversion and packing ops. The game has none of them
 * under a vpfxs/vpfxt (counted in the generated code), so: vf2i* applies
 * vpfxs element-wise (UNVERIFIED); vi2f and the packing ops trap on vpfxs,
 * since neg/abs/constants on integer bits are unverified; none has a vt
 * operand, so a pending vpfxt traps. A vpfxd masks the destination lanes (an
 * integer result is never saturated). */
static void test_conversion_prefixes(void) {
    uint32_t o[4];
    float g[4];
    psp_vfpu_reset();
    psp_vfpu_set_trap_handler(count_trap);

    const float f[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    set4(0x01u, 4, f);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);                 /* [W, Z, Y, X] */
    psp_vf2i(1, 0x00u, 0x01u, 0, 4);
    get_bits(0x00u, 4, o);
    CHECK(o[0] == 4 && o[1] == 3 && o[2] == 2 && o[3] == 1 && !psp_vfpu_prefix_pending(),
          "vf2iz with vpfxs: %u %u %u %u", o[0], o[1], o[2], o[3]);

    const uint32_t keep[4] = { 77, 77, 77, 77 };
    const uint32_t ints[4] = { 1, 2, 3, 4 };
    g_traps = 0;
    set_bits(0x00u, 4, keep); set_bits(0x01u, 4, ints);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x1B);
    psp_vf2i(1, 0x00u, 0x01u, 0, 4);
    get_bits(0x00u, 4, o);
    CHECK(g_traps == 1 && o[0] == 77 && !psp_vfpu_prefix_pending(),
          "vf2i with vpfxt traps and writes nothing (%d)", g_traps);

    g_traps = 0;
    {
        const float nine[4] = { 9, 9, 9, 9 };
        set4(0x00u, 4, nine);
    }
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);
    psp_vi2f(0x00u, 0x01u, 0, 4);
    get(0x00u, 4, g);
    CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vi2f with vpfxs/vpfxt") && g[0] == 9 &&
          !psp_vfpu_prefix_pending(), "vi2f with vpfxs traps (%d)", g_traps);
    g_traps = 0;
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x1B);
    psp_vi2f(0x00u, 0x01u, 0, 4);
    CHECK(g_traps == 1, "vi2f with vpfxt traps (%d)", g_traps);

    /* vpfxd on vi2f is an ordinary float write: saturation applies. */
    g_traps = 0;
    {
        const uint32_t pm[2] = { 5u, (uint32_t)-5 };
        set_bits(0x01u, 2, pm);
    }
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x0001u | 0x0004u);    /* sat 0:1 on lanes 0 and 1 */
    psp_vi2f(0x00u, 0x01u, 0, 2);
    get(0x00u, 2, g);
    CHECK(g_traps == 0 && g[0] == 1.0f && g[1] == 0.0f, "vi2f under vpfxd saturates: %g %g", g[0], g[1]);

    for (int w = 0; w < 2; w++) {
        const int which = w ? PSP_VPFX_T : PSP_VPFX_S;
        const uint32_t src[4] = { 0x7F800000u, 0, 0, 0 };
        g_traps = 0;
        set_bits(0x00u, 4, keep);
        set_bits(0x01u, 4, src);
        psp_vfpu_set_prefix(which, 0x1B);
        psp_vpack(PSP_VP_I2UC, 0x00u, 0x01u, 4);
        get_bits(0x00u, 1, o);
        CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vpack with vpfxs/vpfxt") && o[0] == 77 &&
              !psp_vfpu_prefix_pending(), "vpack with prefix %d traps and writes nothing (%d)", which, g_traps);
    }

    /* vpfxd masks packed lanes: vuc2i writes a quad, lane 1 masked. */
    g_traps = 0;
    {
        const uint32_t all = 0xFFFFFFFFu;
        set_bits(0x00u, 4, keep);
        set_bits(0x01u, 1, &all);
    }
    psp_vfpu_set_prefix(PSP_VPFX_D, 1u << 9);
    psp_vpack(PSP_VP_UC2I, 0x00u, 0x01u, 1);
    get_bits(0x00u, 4, o);
    CHECK(g_traps == 0 && o[0] == 0x7FFFFFFFu && o[1] == 77 && o[2] == 0x7FFFFFFFu && o[3] == 0x7FFFFFFFu,
          "vuc2i under vpfxd mask: %08X %08X %08X %08X", o[0], o[1], o[2], o[3]);

    /* A width the op does not have traps rather than reading garbage. */
    g_traps = 0;
    set_bits(0x00u, 4, keep);
    psp_vpack(PSP_VP_I2UC, 0x00u, 0x01u, 2);
    psp_vpack(PSP_VP_UC2I, 0x00u, 0x01u, 2);
    psp_vpack(99, 0x00u, 0x01u, 1);
    get_bits(0x00u, 4, o);
    CHECK(g_traps == 3 && o[0] == 77, "bad vpack widths trap (%d)", g_traps);

    psp_vfpu_set_trap_handler(NULL);
    psp_vfpu_reset();
}

int main(void) {
    psp_vfpu_reset();
    test_moves();
    test_prefix_untouched();
    test_rot_cmov_crsp();
    test_vcmov_prefixes();
    test_cross_quat_general();
    test_sincos_edges();
    test_rot_lanes();
    test_cmp_signed_zero();
    test_cvt_w();
    test_conversions();
    test_conversion_prefixes();
    printf("%s\n", failures ? "test_vfpu_ops: FAILED" : "test_vfpu_ops: ok");
    return failures ? 1 : 0;
}
