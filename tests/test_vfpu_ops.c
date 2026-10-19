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
    /* Denormal inputs are not flushed: vf2iu of a denormal float is 1. */
    CHECK(psp_cvt_w((double)1e-40f, 2) == 1 && psp_cvt_w((double)-1e-40f, 3) == -1, "cvt_w denormal");
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

    /* vf2i* traps on vpfxs too: its semantics on this op are unverified and
     * the game never uses one. The destination stays untouched. */
    const float f[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const uint32_t keep[4] = { 77, 77, 77, 77 };
    g_traps = 0;
    set4(0x01u, 4, f);
    set_bits(0x00u, 4, keep);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);                 /* [W, Z, Y, X] */
    psp_vf2i(1, 0x00u, 0x01u, 0, 4);
    get_bits(0x00u, 4, o);
    CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vf2i with vpfxs/vpfxt") &&
          o[0] == 77 && o[1] == 77 && o[2] == 77 && o[3] == 77 && !psp_vfpu_prefix_pending(),
          "vf2iz with vpfxs traps and writes nothing (%d): %u %u %u %u", g_traps, o[0], o[1], o[2], o[3]);

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

/* Destination addressing: the result width differs from the source width, and
 * vd 0x20/0x40 are where single, pair and quad addressing differ. Every lane
 * of the register file starts at 77; only the result lanes may change. */
static void fill77(void) {
    for (size_t i = 0; i < sizeof psp_cpu.v / sizeof psp_cpu.v[0]; i++) psp_cpu.v[i] = psp_bits_to_f32(77u);
}
static float snap[128];                       /* the register file once the source is in place */
static void snapshot(void) { memcpy(snap, psp_cpu.v, sizeof snap); }
static int only_lanes_changed(uint32_t vd, int n, const uint32_t *want, char *why, size_t whylen) {
    int r[4], mine[128] = { 0 };
    psp_vfpu_regs(vd, n, r);
    for (int i = 0; i < n; i++) {
        mine[r[i]] = 1;
        if (psp_f32_to_bits(psp_cpu.v[r[i]]) != want[i]) {
            snprintf(why, whylen, "lane %d is 0x%08X, want 0x%08X", i, psp_f32_to_bits(psp_cpu.v[r[i]]), want[i]);
            return 0;
        }
    }
    for (size_t i = 0; i < sizeof psp_cpu.v / sizeof psp_cpu.v[0]; i++)
        if (!mine[i] && psp_f32_to_bits(psp_cpu.v[i]) != psp_f32_to_bits(snap[i])) {
            snprintf(why, whylen, "register %zu outside the result was written", i);
            return 0;
        }
    return 1;
}

static void test_pack_destination_lanes(void) {
    static const uint32_t vds[] = { 0x00u, 0x20u, 0x40u, 0x60u, 0x22u };
    char why[96];
    psp_vfpu_reset();
    for (size_t k = 0; k < sizeof vds / sizeof vds[0]; k++) {
        const uint32_t vd = vds[k], vs = 0x0Cu;                  /* vs clear of every vd above */
        {   /* vi2uc.q: one result word */
            const uint32_t q[4] = { 0x7F800000u, 0x00800000u, 0xFF800000u, 0x3FFFFFFFu }, want[1] = { 0x7F0001FFu };
            fill77(); set_bits(vs, 4, q); snapshot();
            psp_vpack(PSP_VP_I2UC, vd, vs, 4);
            CHECK(only_lanes_changed(vd, 1, want, why, sizeof why), "vi2uc.q into 0x%02X: %s", vd, why);
        }
        {   /* vi2s.q: two result words */
            const uint32_t q[4] = { 0x12345678u, 0xFEDCBA98u, 0x7FFFFFFFu, 0x00010000u }, want[2] = { 0xFEDC1234u, 0x00017FFFu };
            fill77(); set_bits(vs, 4, q); snapshot();
            psp_vpack(PSP_VP_I2S, vd, vs, 4);
            CHECK(only_lanes_changed(vd, 2, want, why, sizeof why), "vi2s.q into 0x%02X: %s", vd, why);
        }
        {   /* vuc2i.s: a single word widens to four */
            const uint32_t one[1] = { 0x80FF0001u }, want[4] = { 0x00808080u, 0, 0x7FFFFFFFu, 0x40404040u };
            fill77(); set_bits(vs, 1, one); snapshot();
            psp_vpack(PSP_VP_UC2I, vd, vs, 1);
            CHECK(only_lanes_changed(vd, 4, want, why, sizeof why), "vuc2i.s into 0x%02X: %s", vd, why);
        }
        {   /* vs2i.s: a single word widens to two */
            const uint32_t one[1] = { 0x8000FFFFu }, want[2] = { 0xFFFF0000u, 0x80000000u };
            fill77(); set_bits(vs, 1, one); snapshot();
            psp_vpack(PSP_VP_S2I, vd, vs, 1);
            CHECK(only_lanes_changed(vd, 2, want, why, sizeof why), "vs2i.s into 0x%02X: %s", vd, why);
        }
    }
}

/* ---- VFPU9 lane ops and the random-number ops ----------------------------- */

static void lane_expect(const char *what, const float *o, const float *want, int n) {
    for (int i = 0; i < n; i++)
        CHECK(bits(o[i]) == bits(want[i]), "%s lane %d: got %g, want %g", what, i, o[i], want[i]);
}

static void test_lane_ops(void) {
    float o[4];
    psp_vfpu_reset();

    /* Two operand sets, so that every compared pair is ascending in one and
     * descending in the other: a min/max swap or a wrong lane pairing shows. */
    static const float A[4] = { 1, 4, 2, 8 }, B[4] = { 9, 3, 7, 5 };
    static const struct { int op; const char *name; float want[2][4]; } Q[] = {
        { PSP_VL_OCP,  "vocp.q",  { {  0, -3, -1, -7 }, { -8, -2, -6, -4 } } },
        { PSP_VL_BFY1, "vbfy1.q", { {  5, -3, 10, -6 }, { 12,  6, 12,  2 } } },
        { PSP_VL_BFY2, "vbfy2.q", { {  3, 12, -1, -4 }, { 16,  8,  2, -2 } } },
        { PSP_VL_SRT1, "vsrt1.q", { {  1,  4,  2,  8 }, {  3,  9,  5,  7 } } },
        { PSP_VL_SRT2, "vsrt2.q", { {  1,  2,  4,  8 }, {  5,  3,  7,  9 } } },
        { PSP_VL_SRT3, "vsrt3.q", { {  4,  1,  8,  2 }, {  9,  3,  7,  5 } } },
        { PSP_VL_SRT4, "vsrt4.q", { {  8,  4,  2,  1 }, {  9,  7,  3,  5 } } },
        { PSP_VL_SGN,  "vsgn.q",  { {  1,  1,  1,  1 }, {  1,  1,  1,  1 } } },
    };
    for (size_t k = 0; k < sizeof Q / sizeof Q[0]; k++)
        for (int set = 0; set < 2; set++) {
            set4(0x01u, 4, set ? B : A);
            psp_vlane(Q[k].op, 0x00u, 0x01u, 4);
            get(0x00u, 4, o);
            lane_expect(Q[k].name, o, Q[k].want[set], 4);
        }

    /* vfad and vavg: one lane. */
    set4(0x01u, 4, A);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 4);  get(0x00u, 1, o);  CHECK(o[0] == 15, "vfad.q %g", o[0]);
    psp_vlane(PSP_VL_AVG, 0x00u, 0x01u, 4);  get(0x00u, 1, o);  CHECK(o[0] == 3.75f, "vavg.q %g", o[0]);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 3);  get(0x00u, 1, o);  CHECK(o[0] == 7, "vfad.t %g", o[0]);
    psp_vlane(PSP_VL_AVG, 0x00u, 0x01u, 3);  get(0x00u, 1, o);  CHECK(fabsf(o[0] - 7.0f / 3.0f) < 1e-6f, "vavg.t %g", o[0]);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 2);  get(0x00u, 1, o);  CHECK(o[0] == 5, "vfad.p %g", o[0]);
    psp_vlane(PSP_VL_AVG, 0x00u, 0x01u, 2);  get(0x00u, 1, o);  CHECK(o[0] == 2.5f, "vavg.p %g", o[0]);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 1);  get(0x00u, 1, o);  CHECK(o[0] == 1, "vfad.s %g", o[0]);
    psp_vlane(PSP_VL_AVG, 0x00u, 0x01u, 1);  get(0x00u, 1, o);  CHECK(o[0] == 1, "vavg.s %g", o[0]);

    /* Narrower forms of vocp, vbfy1, vsgn. */
    psp_vlane(PSP_VL_OCP, 0x00u, 0x01u, 3);  get(0x00u, 3, o);
    CHECK(o[0] == 0 && o[1] == -3 && o[2] == -1, "vocp.t");
    psp_vlane(PSP_VL_BFY1, 0x00u, 0x01u, 2); get(0x00u, 2, o);
    CHECK(o[0] == 5 && o[1] == -3, "vbfy1.p %g %g", o[0], o[1]);

    /* vsgn: sign by value, NaN and both zeros give 0. */
    const float g[4] = { -2, 0, 3, NAN };
    set4(0x01u, 4, g);
    psp_vlane(PSP_VL_SGN, 0x00u, 0x01u, 4);  get(0x00u, 4, o);
    CHECK(o[0] == -1 && o[1] == 0 && o[2] == 1 && o[3] == 0 && bits(o[3]) == 0, "vsgn: %g %g %g %g", o[0], o[1], o[2], o[3]);
    const float z[2] = { -0.0f, -INFINITY };
    set4(0x01u, 2, z);
    psp_vlane(PSP_VL_SGN, 0x00u, 0x01u, 2);  get(0x00u, 2, o);
    CHECK(o[0] == 0 && o[1] == -1, "vsgn of -0 and -inf: %g %g", o[0], o[1]);

    /* vsocp: width 2n, [sat0(1-s), sat0(s)] per source lane. */
    const float h = 0.25f;
    set4(0x01u, 1, &h);
    psp_vlane(PSP_VL_SOCP, 0x00u, 0x01u, 1); get(0x00u, 2, o);
    CHECK(o[0] == 0.75f && o[1] == 0.25f, "vsocp.s %g %g", o[0], o[1]);
    const float pq[2] = { 0.25f, 3.0f };
    set4(0x01u, 2, pq);
    psp_vlane(PSP_VL_SOCP, 0x00u, 0x01u, 2); get(0x00u, 4, o);
    CHECK(o[0] == 0.75f && o[1] == 0.25f && o[2] == 0.0f && o[3] == 1.0f,
          "vsocp.p %g %g %g %g", o[0], o[1], o[2], o[3]);
    const float neg = -2.0f;
    set4(0x01u, 1, &neg);
    psp_vlane(PSP_VL_SOCP, 0x00u, 0x01u, 1); get(0x00u, 2, o);
    CHECK(o[0] == 1.0f && o[1] == 0.0f, "vsocp.s of -2: %g %g", o[0], o[1]);

    /* NaN in a vfad sum stays NaN; -0 survives a single-lane sum. */
    const float nn[2] = { 1, NAN };
    set4(0x01u, 2, nn);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 2);  get(0x00u, 1, o);
    CHECK(isnan(o[0]), "vfad with a NaN lane");
    const float mz = -0.0f;
    set4(0x01u, 1, &mz);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 1);  get(0x00u, 1, o);
    CHECK(bits(o[0]) == 0x80000000u, "vfad.s keeps -0: 0x%08X", bits(o[0]));
}

/* Every lane of the register file starts at 77; only the result lanes may
 * change, wherever the destination sits, and the source (clear of every
 * destination) is read whole before anything is written. */
static void test_lane_destinations(void) {
    static const uint32_t vds[] = { 0x00u, 0x20u, 0x40u, 0x60u, 0x22u };
    char why[96];
    psp_vfpu_reset();
    for (size_t k = 0; k < sizeof vds / sizeof vds[0]; k++) {
        const uint32_t vd = vds[k], vs = 0x0Cu;
        const float q[4] = { 1, 4, 2, 8 };
        uint32_t want[4];
        {   /* vfad.q: one lane */
            fill77(); set4(vs, 4, q); snapshot();
            psp_vlane(PSP_VL_FAD, vd, vs, 4);
            want[0] = bits(15.0f);
            CHECK(only_lanes_changed(vd, 1, want, why, sizeof why), "vfad.q into 0x%02X: %s", vd, why);
        }
        {   /* vavg.t: one lane */
            fill77(); set4(vs, 3, q); snapshot();
            psp_vlane(PSP_VL_AVG, vd, vs, 3);
            want[0] = bits(7.0f / 3.0f);
            CHECK(only_lanes_changed(vd, 1, want, why, sizeof why), "vavg.t into 0x%02X: %s", vd, why);
        }
        {   /* vocp.t: three lanes */
            fill77(); set4(vs, 3, q); snapshot();
            psp_vlane(PSP_VL_OCP, vd, vs, 3);
            want[0] = bits(0); want[1] = bits(-3); want[2] = bits(-1);
            CHECK(only_lanes_changed(vd, 3, want, why, sizeof why), "vocp.t into 0x%02X: %s", vd, why);
        }
        {   /* vsocp.s: a pair from one lane */
            fill77(); set4(vs, 1, q); snapshot();
            psp_vlane(PSP_VL_SOCP, vd, vs, 1);
            want[0] = bits(0.0f); want[1] = bits(1.0f);                 /* sat0(1-1), sat0(1) */
            CHECK(only_lanes_changed(vd, 2, want, why, sizeof why), "vsocp.s into 0x%02X: %s", vd, why);
        }
        {   /* vsocp.p: a quad from two lanes */
            const float p[2] = { 0.5f, 0.125f };
            fill77(); set4(vs, 2, p); snapshot();
            psp_vlane(PSP_VL_SOCP, vd, vs, 2);
            want[0] = bits(0.5f); want[1] = bits(0.5f); want[2] = bits(0.875f); want[3] = bits(0.125f);
            CHECK(only_lanes_changed(vd, 4, want, why, sizeof why), "vsocp.p into 0x%02X: %s", vd, why);
        }
        {   /* vbfy1.p: a pair */
            fill77(); set4(vs, 2, q); snapshot();
            psp_vlane(PSP_VL_BFY1, vd, vs, 2);
            want[0] = bits(5); want[1] = bits(-3);
            CHECK(only_lanes_changed(vd, 2, want, why, sizeof why), "vbfy1.p into 0x%02X: %s", vd, why);
        }
        {   /* vsrt3.q: a quad */
            fill77(); set4(vs, 4, q); snapshot();
            psp_vlane(PSP_VL_SRT3, vd, vs, 4);
            want[0] = bits(4); want[1] = bits(1); want[2] = bits(8); want[3] = bits(2);
            CHECK(only_lanes_changed(vd, 4, want, why, sizeof why), "vsrt3.q into 0x%02X: %s", vd, why);
        }
    }

    /* Aliasing: the destination is (part of) the source. */
    float o[4];
    const float q[4] = { 1, 4, 2, 8 };
    set4(0x00u, 4, q);
    psp_vlane(PSP_VL_BFY1, 0x00u, 0x00u, 4); get(0x00u, 4, o);
    CHECK(o[0] == 5 && o[1] == -3 && o[2] == 10 && o[3] == -6, "vbfy1.q in place");
    set4(0x00u, 4, q);
    psp_vlane(PSP_VL_BFY2, 0x00u, 0x00u, 4); get(0x00u, 4, o);
    CHECK(o[0] == 3 && o[1] == 12 && o[2] == -1 && o[3] == -4, "vbfy2.q in place");
    set4(0x00u, 4, q);
    psp_vlane(PSP_VL_SRT4, 0x00u, 0x00u, 4); get(0x00u, 4, o);
    CHECK(o[0] == 8 && o[1] == 4 && o[2] == 2 && o[3] == 1, "vsrt4.q in place");
    set4(0x00u, 4, q);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x00u, 4); get(0x00u, 4, o);
    CHECK(o[0] == 15 && o[1] == 4 && o[2] == 2 && o[3] == 8, "vfad.q into its own first lane: %g %g %g %g", o[0], o[1], o[2], o[3]);
    const float one = 0.25f;
    set4(0x00u, 1, &one);
    psp_vlane(PSP_VL_SOCP, 0x00u, 0x00u, 1); get(0x00u, 2, o);
    CHECK(o[0] == 0.75f && o[1] == 0.25f, "vsocp.s in place: %g %g", o[0], o[1]);
}

/* Prefix policy. The game has one vfad with a pending vpfxs (a sum with lanes
 * 1 and 3 negated) and one vocp under a vpfxd (sat0 on lane 0, the rest
 * masked); every other site has none (counted in the generated code). So: the
 * destination prefix applies everywhere, vpfxs applies to vfad only
 * (UNVERIFIED on hardware), and everything else traps and writes nothing. */
static void test_lane_prefixes(void) {
    float o[4];
    static const float src[4] = { 1, 4, 2, 8 }, keep[4] = { 9, 9, 9, 9 };
    psp_vfpu_reset();
    psp_vfpu_set_trap_handler(count_trap);

    /* vfad.q under the game's vpfxs 0x0A00E4: x - y + z - w. */
    g_traps = 0;
    set4(0x01u, 4, src);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x0A00E4u);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 4); get(0x00u, 1, o);
    CHECK(g_traps == 0 && o[0] == -9 && !psp_vfpu_prefix_pending(), "vfad under vpfxs: %g (%d traps)", o[0], g_traps);

    /* vocp.s under the game's vpfxd 0xE01: 1 - 2 = -1 saturates to 0; with .s
     * only lane 0 exists, so the mask bits are moot. */
    const float two = 2.0f, qtr = 0.25f;
    set4(0x01u, 1, &two);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x000E01u);
    psp_vlane(PSP_VL_OCP, 0x00u, 0x01u, 1); get(0x00u, 1, o);
    CHECK(g_traps == 0 && bits(o[0]) == bits(0.0f) && !psp_vfpu_prefix_pending(), "vocp.s under vpfxd: %g", o[0]);
    set4(0x01u, 1, &qtr);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x000E01u);
    psp_vlane(PSP_VL_OCP, 0x00u, 0x01u, 1); get(0x00u, 1, o);
    CHECK(o[0] == 0.75f, "vocp.s of 0.25 under vpfxd: %g", o[0]);

    /* vpfxd masks lanes of a wider result and saturates the rest. */
    set4(0x01u, 4, src);
    set4(0x00u, 4, keep);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x01u | (1u << 9));    /* sat0 lane 0, mask lane 1 */
    psp_vlane(PSP_VL_OCP, 0x00u, 0x01u, 4); get(0x00u, 4, o);
    CHECK(g_traps == 0 && o[0] == 0 && o[1] == 9 && o[2] == -1 && o[3] == -7,
          "vocp.q under vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    psp_vfpu_set_prefix(PSP_VPFX_D, 1u << 8);              /* mask lane 0 of the single result */
    set4(0x00u, 1, keep);
    psp_vlane(PSP_VL_FAD, 0x00u, 0x01u, 4); get(0x00u, 1, o);
    CHECK(g_traps == 0 && o[0] == 9, "vfad under a masking vpfxd: %g", o[0]);

    /* Everything else traps on vpfxs or vpfxt, leaves vd alone, consumes. */
    static const struct { int op; int size; int which; const char *name; } T[] = {
        { PSP_VL_FAD,  4, PSP_VPFX_T, "vfad+vpfxt"  }, { PSP_VL_AVG,  4, PSP_VPFX_S, "vavg+vpfxs"  },
        { PSP_VL_AVG,  4, PSP_VPFX_T, "vavg+vpfxt"  }, { PSP_VL_OCP,  4, PSP_VPFX_S, "vocp+vpfxs"  },
        { PSP_VL_OCP,  4, PSP_VPFX_T, "vocp+vpfxt"  }, { PSP_VL_SOCP, 2, PSP_VPFX_S, "vsocp+vpfxs" },
        { PSP_VL_BFY1, 4, PSP_VPFX_S, "vbfy1+vpfxs" }, { PSP_VL_BFY2, 4, PSP_VPFX_T, "vbfy2+vpfxt" },
        { PSP_VL_SRT1, 4, PSP_VPFX_S, "vsrt1+vpfxs" }, { PSP_VL_SRT4, 4, PSP_VPFX_T, "vsrt4+vpfxt" },
        { PSP_VL_SGN,  4, PSP_VPFX_S, "vsgn+vpfxs"  },
    };
    for (size_t k = 0; k < sizeof T / sizeof T[0]; k++) {
        g_traps = 0;
        g_trap_what = NULL;
        set4(0x01u, 4, src);
        set4(0x00u, 4, keep);
        psp_vfpu_set_prefix(T[k].which, 0x1B);
        psp_vlane(T[k].op, 0x00u, 0x01u, T[k].size);
        get(0x00u, 4, o);
        CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vlane with vpfxs/vpfxt") &&
              o[0] == 9 && o[1] == 9 && o[2] == 9 && o[3] == 9 && !psp_vfpu_prefix_pending(),
              "%s traps and writes nothing (%d): %g %g %g %g", T[k].name, g_traps, o[0], o[1], o[2], o[3]);
    }

    /* A width the op does not have traps rather than reading garbage. */
    static const struct { int op; int size; } W[] = {
        { PSP_VL_SOCP, 3 }, { PSP_VL_SOCP, 4 }, { PSP_VL_BFY1, 1 }, { PSP_VL_BFY1, 3 }, { PSP_VL_BFY2, 2 },
        { PSP_VL_SRT1, 3 }, { PSP_VL_SRT2, 2 }, { PSP_VL_SRT3, 1 }, { PSP_VL_SRT4, 3 }, { PSP_VL_FAD, 0 },
        { PSP_VL_FAD, 5 },
    };
    for (size_t k = 0; k < sizeof W / sizeof W[0]; k++) {
        g_traps = 0;
        g_trap_what = NULL;
        set4(0x01u, 4, src);
        set4(0x00u, 4, keep);
        psp_vlane(W[k].op, 0x00u, 0x01u, W[k].size);
        get(0x00u, 4, o);
        CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vlane width") && o[0] == 9 && o[1] == 9,
              "vlane op %d width %d traps (%d)", W[k].op, W[k].size, g_traps);
    }

    /* An op the function does not have says so, rather than blaming the width. */
    g_traps = 0;
    g_trap_what = NULL;
    set4(0x01u, 4, src);
    set4(0x00u, 4, keep);
    psp_vlane(99, 0x00u, 0x01u, 4);
    get(0x00u, 4, o);
    CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vlane op") && o[0] == 9 && o[1] == 9,
          "unknown vlane op traps as such (%d)", g_traps);
    psp_vfpu_set_trap_handler(NULL);
    psp_vfpu_reset();
}

/* vsrt permutes its inputs, whatever they hold: on a tie or a NaN min and max
 * are exact complements, so each input appears once in the pair. (+0, -0)
 * keeps both zeros; (NaN, 1) puts the NaN in the max slot. UNVERIFIED on
 * hardware. vsocp passes a NaN through its saturation: UNVERIFIED too. */
static void test_lane_nan_ties(void) {
    uint32_t a[4];
    psp_vfpu_reset();
    const uint32_t tie[4] = { 0x00000000u, 0x80000000u, 0x7FC00000u, 0x3F800000u };   /* +0 -0 NaN 1 */
    set_bits(0x01u, 4, tie);
    psp_vlane(PSP_VL_SRT1, 0x00u, 0x01u, 4);
    get_bits(0x00u, 4, a);
    CHECK(a[0] == 0x80000000u && a[1] == 0x00000000u && a[2] == 0x3F800000u && a[3] == 0x7FC00000u,
          "vsrt1 on ties and NaN: %08X %08X %08X %08X", a[0], a[1], a[2], a[3]);
    const float nn[4] = { NAN, 1.0f, 1.0f, 1.0f };
    float o[4];
    set4(0x01u, 4, nn);
    psp_vlane(PSP_VL_SOCP, 0x00u, 0x01u, 2); get(0x00u, 4, o);
    CHECK(isnan(o[0]) && isnan(o[1]) && o[2] == 0.0f && o[3] == 1.0f,
          "vsocp.p with a NaN lane: %g %g %g %g", o[0], o[1], o[2], o[3]);
}

static void test_random(void) {
    float o[4];
    uint32_t a[4], b[4];
    psp_vfpu_reset();

    /* The generator is ours (xorshift32), not the PSP's, and what matters is
     * that it is fixed: pin the sequence from a known seed. */
    const uint32_t seed = 0x12345678u;
    set_bits(0x01u, 1, &seed);
    psp_vrnds(0x01u);
    psp_vrnd(PSP_VR_I, 0x00u, 4); get_bits(0x00u, 4, a);
    CHECK(a[0] == 0x87985AA5u && a[1] == 0x155B24A3u && a[2] == 0x4820F4C4u && a[3] == 0x81B3AC98u,
          "vrndi from seed 0x12345678: %08X %08X %08X %08X", a[0], a[1], a[2], a[3]);
    psp_vrnds(0x01u);
    psp_vrnd(PSP_VR_I, 0x00u, 4); get_bits(0x00u, 4, b);
    CHECK(memcmp(a, b, sizeof a) == 0 && a[0] != a[1], "vrnds reseeds");
    psp_vrnd(PSP_VR_I, 0x00u, 4); get_bits(0x00u, 4, b);
    CHECK(memcmp(a, b, sizeof a) != 0, "the sequence advances");

    /* Float forms take the mantissa of the same draw. */
    psp_vrnds(0x01u);
    psp_vrnd(PSP_VR_F1, 0x00u, 4); get_bits(0x00u, 4, a);
    CHECK(a[0] == 0x3FC3CC2Du && a[1] == 0x3F8AAD92u && a[2] == 0x3FA4107Au && a[3] == 0x3FC0D9D6u,
          "vrndf1 from seed 0x12345678: %08X %08X %08X %08X", a[0], a[1], a[2], a[3]);
    psp_vrnds(0x01u);
    psp_vrnd(PSP_VR_F2, 0x00u, 4); get_bits(0x00u, 4, a);
    CHECK(a[0] == 0x4043CC2Du && a[1] == 0x400AAD92u && a[2] == 0x4024107Au && a[3] == 0x4040D9D6u,
          "vrndf2 from seed 0x12345678: %08X %08X %08X %08X", a[0], a[1], a[2], a[3]);

    /* Reset restores the start-up state, whatever the generator did since. */
    psp_vfpu_reset();
    psp_vrnd(PSP_VR_F1, 0x00u, 4); get(0x00u, 4, o);
    float first[4];
    memcpy(first, o, sizeof o);
    for (int i = 0; i < 4; i++) CHECK(o[i] >= 1.0f && o[i] < 2.0f, "vrndf1 in [1,2): %g", o[i]);
    psp_vrnd(PSP_VR_I, 0x00u, 4);
    psp_vfpu_reset();
    psp_vrnd(PSP_VR_F1, 0x00u, 4); get(0x00u, 4, o);
    CHECK(memcmp(o, first, sizeof o) == 0, "same sequence after reset");

    /* Range, over many draws; the mantissa spreads over the whole range. */
    float lo1 = 4, hi1 = 0, lo2 = 4, hi2 = 0;
    for (int i = 0; i < 500; i++) {
        psp_vrnd(PSP_VR_F1, 0x00u, 4); get(0x00u, 4, o);
        for (int j = 0; j < 4; j++) {
            CHECK(o[j] >= 1.0f && o[j] < 2.0f, "vrndf1 in [1,2): %g", o[j]);
            if (o[j] < lo1) lo1 = o[j];
            if (o[j] > hi1) hi1 = o[j];
        }
        psp_vrnd(PSP_VR_F2, 0x00u, 4); get(0x00u, 4, o);
        for (int j = 0; j < 4; j++) {
            CHECK(o[j] >= 2.0f && o[j] < 4.0f, "vrndf2 in [2,4): %g", o[j]);
            if (o[j] < lo2) lo2 = o[j];
            if (o[j] > hi2) hi2 = o[j];
        }
    }
    CHECK(lo1 < 1.05f && hi1 > 1.95f && lo2 < 2.1f && hi2 > 3.9f, "draws cover the range: %g..%g %g..%g", lo1, hi1, lo2, hi2);

    /* A zero seed would lock xorshift at zero: it falls back to the start-up seed. */
    const uint32_t zero = 0;
    set_bits(0x01u, 1, &zero);
    psp_vrnds(0x01u);
    psp_vrnd(PSP_VR_I, 0x00u, 2); get_bits(0x00u, 2, a);
    CHECK(a[0] != 0 && a[1] != 0, "seed 0 does not lock the generator");

    /* Destination addressing and width: one to four lanes at every vd form
     * get the first draws from the start-up state; nothing else changes. */
    static const uint32_t vds[] = { 0x00u, 0x20u, 0x40u, 0x60u, 0x22u };
    char why[96];
    psp_vfpu_reset();
    psp_vrnd(PSP_VR_I, 0x0Cu, 4);
    uint32_t draws[4];
    get_bits(0x0Cu, 4, draws);
    for (size_t k = 0; k < sizeof vds / sizeof vds[0]; k++)
        for (int n = 1; n <= 4; n++) {
            fill77(); snapshot();
            psp_vfpu_reset();
            psp_vrnd(PSP_VR_I, vds[k], n);
            CHECK(only_lanes_changed(vds[k], n, draws, why, sizeof why), "vrndi width %d into 0x%02X: %s", n, vds[k], why);
        }

    /* Prefixes. vrnds reads vs and the others have no source, so vpfxs and
     * vpfxt trap (the game has none pending at its vrnd* sites); vpfxd masks
     * the destination lanes as everywhere else. */
    psp_vfpu_set_trap_handler(count_trap);
    const uint32_t keep[4] = { 77, 77, 77, 77 };
    for (int which = 0; which < 2; which++) {
        g_traps = 0;
        g_trap_what = NULL;
        set_bits(0x00u, 4, keep);
        psp_vfpu_set_prefix(which, 0x1B);
        psp_vrnd(PSP_VR_I, 0x00u, 4);
        get_bits(0x00u, 4, b);
        CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vrnd with vpfxs/vpfxt") &&
              b[0] == 77 && b[3] == 77 && !psp_vfpu_prefix_pending(), "vrndi with prefix %d traps (%d)", which, g_traps);
    }
    {
        g_traps = 0;
        set_bits(0x01u, 1, &seed);
        psp_vrnds(0x01u);
        psp_vrnd(PSP_VR_I, 0x00u, 1); get_bits(0x00u, 1, a);              /* the draw after a plain reseed */
        psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);
        psp_vrnds(0x01u);                                                   /* traps: the seed is not taken */
        psp_vrnd(PSP_VR_I, 0x00u, 1); get_bits(0x00u, 1, b);
        CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, "vrnd with vpfxs/vpfxt") &&
              b[0] != a[0] && !psp_vfpu_prefix_pending(), "vrnds with vpfxs traps and does not reseed (%d)", g_traps);
    }
    g_traps = 0;
    set_bits(0x00u, 4, keep);
    psp_vfpu_set_prefix(PSP_VPFX_D, (1u << 8) | (1u << 10));             /* mask lanes 0 and 2 */
    psp_vrnd(PSP_VR_I, 0x00u, 4); get_bits(0x00u, 4, b);
    CHECK(g_traps == 0 && b[0] == 77 && b[1] != 77 && b[2] == 77 && b[3] != 77, "vrndi under a masking vpfxd");
    set_bits(0x00u, 4, keep);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x01u | (1u << 9));                  /* floats: sat0 lane 0, mask lane 1 */
    psp_vrnd(PSP_VR_F2, 0x00u, 4); get(0x00u, 4, o);
    CHECK(g_traps == 0 && o[0] == 1.0f && bits(o[1]) == 77u && o[2] >= 2.0f && o[3] >= 2.0f,
          "vrndf2 under vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    /* A size or kind the op does not have traps and writes nothing. */
    {
        static const struct { int kind; int size; const char *what; } B[] = {
            { PSP_VR_I, 0, "vrnd width" }, { PSP_VR_F1, 5, "vrnd width" }, { PSP_VR_F2, -1, "vrnd width" },
            { 99, 4, "vrnd kind" },
        };
        for (size_t k = 0; k < sizeof B / sizeof B[0]; k++) {
            g_traps = 0;
            g_trap_what = NULL;
            set_bits(0x00u, 4, keep);
            psp_vfpu_set_prefix(PSP_VPFX_D, 0x1u);
            psp_vrnd(B[k].kind, 0x00u, B[k].size);
            get_bits(0x00u, 4, b);
            CHECK(g_traps == 1 && g_trap_what && !strcmp(g_trap_what, B[k].what) &&
                  b[0] == 77 && b[1] == 77 && b[3] == 77 && !psp_vfpu_prefix_pending(),
                  "vrnd kind %d size %d traps (%d)", B[k].kind, B[k].size, g_traps);
        }
    }
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
    test_pack_destination_lanes();
    test_lane_ops();
    test_lane_destinations();
    test_lane_prefixes();
    test_lane_nan_ties();
    test_random();
    printf("%s\n", failures ? "test_vfpu_ops: FAILED" : "test_vfpu_ops: ok");
    return failures ? 1 : 0;
}
