/* psprecomp: the VFPU ops added for KH BBS (P3). See vfpu.h. */
#include "psprecomp/vfpu.h"
#include "psprecomp/recomp_rt.h"
#include "vfpu_internal.h"

#include <math.h>
#include <string.h>

uint32_t psp_mfv(uint32_t vd) {
    int r[4];
    psp_vfpu_regs(vd, 1, r);
    return psp_f32_to_bits(psp_cpu.v[r[0]]);
}

void psp_mtv(uint32_t vd, uint32_t bits) {
    int r[4];
    psp_vfpu_regs(vd, 1, r);
    psp_cpu.v[r[0]] = psp_bits_to_f32(bits);
}

/* vrot: cosine into one lane, sine into another (or into every other lane
 * when both codes name the same lane), zeros elsewhere. code bit 4 negates
 * the sine. */
void psp_vrot(uint32_t vd, uint32_t vs, uint32_t code, int size) {
    float a[4], d[4], s, c;
    vfpu_read(PSP_VPFX_S, vs, 1, a);
    vfpu_sincos_q(a[0], &s, &c);
    if (code & 0x10) s = -s;
    const int sl = (int)(code >> 2) & 3, cl = (int)code & 3;
    const int n = vfpu_width(size);
    for (int i = 0; i < n; i++) d[i] = (sl == cl) ? s : 0.0f;
    if (sl != cl && sl < n) d[sl] = s;
    if (cl < n) d[cl] = c;
    vfpu_write(vd, n, d);
    vfpu_end();
}

/* vcmovt (tf = 0) moves when the condition bit is 1, vcmovf when it is 0.
 * cc 0-5 is one vfpu_cc bit for all lanes, 6 is each lane's own bit, 7 never
 * moves. A lane that does not move keeps its old value, but every lane is
 * written back through the destination prefix, so a pending vpfxd saturates
 * the kept values too and a masked lane is skipped. UNVERIFIED on hardware:
 * the game has one vcmov with a prefix pending, and it is a vpfxs. There is
 * no vt operand, so a pending vpfxt is not ours to interpret: trap. */
void psp_vcmov(uint32_t vd, uint32_t vs, uint32_t code, int size) {
    if (vfpu_prefix_is_set(PSP_VPFX_T)) { vfpu_end(); vfpu_trap("vcmov with vpfxt"); return; }
    const uint32_t tf = (code >> 3) & 1, cc = code & 7;
    const int n = vfpu_width(size);
    float s[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);
    vfpu_read_raw(vd, n, d);
    for (int i = 0; i < n && cc != 7; i++) {
        const uint32_t bit = (psp_cpu.vfpu_cc >> (cc < 6 ? cc : (uint32_t)i)) & 1;
        if (bit ^ tf) d[i] = s[i];
    }
    vfpu_write(vd, n, d);
    vfpu_end();
}

/* The semantics of vpfxs/vpfxt on vcrsp and vqmul are unverified, and none of
 * the game's 30 sites has one pending (counted in the generated code): trap
 * rather than guess. The destination prefix applies as for any op. */
static int crsp_qmul_prefix_trap(const char *what) {
    if (!vfpu_prefix_is_set(PSP_VPFX_S) && !vfpu_prefix_is_set(PSP_VPFX_T)) return 0;
    vfpu_end();
    vfpu_trap(what);
    return 1;
}

void psp_vcrsp(uint32_t vd, uint32_t vs, uint32_t vt) {
    if (crsp_qmul_prefix_trap("vcrsp with vpfxs/vpfxt")) return;
    float s[4], t[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, 3, s);
    vfpu_read(PSP_VPFX_T, vt, 3, t);
    d[0] = s[1] * t[2] - s[2] * t[1];
    d[1] = s[2] * t[0] - s[0] * t[2];
    d[2] = s[0] * t[1] - s[1] * t[0];
    vfpu_write(vd, 3, d);
    vfpu_end();
}

/* Quaternions as (x, y, z, w), w the real part. */
void psp_vqmul(uint32_t vd, uint32_t vs, uint32_t vt) {
    if (crsp_qmul_prefix_trap("vqmul with vpfxs/vpfxt")) return;
    float s[4], t[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, 4, s);
    vfpu_read(PSP_VPFX_T, vt, 4, t);
    d[0] =  s[0] * t[3] + s[1] * t[2] - s[2] * t[1] + s[3] * t[0];
    d[1] = -s[0] * t[2] + s[1] * t[3] + s[2] * t[0] + s[3] * t[1];
    d[2] =  s[0] * t[1] - s[1] * t[0] + s[2] * t[3] + s[3] * t[2];
    d[3] = -s[0] * t[0] - s[1] * t[1] - s[2] * t[2] + s[3] * t[3];
    vfpu_write(vd, 4, d);
    vfpu_end();
}

static void compare_set(uint32_t vd, uint32_t vs, uint32_t vt, int size, int ge) {
    const int n = vfpu_width(size);
    float s[4], t[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);
    vfpu_read(PSP_VPFX_T, vt, n, t);
    for (int i = 0; i < n; i++)
        d[i] = (ge ? s[i] >= t[i] : s[i] < t[i]) ? 1.0f : 0.0f;   /* NaN: false, 0 */
    vfpu_write(vd, n, d);
    vfpu_end();
}
void psp_vsge(uint32_t vd, uint32_t vs, uint32_t vt, int size) { compare_set(vd, vs, vt, size, 1); }
void psp_vslt(uint32_t vd, uint32_t vs, uint32_t vt, int size) { compare_set(vd, vs, vt, size, 0); }

/* ---- conversions and packing --------------------------------------------- */

/* Prefix policy. The game has no site of the vf2i forms, vi2f or the packing
 * ops with a pending vpfxs, vpfxt or vpfxd (counted in the generated code), so
 * what we cannot verify traps rather than guesses:
 *  - vpfxs traps on all of them. For vi2f and the packing ops the source lanes
 *    are integer bits, so abs/neg/constants on them are doubtful; for vf2i* it
 *    is plausible element-wise but equally unverified.
 *  - none has a vt operand, so a pending vpfxt is not ours to interpret: trap.
 *  - vpfxd is the one destination mechanism every op shares, so it is applied
 *    as everywhere else: it masks destination lanes. An integer result goes
 *    through vfpu_write_bits (mask, no saturation); vi2f's float result goes
 *    through vfpu_write, so a saturation mode applies. Whether hardware does
 *    the same for conversions is unverified, as for the other ops.
 * P2 note: denormal inputs are not flushed here (vf2iu of a denormal float is
 * 1). A VFPU that flushes denormals to zero, or a Vita host with FPSCR.FZ set,
 * would differ. */
static int conv_prefix_trap(const char *what) {
    if (!vfpu_prefix_is_set(PSP_VPFX_S) && !vfpu_prefix_is_set(PSP_VPFX_T)) return 0;
    vfpu_end();
    vfpu_trap(what);
    return 1;
}

void psp_vf2i(int rm, uint32_t vd, uint32_t vs, uint32_t scale, int size) {
    if (conv_prefix_trap("vf2i with vpfxs/vpfxt")) return;
    const int n = vfpu_width(size);
    float s[4];
    uint32_t d[4];
    const double k = ldexp(1.0, (int)scale);       /* 2^scale; the products below are exact in double */
    vfpu_read(PSP_VPFX_S, vs, n, s);
    for (int i = 0; i < n; i++)
        d[i] = (uint32_t)psp_cvt_w((double)s[i] * k, rm);
    vfpu_write_bits(vd, n, d);
    vfpu_end();
}

void psp_vi2f(uint32_t vd, uint32_t vs, uint32_t scale, int size) {
    if (conv_prefix_trap("vi2f with vpfxs/vpfxt")) return;
    const int n = vfpu_width(size);
    float s[4], d[4];
    const double k = ldexp(1.0, -(int)scale);      /* 2^-scale, exact */
    vfpu_read_raw(vs, n, s);
    for (int i = 0; i < n; i++)
        d[i] = (float)((double)(int32_t)psp_f32_to_bits(s[i]) * k);
    vfpu_write(vd, n, d);
    vfpu_end();
}

void psp_vpack(int op, uint32_t vd, uint32_t vs, int size) {
    if (conv_prefix_trap("vpack with vpfxs/vpfxt")) return;
    float sf[4] = { 0, 0, 0, 0 };
    uint32_t s[4], d[4] = { 0, 0, 0, 0 };
    int dn;
    vfpu_read_raw(vs, size >= 1 && size <= 4 ? size : 4, sf);
    for (int i = 0; i < 4; i++) s[i] = psp_f32_to_bits(sf[i]);
    switch (op) {
    case PSP_VP_I2UC:
        if (size != 4) goto bad;
        dn = 1;
        for (int i = 0; i < 4; i++)
            d[0] |= ((int32_t)s[i] < 0 ? 0u : (s[i] >> 23) & 0xFFu) << (8 * i);
        break;
    case PSP_VP_I2C:
        if (size != 4) goto bad;
        dn = 1;
        for (int i = 0; i < 4; i++) d[0] |= ((s[i] >> 24) & 0xFFu) << (8 * i);
        break;
    case PSP_VP_I2US:
        if (size != 2 && size != 4) goto bad;
        dn = size / 2;
        for (int i = 0; i < size; i++)
            d[i / 2] |= ((int32_t)s[i] < 0 ? 0u : (s[i] >> 15) & 0xFFFFu) << (16 * (i & 1));
        break;
    case PSP_VP_I2S:
        if (size != 2 && size != 4) goto bad;
        dn = size / 2;
        for (int i = 0; i < size; i++) d[i / 2] |= ((s[i] >> 16) & 0xFFFFu) << (16 * (i & 1));
        break;
    case PSP_VP_UC2I:
        if (size != 1) goto bad;
        dn = 4;
        for (int i = 0; i < 4; i++) d[i] = (((s[0] >> (8 * i)) & 0xFFu) * 0x01010101u) >> 1;
        break;
    case PSP_VP_C2I:
        if (size != 1) goto bad;
        dn = 4;
        for (int i = 0; i < 4; i++) d[i] = ((s[0] >> (8 * i)) & 0xFFu) << 24;
        break;
    case PSP_VP_US2I:
        if (size != 1 && size != 2) goto bad;
        dn = size * 2;
        for (int i = 0; i < dn; i++) d[i] = ((s[i / 2] >> (16 * (i & 1))) & 0xFFFFu) << 15;
        break;
    case PSP_VP_S2I:
        if (size != 1 && size != 2) goto bad;
        dn = size * 2;
        for (int i = 0; i < dn; i++) d[i] = ((s[i / 2] >> (16 * (i & 1))) & 0xFFFFu) << 16;
        break;
    default:
        goto bad;
    }
    vfpu_write_bits(vd, dn, d);
    vfpu_end();
    return;
bad:
    vfpu_end();
    vfpu_trap("vpack width");
}
