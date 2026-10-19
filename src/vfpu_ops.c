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
    for (int i = 0; i < size; i++) d[i] = (sl == cl) ? s : 0.0f;
    if (sl != cl && sl < size) d[sl] = s;
    if (cl < size) d[cl] = c;
    vfpu_write(vd, size, d);
    vfpu_end();
}

/* vcmovt (tf = 0) moves when the condition bit is 1, vcmovf when it is 0.
 * cc 0-5 is one vfpu_cc bit for all lanes, 6 is each lane's own bit, 7 never
 * moves. Lanes that do not move keep their value. */
void psp_vcmov(uint32_t vd, uint32_t vs, uint32_t code, int size) {
    const uint32_t tf = (code >> 3) & 1, cc = code & 7;
    float s[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, size, s);
    vfpu_read_raw(vd, size, d);
    for (int i = 0; i < size && cc != 7; i++) {
        const uint32_t bit = (psp_cpu.vfpu_cc >> (cc < 6 ? cc : (uint32_t)i)) & 1;
        if (bit ^ tf) d[i] = s[i];
    }
    vfpu_write(vd, size, d);
    vfpu_end();
}

void psp_vcrsp(uint32_t vd, uint32_t vs, uint32_t vt) {
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
    float s[4], t[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, size, s);
    vfpu_read(PSP_VPFX_T, vt, size, t);
    for (int i = 0; i < size; i++)
        d[i] = (ge ? s[i] >= t[i] : s[i] < t[i]) ? 1.0f : 0.0f;   /* NaN: false, 0 */
    vfpu_write(vd, size, d);
    vfpu_end();
}
void psp_vsge(uint32_t vd, uint32_t vs, uint32_t vt, int size) { compare_set(vd, vs, vt, size, 1); }
void psp_vslt(uint32_t vd, uint32_t vs, uint32_t vt, int size) { compare_set(vd, vs, vt, size, 0); }
