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
