/* psprecomp — the VFPU.
 *
 * 128 single-precision registers, addressed as 8 matrices of 4x4. A single
 * 7-bit register field can name a scalar, a 2/3/4-element row, a column, or a
 * whole matrix, depending on the instruction's width bits and a transpose bit.
 *
 * ## Register layout
 *
 * The file is indexed `v[matrix*4 + column*32 + row]`. That looks arbitrary
 * and is not: it is the layout that makes a row and a column of the same
 * matrix alias the same storage the way the hardware does. A game that writes
 * a matrix by columns and reads it by rows -- which is exactly what a
 * transpose does -- only works if this matches.
 *
 * ## What is implemented, and what deliberately is not
 *
 * Measured against a real module (WTF's Lumberjack, 1276 VFPU-space words in
 * .text), the distribution is heavily skewed: quad load/store is 48% of it,
 * and vscl/vmul/vadd/vdot another 18%. Those, plus the compare/min/max family,
 * are implemented here and unit-tested.
 *
 * The **prefix** instructions `vpfxs`/`vpfxt`/`vpfxd` do not compute
 * anything -- they set a register that rewrites the *operands of the next
 * instruction*: swizzling lanes, taking abs, negating, forcing constants,
 * saturating and masking writes. An op executed while a prefix is pending
 * computes something different from the same op without one.
 *
 * ## Prefix support
 *
 * "apply" means the operand is read (S, T) or written (D) through the
 * prefix; "trap" means the op reports through the trap handler below, writes
 * nothing, and computes nothing. Either way the op consumes all three
 * prefixes. A trap is used where the op has no such operand, or where the
 * semantics are unverified and the game never needs them: ignoring a prefix
 * would produce numbers that are silently wrong instead of an error that
 * says so. Loud beats plausible.
 *
 *   op                                          vpfxs   vpfxt   vpfxd
 *   vadd vsub vmul vdiv vmin vmax vdot          apply   apply   apply
 *   vsge vslt                                   apply   apply   apply
 *   vscl                                        apply   trap    apply
 *   vcmp (no vector destination)                apply   apply   trap
 *   vunary: vmov vabs vneg vzero vone vrcp      apply   trap    apply
 *     vnrcp vrsq vsqrt vsin vcos vnsin vasin
 *     vexp2 vrexp2 vlog2 vsat0 vsat1
 *   vrot (the angle is vs)                      apply   trap    apply
 *   vcmov (kept lanes rewritten too)            apply   trap    apply
 *   vfad                                        apply   trap    apply
 *   vavg vocp vsocp vbfy1/2 vsrt1-4 vsgn        trap    trap    apply
 *   vcrsp vqmul                                 trap    trap    apply
 *   vidt vcst viim vfim (no source)             trap    trap    apply
 *   vf2in/z/u/d vi2f, vpack (i2uc ... s2i)      trap    trap    apply (1)
 *   vrndi vrndf1 vrndf2 (no source)             trap    trap    apply (1)
 *   vrnds (no vector destination)               trap    trap    trap
 *   vmidt vmzero vmone vmmov vmscl              trap    trap    trap
 *   vtfm vhtfm vmmul
 *   lv/sv, mfv/mtv, vnop vsync vflush           neither read nor consumed
 *
 * (1) An integer result (vf2i*, the packing ops, vrndi) is masked but never
 * saturated; a float result (vi2f, vrndf1/2) takes both.
 *
 * What "apply" does on hardware is our reading of the documentation in a few
 * places (lane wrap for narrow ops, kept vcmov lanes): see the UNVERIFIED
 * notes in vfpu.c and vfpu_ops.c. P2's oracle checks them.
 */
#ifndef PSPRECOMP_VFPU_H
#define PSPRECOMP_VFPU_H

#include "cpu.h"
#include "mem.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve a 7-bit register field and width into up to four indices into
 * psp_cpu.v[]. Returns the number of lanes. */
int psp_vfpu_regs(uint32_t vreg, int size, int out[4]);

/* Load/store. Quad forms are 16-byte aligned on hardware; the address is
 * masked rather than faulting, which is what the hardware does. */
void psp_lv_s(uint32_t vt, uint32_t addr);
void psp_lv_q(uint32_t vt, uint32_t addr);
void psp_sv_s(uint32_t vt, uint32_t addr);
void psp_sv_q(uint32_t vt, uint32_t addr);

/* Element-wise arithmetic across `size` lanes. */
void psp_vadd(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vsub(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vmul(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vdiv(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vmin(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vmax(uint32_t vd, uint32_t vs, uint32_t vt, int size);

/* Reductions. vdot writes one lane; vscl scales a vector by a scalar. */
void psp_vdot(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vscl(uint32_t vd, uint32_t vs, uint32_t vt, int size);

/* Comparison, writing the VFPU condition codes. */
void psp_vcmp(uint32_t cond, uint32_t vs, uint32_t vt, int size);

/* Unary element-wise ops (VFPU4). One entry point rather than eighteen, since
 * they differ only in the scalar function applied per lane. */
enum {
    PSP_VU_MOV = 0, PSP_VU_ABS, PSP_VU_NEG, PSP_VU_ZERO, PSP_VU_ONE,
    PSP_VU_RCP, PSP_VU_RSQ, PSP_VU_SQRT, PSP_VU_SIN, PSP_VU_COS,
    PSP_VU_EXP2, PSP_VU_LOG2, PSP_VU_SAT0, PSP_VU_SAT1,
    PSP_VU_NRCP, PSP_VU_NSIN, PSP_VU_REXP2, PSP_VU_ASIN,
    PSP_VU_F2IZ_UNUSED, PSP_VU_I2F_UNUSED    /* unused: see psp_vf2i / psp_vi2f, which take the scale */
};
void psp_vunary(int op, uint32_t vd, uint32_t vs, int size);

/* vf2in/vf2iz/vf2iu/vf2id (rm 0/1/2/3, see psp_cvt_w) after scaling by
 * 2^scale; vi2f divides by 2^scale. Prefixes: see vfpu_ops.c. */
void psp_vf2i(int rm, uint32_t vd, uint32_t vs, uint32_t scale, int size);
void psp_vi2f(uint32_t vd, uint32_t vs, uint32_t scale, int size);

/* VFPU7 integer narrowing and widening. `size` is the source width. */
enum { PSP_VP_I2UC, PSP_VP_I2C, PSP_VP_I2US, PSP_VP_I2S,
       PSP_VP_UC2I, PSP_VP_C2I, PSP_VP_US2I, PSP_VP_S2I };
void psp_vpack(int op, uint32_t vd, uint32_t vs, int size);

/* VFPU9 lane ops. `size` is the source width: vfad/vavg write one lane,
 * vsocp twice `size` lanes, the rest `size`. Prefixes: see vfpu_ops.c. */
enum { PSP_VL_FAD, PSP_VL_AVG, PSP_VL_OCP, PSP_VL_SOCP, PSP_VL_BFY1, PSP_VL_BFY2,
       PSP_VL_SRT1, PSP_VL_SRT2, PSP_VL_SRT3, PSP_VL_SRT4, PSP_VL_SGN };
void psp_vlane(int op, uint32_t vd, uint32_t vs, int size);

/* Random numbers: a deterministic xorshift32, not the PSP's generator (which is
 * unknown), restarted from a fixed seed by psp_vfpu_reset. vrnds seeds it from
 * the bits of vs; vrndi writes the raw draws, vrndf1 floats in [1, 2), vrndf2
 * floats in [2, 4). */
enum { PSP_VR_I, PSP_VR_F1, PSP_VR_F2 };
void psp_vrnds(uint32_t vs);
void psp_vrnd(int kind, uint32_t vd, int size);

/* Matrix ops that need no multiply: identity, zero, one, and copy. `size` is
 * the matrix order (2, 3 or 4). */
void psp_vmidt(uint32_t vd, int size);
void psp_vidt(uint32_t vd, int size);
void psp_vimm(uint32_t vd, float value);
void psp_vcst(uint32_t vd, uint32_t which, int size);
void psp_vmzero(uint32_t vd, int size);
void psp_vmone(uint32_t vd, int size);
void psp_vmmov(uint32_t vd, uint32_t vs, int size);

/* mfv/mtv: move a single register's bits to or from a GPR. They ignore
 * prefixes and cannot trap. */
uint32_t psp_mfv(uint32_t vd);
void     psp_mtv(uint32_t vd, uint32_t bits);

/* Matrix multiply, transform and scale. See the note in vfpu.c: the operand
 * orientation of vmmul, vtfm and vhtfm is not independently verified. */
void psp_vmscl(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vtfm(uint32_t vd, uint32_t vs, uint32_t vt, int size);
/* The homogeneous transform vhtfm2/3/4: `order` is the matrix order, and the
 * (order - 1)-lane vector gets an implicit last lane of 1.0, so the matrix's
 * last column is added. Same orientation as psp_vtfm. */
void psp_vhtfm(uint32_t vd, uint32_t vs, uint32_t vt, int order);
void psp_vmmul(uint32_t vd, uint32_t vs, uint32_t vt, int size);

/* Prefix state. vpfxs/vpfxt rewrite the next op's vs/vt operand (lane
 * select, abs, constant, negate); vpfxd saturates and masks its writes.
 * Every VFPU op that computes consumes all three, whether it applies them or
 * traps (see the table at the top). */
enum { PSP_VPFX_S = 0, PSP_VPFX_T = 1, PSP_VPFX_D = 2 };
void psp_vfpu_set_prefix(int which, uint32_t value);
int  psp_vfpu_prefix_pending(void);
/* Clears the prefixes and the trap count. The trap handler is left alone, so
 * a host's handler survives a reset. */
void psp_vfpu_reset(void);

/* What happens when the VFPU meets something it cannot execute: a pending
 * prefix the op traps on (see the table above); a width or order the op does
 * not have (vidt.s/.t, vrnd*, the packing and lane ops, vhtfm); an unknown
 * selector or kind (vunary, vlane, vrnd). The op writes nothing and consumes
 * the prefixes. NULL restores the default: print the first few and count
 * them. A host that must not run on with wrong values installs a handler that
 * stops. */
typedef void (*psp_vfpu_trap_fn)(uint32_t addr, const char *what);
void psp_vfpu_set_trap_handler(psp_vfpu_trap_fn fn);

/* vrot: lane code (bits 20..16) and width; see vfpu_ops.c. */
void psp_vrot(uint32_t vd, uint32_t vs, uint32_t code, int size);
/* vcmovt/vcmovf: code = tf (bit 3) | cc (bits 2..0). */
void psp_vcmov(uint32_t vd, uint32_t vs, uint32_t code, int size);
void psp_vcrsp(uint32_t vd, uint32_t vs, uint32_t vt);   /* .t */
void psp_vqmul(uint32_t vd, uint32_t vs, uint32_t vt);   /* .q */
void psp_vsge(uint32_t vd, uint32_t vs, uint32_t vt, int size);
void psp_vslt(uint32_t vd, uint32_t vs, uint32_t vt, int size);

/* Reports an instruction the VFPU cannot yet execute, by address and name:
 * through the trap handler when one is installed, otherwise printed and
 * counted. `addr` is psp_cpu.pc, which the generated code sets to the
 * instruction's address before every VFPU call that computes. */
void psp_vfpu_unimplemented(uint32_t addr, const char *what);
uint64_t psp_vfpu_trap_count(void);

#ifdef __cplusplus
}
#endif

#endif /* PSPRECOMP_VFPU_H */
