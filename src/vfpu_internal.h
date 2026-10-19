/* Internal to the VFPU implementation: vfpu.c (register addressing, prefix
 * state, operand access, the element-wise, unary and matrix ops) and
 * vfpu_ops.c (mfv/mtv, vrot, vcmov, vcrsp/vqmul, vsge/vslt, the conversions
 * and packing, the VFPU9 lane ops and vrnd*). Every op that computes reads
 * its sources through psp_vfpu__read, writes through psp_vfpu__write (or
 * psp_vfpu__write_bits), and ends with psp_vfpu__end.
 *
 * These are linked externally (two files share them), so they carry the
 * library's psp_ prefix and a double underscore marking them internal:
 * nothing outside the VFPU may call them, and they cannot collide with a
 * host's own names. */
#ifndef PSPRECOMP_VFPU_INTERNAL_H
#define PSPRECOMP_VFPU_INTERNAL_H

#include "psprecomp/vfpu.h"

/* Lanes an element-wise op covers for a width field: 1, 2, 3, or 4. */
static inline int vfpu_width(int size) { return size >= 1 && size <= 3 ? size : 4; }
/* 1 if the S, T or D prefix (PSP_VPFX_S/_T/_D) is pending. */
int psp_vfpu__prefix_is_set(int which);

/* `n` lanes of `vreg` as operand `which` (PSP_VPFX_S or PSP_VPFX_T), with
 * that operand's pending prefix applied. */
void psp_vfpu__read(int which, uint32_t vreg, int n, float out[4]);
/* `n` lanes with no prefix, e.g. a destination's old value. */
void psp_vfpu__read_raw(uint32_t vreg, int n, float out[4]);
/* Write `n` lanes through the pending destination prefix. */
void psp_vfpu__write(uint32_t vreg, int n, const float in[4]);
/* The same for integer results: the write mask applies, saturation does not. */
void psp_vfpu__write_bits(uint32_t vreg, int n, const uint32_t in[4]);
/* Consume the prefixes. */
void psp_vfpu__end(void);
/* For ops without prefix support: 1 to go ahead, or 0 after trapping (and
 * consuming the prefixes) if one is pending. */
int psp_vfpu__no_prefix(const char *what);
/* Sine and cosine of an angle in quarter turns; exact at whole quarter turns. */
void psp_vfpu__sincos_q(float a, float *s, float *c);
/* Restart the vrnd* generator from its fixed seed (psp_vfpu_reset calls it). */
void psp_vfpu__rnd_reset(void);
/* Report through the trap handler. */
void psp_vfpu__trap(const char *what);
/* For an op that has no `which` operand (PSP_VPFX_S, _T or _D): 1 to go
 * ahead, or 0 after trapping with `what` (and consuming the prefixes) if a
 * prefix for it is pending. A prefix for an operand the op does not have is
 * not ours to interpret, so it is never silently consumed. */
int psp_vfpu__no_operand(int which, const char *what);

#endif
