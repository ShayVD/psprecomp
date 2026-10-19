/* Internal to the VFPU implementation (vfpu.c, and the op files later tasks
 * add): operand access with prefixes applied. Every op that computes reads
 * its sources through vfpu_read, writes through vfpu_write (or
 * vfpu_write_bits), and ends with vfpu_end. */
#ifndef PSPRECOMP_VFPU_INTERNAL_H
#define PSPRECOMP_VFPU_INTERNAL_H

#include "psprecomp/vfpu.h"

/* Lanes an element-wise op covers for a width field: 1, 2, 3, or 4. */
static inline int vfpu_width(int size) { return size >= 1 && size <= 3 ? size : 4; }
/* 1 if the S (PSP_VPFX_S) or T (PSP_VPFX_T) prefix is pending. */
int vfpu_prefix_is_set(int which);

/* `n` lanes of `vreg` as operand `which` (PSP_VPFX_S or PSP_VPFX_T), with
 * that operand's pending prefix applied. */
void vfpu_read(int which, uint32_t vreg, int n, float out[4]);
/* `n` lanes with no prefix, e.g. a destination's old value. */
void vfpu_read_raw(uint32_t vreg, int n, float out[4]);
/* Write `n` lanes through the pending destination prefix. */
void vfpu_write(uint32_t vreg, int n, const float in[4]);
/* The same for integer results: the write mask applies, saturation does not. */
void vfpu_write_bits(uint32_t vreg, int n, const uint32_t in[4]);
/* Consume the prefixes. */
void vfpu_end(void);
/* For ops without prefix support: 1 to go ahead, or 0 after trapping (and
 * consuming the prefixes) if one is pending. */
int vfpu_no_prefix(const char *what);
/* Sine and cosine of an angle in quarter turns; exact at whole quarter turns. */
void vfpu_sincos_q(float a, float *s, float *c);
/* Restart the vrnd* generator from its fixed seed (psp_vfpu_reset calls it). */
void vfpu_rnd_reset(void);
/* Report through the trap handler. */
void vfpu_trap(const char *what);

#endif
