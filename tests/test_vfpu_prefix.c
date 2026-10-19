/* VFPU operand prefixes and vcmp conditions. Synthetic values only. */
#include "psprecomp/vfpu.h"
#include "psprecomp/recomp_rt.h"
#include "../src/vfpu_internal.h"      /* vfpu_read, for the identity sweep */

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                           printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

static void set4(uint32_t vreg, float a, float b, float c, float d) {
    int r[4];
    psp_vfpu_regs(vreg, 4, r);
    psp_cpu.v[r[0]] = a; psp_cpu.v[r[1]] = b; psp_cpu.v[r[2]] = c; psp_cpu.v[r[3]] = d;
}
static void get4(uint32_t vreg, float out[4]) {
    int r[4];
    psp_vfpu_regs(vreg, 4, r);
    for (int i = 0; i < 4; i++) out[i] = psp_cpu.v[r[i]];
}
#define EQ4(o, a, b, c, d) ((o)[0] == (a) && (o)[1] == (b) && (o)[2] == (c) && (o)[3] == (d))

static int traps;
static uint32_t trap_addr;
static void on_trap(uint32_t addr, const char *what) { (void)what; trap_addr = addr; traps++; }

static uint32_t bits_at(int reg) { return psp_f32_to_bits(psp_cpu.v[reg]); }

/* vidt on `vd` at width `n` writes 1 in lane `one`, 0 elsewhere. */
static int vidt_is(uint32_t vd, int n, int one) {
    int r[4];
    psp_vfpu_regs(vd, n, r);
    for (int i = 0; i < n; i++) psp_cpu.v[r[i]] = 7.0f;
    psp_vidt(vd, n);
    for (int i = 0; i < n; i++)
        if (psp_cpu.v[r[i]] != (i == one ? 1.0f : 0.0f)) return 0;
    return 1;
}

#define C000 0x00u
#define C010 0x01u
#define C020 0x02u

int main(void) {
    float o[4];
    psp_vfpu_reset();

    /* Identity prefix: same as none. */
    set4(C000, 1, 2, 3, 4); set4(C010, 10, 20, 30, 40);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0xE4);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0xE4);
    psp_vadd(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(EQ4(o, 11, 22, 33, 44), "identity prefixes: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* Swizzle [W, Z, Y, X] on the source of vmov.q. */
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(EQ4(o, 4, 3, 2, 1), "swizzle wzyx: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* The prefix was consumed: the next op is unprefixed. */
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(EQ4(o, 1, 2, 3, 4), "prefix consumed: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* [-X, |Y|, 1, 1/2] (pspas: 0xDC01C2D4). */
    set4(C000, 2, -3, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x01C2D4);
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(EQ4(o, -2, 3, 1, 0.5f), "neg/abs/constants: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* The other constant table: abs bit set, constant bit set, picks 0..3
     * (pspas: vpfxs 3, 1/3, 1/4, 1/6 = 0xDC00FFE4). */
    psp_vfpu_set_prefix(PSP_VPFX_S, 0xF000u | 0x0F00u | 0xE4u);
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(o[0] == 3.0f && o[1] == 1.0f / 3.0f && o[2] == 0.25f && o[3] == 1.0f / 6.0f,
          "second constant table: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* A negated constant: -1/2 in lane 0 (constant, pick 3, negate). */
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x10000u | 0x1000u | 0xE7u);
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(o[0] == -0.5f, "negated constant: %g", o[0]);

    /* Neg and abs work on the bits: |NaN| keeps the payload, -(+0) is -0. */
    {
        int r[4];
        psp_vfpu_regs(C000, 4, r);
        psp_cpu.v[r[0]] = psp_bits_to_f32(0xFFC01234u);   /* negative NaN, payload */
        psp_cpu.v[r[1]] = 0.0f;
        psp_vfpu_set_prefix(PSP_VPFX_S, 0x20000u | 0x100u | 0xE4u);  /* |X|, -Y */
        psp_vunary(PSP_VU_MOV, C020, C000, 4);
        psp_vfpu_regs(C020, 4, r);
        CHECK(psp_f32_to_bits(psp_cpu.v[r[0]]) == 0x7FC01234u, "|NaN| bits: 0x%08X",
              psp_f32_to_bits(psp_cpu.v[r[0]]));
        CHECK(psp_f32_to_bits(psp_cpu.v[r[1]]) == 0x80000000u, "-(+0) bits: 0x%08X",
              psp_f32_to_bits(psp_cpu.v[r[1]]));
    }

    /* Destination: [0:1, -1:1, M, -] (pspas: 0xDE00040D). */
    set4(C000, 1.5f, -2, 7, -0.25f);
    set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x00040D);
    psp_vunary(PSP_VU_MOV, C020, C000, 4);
    get4(C020, o);
    CHECK(EQ4(o, 1, -1, 9, -0.25f), "saturate and mask: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* T prefix applies to vt, not vs (pspas: vpfxt W, Z, Y, X = 0xDD00001B). */
    set4(C000, 1, 1, 1, 1); set4(C010, 1, 2, 3, 4);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x1B);
    psp_vsub(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(EQ4(o, -3, -2, -1, 0), "t prefix: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* Aliasing: a swizzled source that is also the destination is read in
     * full before any lane is written. */
    set4(C000, 1, 2, 3, 4);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);
    psp_vunary(PSP_VU_MOV, C000, C000, 4);
    get4(C000, o);
    CHECK(EQ4(o, 4, 3, 2, 1), "aliased swizzle: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* A narrower op may pick any lane of the row: vmov.p from [W, Z]. */
    set4(C000, 1, 2, 3, 4); set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x0Bu);         /* lane0 = W (3), lane1 = Z (2) */
    psp_vunary(PSP_VU_MOV, C020, C000, 2);
    get4(C020, o);
    CHECK(EQ4(o, 4, 3, 9, 9), "pair picks beyond its width: %g %g %g %g", o[0], o[1], o[2], o[3]);

    /* vdot and vscl consume their prefixes too. */
    set4(C000, 1, 2, 3, 4); set4(C010, 1, 1, 1, 1);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x0F000u | 0x55u);   /* every lane the constant 1 */
    psp_vdot(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(o[0] == 4.0f && !psp_vfpu_prefix_pending(), "vdot with prefix: %g", o[0]);

    /* An integer result is masked but never saturated. */
    {
        int r[4];
        psp_vfpu_regs(C000, 4, r);
        set4(C000, 5.0f, -7.0f, 2.0f, 3.0f);
        set4(C020, 9, 9, 9, 9);
        psp_vfpu_set_prefix(PSP_VPFX_D, 0x0800u | 0x0055u);  /* sat 0:1 on 0..3, mask W */
        psp_vf2i(1, C020, C000, 0, 4);
        psp_vfpu_regs(C020, 4, r);
        CHECK(psp_f32_to_bits(psp_cpu.v[r[0]]) == 5u &&
              psp_f32_to_bits(psp_cpu.v[r[1]]) == (uint32_t)-7 &&
              psp_f32_to_bits(psp_cpu.v[r[2]]) == 2u && psp_cpu.v[r[3]] == 9.0f,
              "vf2iz through vpfxd: 0x%08X 0x%08X 0x%08X %g",
              psp_f32_to_bits(psp_cpu.v[r[0]]), psp_f32_to_bits(psp_cpu.v[r[1]]),
              psp_f32_to_bits(psp_cpu.v[r[2]]), psp_cpu.v[r[3]]);
    }

    /* vcmp conditions 8..15 test vs alone. */
    set4(C000, 0.0f, NAN, INFINITY, 5.0f);
    psp_vcmp(9, C000, C000, 4);                    /* EN: is NaN */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0x2 && (psp_cpu.vfpu_cc & 0x10) && !(psp_cpu.vfpu_cc & 0x20),
          "vcmp EN: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(8, C000, C000, 4);                    /* EZ */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0x1, "vcmp EZ: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(10, C000, C000, 4);                   /* EI */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0x4, "vcmp EI: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(11, C000, C000, 4);                   /* ES: NaN or inf */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0x6, "vcmp ES: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(12, C000, C000, 4);                   /* NZ */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0xE, "vcmp NZ: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(13, C000, C000, 4);                   /* NN */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0xD, "vcmp NN: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(14, C000, C000, 4);                   /* NI */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0xB, "vcmp NI: cc=0x%X", psp_cpu.vfpu_cc);
    psp_vcmp(15, C000, C000, 4);                   /* NS */
    CHECK((psp_cpu.vfpu_cc & 0xF) == 0x9, "vcmp NS: cc=0x%X", psp_cpu.vfpu_cc);

    /* A narrower vcmp updates only its own lanes and bits 4/5; the lane bits
     * past its width keep their old value, and any/all cover its n lanes. */
    psp_cpu.vfpu_cc = 0x0A;
    psp_vcmp(13, C000, C000, 1);                   /* NN on lane 0 alone (0.0) */
    CHECK(psp_cpu.vfpu_cc == 0x3B, "vcmp.s NN: cc=0x%X", psp_cpu.vfpu_cc);
    psp_cpu.vfpu_cc = 0x3F;
    psp_vcmp(0, C000, C000, 2);                    /* FL on lanes 0, 1 */
    CHECK(psp_cpu.vfpu_cc == 0x0C, "vcmp.p FL: cc=0x%X", psp_cpu.vfpu_cc);

    /* vcmp reads vs through the S prefix: EZ on a forced 0 in every lane. */
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x0F000u);
    psp_vcmp(8, C000, C000, 4);
    CHECK(psp_cpu.vfpu_cc == 0x3F && !psp_vfpu_prefix_pending(), "vcmp with prefix: cc=0x%X",
          psp_cpu.vfpu_cc);

    /* A T-only prefix is consumed by vcmp, even for a condition that reads
     * vs alone. */
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x0F000u);
    psp_vcmp(8, C000, C000, 4);
    CHECK(!psp_vfpu_prefix_pending(), "vcmp EZ consumes a T-only prefix");

    /* An op without prefix support traps through the handler when one is
     * pending, with the address the generated code stored in psp_cpu.pc. */
    psp_vfpu_set_trap_handler(on_trap);
    psp_vfpu_set_prefix(PSP_VPFX_S, 0x1B);
    psp_cpu.pc = 0x08812340u;
    psp_vmmov(0x00u, 0x04u, 4);
    CHECK(traps == 1, "matrix op with a pending prefix traps (%d)", traps);
    CHECK(trap_addr == 0x08812340u, "trap address 0x%08X", trap_addr);
    CHECK(!psp_vfpu_prefix_pending(), "the trap consumed the prefix");

    /* The handler survives a reset. */
    psp_vfpu_reset();
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x100);
    psp_vmidt(0x00u, 4);
    CHECK(traps == 2, "the handler survives psp_vfpu_reset (%d)", traps);

    /* vidt: a quad puts its 1 in the lane matching the register's column
     * (C0c0, vd = c) or row (R0r0, vd = 0x20 | r) index, vd & 3, so
     * vidt C000..C030 builds an identity matrix by columns and R000..R030 by
     * rows. A pair uses vd & 1. */
    for (uint32_t c = 0; c < 4; c++) {
        CHECK(vidt_is(c, 4, (int)c), "vidt.q C0%u0", c);
        CHECK(vidt_is(0x20u | c, 4, (int)c), "vidt.q R0%u0", c);
        CHECK(vidt_is(0x0Cu | c, 4, (int)c), "vidt.q C3%u0", c);
    }
    CHECK(vidt_is(0x00u, 2, 0) && vidt_is(0x01u, 2, 1), "vidt.p C000 / C010");
    CHECK(vidt_is(0x40u, 2, 0) && vidt_is(0x41u, 2, 1), "vidt.p C002 / C012");
    CHECK(vidt_is(0x20u, 2, 0) && vidt_is(0x21u, 2, 1), "vidt.p R000 / R010");
    /* Single and triple vidt are not verified: they trap. */
    traps = 0;
    psp_vidt(0x00u, 1);
    psp_vidt(0x00u, 3);
    CHECK(traps == 2, "vidt.s / vidt.t trap (%d)", traps);

    /* vscl with a pending T prefix traps (per-lane semantics unverified), and
     * the prefixes are consumed. */
    traps = 0;
    set4(C000, 1, 2, 3, 4); set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x20000u | 0xE4u);
    psp_vscl(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(traps == 1 && EQ4(o, 9, 9, 9, 9) && !psp_vfpu_prefix_pending(),
          "vscl with a T prefix traps (%d): %g %g %g %g", traps, o[0], o[1], o[2], o[3]);
    psp_vfpu_set_trap_handler(NULL);

    /* vscl through a D prefix: mask lane 1, saturate lane 2 to [0, 1]. */
    {
        int r[4];
        psp_vfpu_regs(0x41u, 1, r);                 /* S012: a scalar elsewhere */
        psp_cpu.v[r[0]] = 2.0f;
        set4(C000, 1, 2, 3, 4); set4(C020, 9, 9, 9, 9);
        psp_vfpu_set_prefix(PSP_VPFX_D, 0x0200u | 0x0010u);
        psp_vscl(C020, C000, 0x41u, 4);
        get4(C020, o);
        CHECK(EQ4(o, 2, 9, 1, 8), "vscl through vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    }

    /* vdot with T and D prefixes: t forced to [1, 1, 1, -1], result saturated
     * to [-1, 1]. 1 + 2 + 3 - 4 = 2 -> 1. */
    set4(C000, 1, 2, 3, 4); set4(C010, 5, 5, 5, 5); set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x80000u | 0x0F000u | 0x55u);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x3u);
    psp_vdot(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(EQ4(o, 1, 9, 9, 9), "vdot with T and D: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_T, 0x80000u | 0x0F000u | 0x55u);
    psp_vdot(C020, C000, C010, 4);
    get4(C020, o);
    CHECK(o[0] == 2.0f, "vdot with T alone: %g", o[0]);

    /* vidt, vcst and viim/vfim write through the D prefix. */
    set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x0100u);       /* mask lane 0 */
    psp_vidt(C020, 4);                              /* the 1 goes in lane 2 */
    get4(C020, o);
    CHECK(EQ4(o, 9, 0, 1, 0), "vidt through vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    set4(C020, 9, 9, 9, 9);
    psp_vfpu_set_prefix(PSP_VPFX_D, 0x0100u | 0x00F0u);  /* mask X, saturate Z, W */
    psp_vcst(C020, 9, 4);                           /* pi */
    get4(C020, o);
    CHECK(o[0] == 9.0f && o[1] > 3.14f && o[1] < 3.15f && o[2] == 1.0f && o[3] == 1.0f,
          "vcst through vpfxd: %g %g %g %g", o[0], o[1], o[2], o[3]);
    {
        int r[4];
        psp_vfpu_regs(0x00u, 1, r);
        psp_cpu.v[r[0]] = 9.0f;
        psp_vfpu_set_prefix(PSP_VPFX_D, 0x100u);
        psp_vimm(0x00u, 5.0f);
        CHECK(psp_cpu.v[r[0]] == 9.0f && !psp_vfpu_prefix_pending(), "viim, masked: %g",
              psp_cpu.v[r[0]]);
        psp_vfpu_set_prefix(PSP_VPFX_D, 0x3u);
        psp_vimm(0x00u, -5.0f);
        CHECK(psp_cpu.v[r[0]] == -1.0f, "viim, saturated: %g", psp_cpu.v[r[0]]);
    }

    /* The identity prefix reads exactly what an unprefixed read does, for
     * every register field and width. */
    {
        for (int i = 0; i < 128; i++) psp_cpu.v[i] = (float)(i + 1);
        int bad = 0;
        for (int n = 1; n <= 4; n++)
            for (uint32_t v = 0; v < 128; v++) {
                float a[4], b[4];
                vfpu_read_raw(v, n, a);
                psp_vfpu_set_prefix(PSP_VPFX_S, 0xE4);
                vfpu_read(PSP_VPFX_S, v, n, b);
                vfpu_end();
                if (memcmp(a, b, sizeof(float) * (size_t)n)) bad++;
            }
        CHECK(bad == 0, "identity prefix differs from a raw read in %d operands", bad);
    }

    /* A narrow op picks lanes past its width within its row or column, from
     * an offset start, transposed or not (the wrap is unverified; this pins
     * our reading). The pair at 0x60 is the transposed rows 2, 3 of column 0:
     * registers 64, 96, then (wrapping) 0, 32. The single at 0x60 is row 3 of
     * column 0, register 3; its lane pick 1 wraps to row 0, register 0. */
    {
        for (int i = 0; i < 128; i++) psp_cpu.v[i] = (float)i;
        float b[4];
        psp_vfpu_set_prefix(PSP_VPFX_S, 0x0Eu);     /* lane0 <- 2, lane1 <- 3 */
        vfpu_read(PSP_VPFX_S, 0x60u, 2, b);
        vfpu_end();
        CHECK(b[0] == 0.0f && b[1] == 32.0f, "transposed pair wrap: %g %g", b[0], b[1]);
        psp_vfpu_set_prefix(PSP_VPFX_S, 0x01u);     /* lane0 <- 1 */
        vfpu_read(PSP_VPFX_S, 0x60u, 1, b);
        vfpu_end();
        CHECK(b[0] == 0.0f, "single row-3 wrap: %g", b[0]);
        psp_vfpu_set_prefix(PSP_VPFX_S, 0x00u);     /* lane0 <- 0 */
        vfpu_read(PSP_VPFX_S, 0x60u, 1, b);
        vfpu_end();
        CHECK(b[0] == 3.0f, "single row-3 pick 0: %g", b[0]);
    }

    /* No fused multiply-add: vdot.p of (1, 1+2^-12) and (-(1+2^-11), 1+2^-12).
     * Rounded separately, the second product is 1+2^-11 and the sum is 0;
     * fused, the sum is 2^-24. */
    {
        int r[4];
        const float a = 1.0f + 0x1p-12f, c = -(1.0f + 0x1p-11f);
        psp_vfpu_regs(0x00u, 2, r); psp_cpu.v[r[0]] = 1.0f; psp_cpu.v[r[1]] = a;
        psp_vfpu_regs(0x01u, 2, r); psp_cpu.v[r[0]] = c;    psp_cpu.v[r[1]] = a;
        psp_vdot(0x02u, 0x00u, 0x01u, 2);
        psp_vfpu_regs(0x02u, 1, r);
        CHECK(bits_at(r[0]) == 0u, "vdot.p rounds each product: %a", (double)psp_cpu.v[r[0]]);
    }

    printf("%s\n", failures ? "test_vfpu_prefix: FAILED" : "test_vfpu_prefix: ok");
    return failures ? 1 : 0;
}
