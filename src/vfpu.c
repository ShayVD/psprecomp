/* psprecomp — the VFPU. See include/psprecomp/vfpu.h. */

#include "psprecomp/vfpu.h"
#include "psprecomp/recomp_rt.h"
#include "vfpu_internal.h"

#include <math.h>

#include <stdio.h>
#include <string.h>

static uint32_t g_prefix[3];      /* vpfxs, vpfxt, vpfxd */
static int      g_prefix_set[3];
static uint64_t g_traps;
static psp_vfpu_trap_fn g_trap_fn;

/* Leaves the trap handler alone: a host installs it once at start-up and it
 * must survive the resets a run does. */
void psp_vfpu_reset(void) {
    memset(g_prefix, 0, sizeof g_prefix);
    memset(g_prefix_set, 0, sizeof g_prefix_set);
    g_traps = 0;
}

void psp_vfpu_set_prefix(int which, uint32_t value) {
    if (which < 0 || which > 2) return;
    g_prefix[which] = value;
    g_prefix_set[which] = 1;
}

int psp_vfpu_prefix_pending(void) {
    return g_prefix_set[0] || g_prefix_set[1] || g_prefix_set[2];
}

uint64_t psp_vfpu_trap_count(void) { return g_traps; }

void psp_vfpu_set_trap_handler(psp_vfpu_trap_fn fn) { g_trap_fn = fn; }

void psp_vfpu_unimplemented(uint32_t addr, const char *what) {
    if (g_trap_fn) { g_trap_fn(addr, what); return; }
    /* First few only: a VFPU-heavy inner loop would otherwise produce
     * megabytes of identical lines and hide everything else. */
    if (g_traps < 16)
        fprintf(stderr, "psprecomp: VFPU %s at 0x%08X not implemented\n", what, addr);
    else if (g_traps == 16)
        fprintf(stderr, "psprecomp: (further VFPU traps suppressed)\n");
    g_traps++;
}

void vfpu_trap(const char *what) { psp_vfpu_unimplemented(psp_cpu.pc, what); }

/* ---- register addressing ------------------------------------------------- */

int psp_vfpu_regs(uint32_t vreg, int size, int out[4]) {
    const int mtx       = (vreg >> 2) & 7;
    const int col       = vreg & 3;
    int transpose       = (vreg >> 5) & 1;
    int row = 0, len = 1;

    switch (size) {
    case 1: row = (vreg >> 5) & 3; transpose = 0; len = 1; break;
    case 2: row = (vreg >> 5) & 2;                len = 2; break;
    case 3: row = (vreg >> 6) & 1;                len = 3; break;
    default:row = (vreg >> 5) & 2;                len = 4; break;
    }

    for (int i = 0; i < len; i++) {
        /* Transposed access walks columns instead of rows -- the same storage
         * seen the other way round, which is what makes a matrix transpose
         * free on this hardware. */
        const int step = (row + i) & 3;
        out[i] = transpose ? mtx * 4 + step * 32 + col
                           : mtx * 4 + col  * 32 + step;
    }
    return len;
}

/* ---- operands through the prefixes -------------------------------------- */

/* Lanes an element-wise op covers for a width field: 1, 2, 3, or 4. */
static int width(int size) { return size >= 1 && size <= 3 ? size : 4; }

static float sat0(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
static float sat1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }

/* The constants a vpfxs/vpfxt lane can force: indexed by the lane-pick bits,
 * the table chosen by the abs bit. */
static const float PFX_CONST[2][4] = {
    { 0.0f, 1.0f, 2.0f, 0.5f },
    { 3.0f, 1.0f / 3.0f, 0.25f, 1.0f / 6.0f },
};

/* All four lanes of the row or column `vreg` names, starting at its first
 * lane and wrapping, so a prefix can pick any of them for a narrower op.
 * UNVERIFIED: the wrap is our reading of the documentation, not checked on
 * hardware; P2's oracle checks it. For n == 4 it is exactly the operand. */
static void lanes4(uint32_t vreg, int n, int out[4]) {
    if (n == 4) { psp_vfpu_regs(vreg, 4, out); return; }
    const int mtx = (vreg >> 2) & 7, col = vreg & 3;
    const int transpose = n == 1 ? 0 : (vreg >> 5) & 1;
    const int row = n == 1 ? (vreg >> 5) & 3 : n == 2 ? (vreg >> 5) & 2 : (vreg >> 6) & 1;
    for (int i = 0; i < 4; i++) {
        const int step = (row + i) & 3;
        out[i] = transpose ? mtx * 4 + step * 32 + col : mtx * 4 + col * 32 + step;
    }
}

void vfpu_read_raw(uint32_t vreg, int n, float out[4]) {
    int r[4];
    n = psp_vfpu_regs(vreg, n, r);
    for (int i = 0; i < n; i++) out[i] = psp_cpu.v[r[i]];
}

/* Lane i of a vpfxs/vpfxt prefix: bits 2i+1..2i pick a source lane (or a
 * constant's index), bit 8+i is abs (or the constant table), bit 12+i forces
 * a constant, bit 16+i negates. */
void vfpu_read(int which, uint32_t vreg, int n, float out[4]) {
    n = width(n);
    if (!g_prefix_set[which]) { vfpu_read_raw(vreg, n, out); return; }
    int r[4];
    lanes4(vreg, n, r);
    const uint32_t p = g_prefix[which];
    for (int i = 0; i < n; i++) {
        const int pick = (p >> (2 * i)) & 3;
        const int abs  = (p >> (8 + i)) & 1;
        const int cst  = (p >> (12 + i)) & 1;
        const int neg  = (p >> (16 + i)) & 1;
        uint32_t bits = psp_f32_to_bits(cst ? PFX_CONST[abs][pick] : psp_cpu.v[r[pick]]);
        if (!cst && abs) bits &= 0x7FFFFFFFu;      /* on the bits, so NaNs keep their payload */
        if (neg) bits ^= 0x80000000u;
        out[i] = psp_bits_to_f32(bits);
    }
}

/* Lane i of a vpfxd prefix: bits 2i+1..2i saturate (1 to [0, 1], 3 to
 * [-1, 1], 0 and 2 leave the value alone), bit 8+i masks the write. */
void vfpu_write(uint32_t vreg, int n, const float in[4]) {
    int r[4];
    n = psp_vfpu_regs(vreg, width(n), r);
    const uint32_t p = g_prefix_set[PSP_VPFX_D] ? g_prefix[PSP_VPFX_D] : 0;
    for (int i = 0; i < n; i++) {
        if ((p >> (8 + i)) & 1) continue;          /* write mask */
        float v = in[i];
        switch ((p >> (2 * i)) & 3) {
        case 1: v = sat0(v); break;
        case 3: v = sat1(v); break;
        default: break;
        }
        psp_cpu.v[r[i]] = v;
    }
}

void vfpu_write_bits(uint32_t vreg, int n, const uint32_t in[4]) {
    int r[4];
    n = psp_vfpu_regs(vreg, width(n), r);
    const uint32_t p = g_prefix_set[PSP_VPFX_D] ? g_prefix[PSP_VPFX_D] : 0;
    for (int i = 0; i < n; i++)
        if (!((p >> (8 + i)) & 1)) psp_cpu.v[r[i]] = psp_bits_to_f32(in[i]);
}

void vfpu_end(void) { memset(g_prefix_set, 0, sizeof g_prefix_set); }

int vfpu_no_prefix(const char *what) {
    if (!psp_vfpu_prefix_pending()) return 1;
    vfpu_end();
    vfpu_trap(what);
    return 0;
}

/* ---- load / store -------------------------------------------------------- */

void psp_lv_s(uint32_t vt, uint32_t addr) {
    int r[4];
    psp_vfpu_regs(vt, 1, r);
    psp_cpu.v[r[0]] = psp_read_f32(addr & ~3u);
}

void psp_sv_s(uint32_t vt, uint32_t addr) {
    int r[4];
    psp_vfpu_regs(vt, 1, r);
    psp_write_f32(addr & ~3u, psp_cpu.v[r[0]]);
}

void psp_lv_q(uint32_t vt, uint32_t addr) {
    int r[4];
    psp_vfpu_regs(vt, 4, r);
    addr &= ~15u;                       /* quad access is 16-byte aligned */
    for (int i = 0; i < 4; i++)
        psp_cpu.v[r[i]] = psp_read_f32(addr + (uint32_t)i * 4);
}

void psp_sv_q(uint32_t vt, uint32_t addr) {
    int r[4];
    psp_vfpu_regs(vt, 4, r);
    addr &= ~15u;
    for (int i = 0; i < 4; i++)
        psp_write_f32(addr + (uint32_t)i * 4, psp_cpu.v[r[i]]);
}

/* ---- arithmetic ---------------------------------------------------------- */

#define BINOP(name, expr)                                                    \
    void psp_##name(uint32_t vd, uint32_t vs, uint32_t vt, int size) {       \
        const int n = width(size);                                           \
        /* Read every source before writing any destination: vd may alias vs \
         * or vt, and a lane-by-lane read/write would then feed results back  \
         * into later lanes. */                                              \
        float s[4], t[4], d[4];                                              \
        vfpu_read(PSP_VPFX_S, vs, n, s);                                     \
        vfpu_read(PSP_VPFX_T, vt, n, t);                                     \
        for (int i = 0; i < n; i++) {                                        \
            const float a = s[i], b = t[i];                                  \
            d[i] = (expr);                                                   \
        }                                                                    \
        vfpu_write(vd, n, d);                                                \
        vfpu_end();                                                          \
    }

BINOP(vadd, a + b)
BINOP(vsub, a - b)
BINOP(vmul, a * b)
BINOP(vdiv, a / b)
BINOP(vmin, a < b ? a : b)
BINOP(vmax, a > b ? a : b)

/* Dot product: sums all lanes, in lane order, into a single destination lane. */
void psp_vdot(uint32_t vd, uint32_t vs, uint32_t vt, int size) {
    const int n = width(size);
    float s[4], t[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);
    vfpu_read(PSP_VPFX_T, vt, n, t);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += s[i] * t[i];
    vfpu_write(vd, 1, &sum);
    vfpu_end();
}

/* Scale: every lane of vs multiplied by the scalar in vt. */
void psp_vscl(uint32_t vd, uint32_t vs, uint32_t vt, int size) {
    /* What a vpfxt does to vscl's scalar operand lane by lane is not
     * verified, and the game never prefixes it: trap until P2's oracle says.
     * The S and D prefixes apply as for any element-wise op. */
    if (g_prefix_set[PSP_VPFX_T]) { vfpu_end(); vfpu_trap("vscl with vpfxt"); return; }
    const int n = width(size);
    float s[4], t[4], d[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);
    vfpu_read(PSP_VPFX_T, vt, 1, t);
    for (int i = 0; i < n; i++) d[i] = s[i] * t[0];
    vfpu_write(vd, n, d);
    vfpu_end();
}

/* ---- unary element-wise ops (VFPU4) -------------------------------------- */

void psp_vunary(int op, uint32_t vd, uint32_t vs, int size) {
    const int n = width(size);
    float s[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);

    float out[4];
    uint32_t bits[4];
    for (int i = 0; i < n; i++) {
        const float a = s[i];
        float r;
        switch (op) {
        case PSP_VU_MOV:  r = a;            break;
        case PSP_VU_ABS:  r = a < 0 ? -a : a; break;
        case PSP_VU_NEG:  r = -a;           break;
        case PSP_VU_ZERO: r = 0.0f;         break;
        case PSP_VU_ONE:  r = 1.0f;         break;
        case PSP_VU_RCP:  r = 1.0f / a;     break;
        case PSP_VU_NRCP: r = -1.0f / a;    break;
        case PSP_VU_RSQ:  r = 1.0f / psp_fsqrt(a); break;
        case PSP_VU_SQRT: r = psp_fsqrt(a); break;
        /* The PSP's trig takes its argument in *quarter turns*: vsin(x) is
         * sin(x * pi/2), not sin(x). Treating it as radians gives a result
         * that is smooth, plausible, and wrong -- rotations end up at the
         * wrong angle rather than visibly broken. */
        case PSP_VU_SIN:  r = sinf(a * 1.5707963267948966f);  break;
        case PSP_VU_COS:  r = cosf(a * 1.5707963267948966f);  break;
        case PSP_VU_NSIN: r = -sinf(a * 1.5707963267948966f); break;
        case PSP_VU_ASIN: r = asinf(a) * 0.6366197723675814f; break;  /* 2/pi */
        case PSP_VU_EXP2: r = powf(2.0f, a);   break;
        case PSP_VU_REXP2:r = 1.0f / powf(2.0f, a); break;
        case PSP_VU_LOG2: r = logf(a) * 1.4426950408889634f; break;   /* 1/ln2 */
        case PSP_VU_SAT0: r = sat0(a);      break;
        case PSP_VU_SAT1: r = sat1(a);      break;
        /* Conversions move between the float and integer *interpretations* of
         * a vector register; the bits are reinterpreted, not just cast. */
        case PSP_VU_F2IZ: bits[i] = (uint32_t)(int32_t)a; r = 0.0f;  break;
        case PSP_VU_I2F:  r = (float)(int32_t)psp_f32_to_bits(a);    break;
        default:          vfpu_end(); vfpu_trap("vunary"); return;
        }
        out[i] = r;
    }
    /* An integer result takes the write mask but is never saturated. */
    if (op == PSP_VU_F2IZ) vfpu_write_bits(vd, n, bits);
    else                   vfpu_write(vd, n, out);
    vfpu_end();
}

/* ---- matrix ops without a multiply --------------------------------------- */

/* A matrix register names `size` consecutive columns, each a vector of `size`
 * lanes. Writing one column at a time through the same addressing the vector
 * ops use keeps the two consistent -- which matters because a game builds a
 * matrix with these and then transforms with vtfm. */
static void matrix_cols(uint32_t vd, int size, int cols[4][4]) {
    /* Build a fresh vector register per column: matrix, column index, row 0.
     * An earlier version modified bits 6:5 of the incoming register, which
     * varies the *row* field and so walked rows while claiming to walk
     * columns -- every matrix op addressed the wrong elements. */
    const uint32_t mtx       = (vd >> 2) & 7;
    const uint32_t transpose = (vd >> 5) & 1;
    for (int c = 0; c < size; c++) {
        uint32_t vreg = (mtx << 2) | (uint32_t)c | (transpose << 5);
        psp_vfpu_regs(vreg, size, cols[c]);
    }
}

/* Read a whole matrix out into [col][row] order. */
static void matrix_read(uint32_t v, int size, float m[4][4]) {
    int cols[4][4];
    matrix_cols(v, size, cols);
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) m[c][r] = psp_cpu.v[cols[c][r]];
}

static void matrix_write(uint32_t v, int size, const float m[4][4]) {
    int cols[4][4];
    matrix_cols(v, size, cols);
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) psp_cpu.v[cols[c][r]] = m[c][r];
}

void psp_vmidt(uint32_t vd, int size) {
    if (!vfpu_no_prefix("vmidt")) return;
    int cols[4][4];
    matrix_cols(vd, size, cols);
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++)
            psp_cpu.v[cols[c][r]] = (c == r) ? 1.0f : 0.0f;
}

/* vidt -- an identity *vector*: all zeroes but for a single 1.0.
 *
 * Which element gets the 1 follows from the destination register itself:
 * it is the lane whose number equals the register's own index within its
 * matrix -- the column number for C<m><c><r>, the row number for a
 * transposed R<m><r><c> -- held in vd bits 1..0. So vidt.q on C000, C010,
 * C020, C030 builds an identity column by column (the game's matrix builders
 * do exactly this), and C030 alone is (0, 0, 0, 1). A pair takes bit 0 only.
 * An earlier version used bits 6..5, which are always 0 for a quad and put
 * every 1 in lane 0.
 *
 * Single and triple forms are not verified (the game uses neither): they
 * trap rather than guess.
 *
 * This is matrix-setup code. Trapping it to a no-op leaves whatever was in the
 * register, so downstream geometry is built on a basis that is not a basis. */
void psp_vidt(uint32_t vd, int size) {
    const int n = width(size);
    if (n != 2 && n != 4) {
        vfpu_end();
        vfpu_trap(n == 1 ? "vidt.s" : "vidt.t");
        return;
    }
    const int one = (int)(vd & (uint32_t)(n - 1));
    float d[4];
    for (int i = 0; i < n; i++) d[i] = (i == one) ? 1.0f : 0.0f;
    vfpu_write(vd, n, d);
    vfpu_end();
}

/* vcst -- load a constant from the VFPU's built-in table.
 *
 * The index is in bits 20..16 of the word (a_insn.vimm5), not in vs; SNC
 * assembles vcst.s S000, VFPU_PI as 0xD0690000. Values follow the hardware table; index 0 is
 * zero and anything past the end reads as zero rather than as garbage. */
void psp_vcst(uint32_t vd, uint32_t which, int size) {
    static const float K[20] = {
        0.0f,
        3.4028235e38f,          /* max float          */
        1.41421356f,            /* sqrt(2)            */
        0.70710678f,            /* sqrt(1/2)          */
        1.12837917f,            /* 2/sqrt(pi)         */
        0.63661977f,            /* 2/pi               */
        0.31830989f,            /* 1/pi               */
        0.78539816f,            /* pi/4               */
        1.57079633f,            /* pi/2               */
        3.14159265f,            /* pi                 */
        2.71828183f,            /* e                  */
        1.44269504f,            /* log2(e)            */
        0.43429448f,            /* log10(e)           */
        0.69314718f,            /* ln(2)              */
        2.30258509f,            /* ln(10)             */
        6.28318531f,            /* 2*pi               */
        0.52359878f,            /* pi/6               */
        0.30103000f,            /* log10(2)           */
        3.32192809f,            /* log2(10)           */
        0.86602540f,            /* sqrt(3)/2          */
    };

    const float k = which < 20 ? K[which] : 0.0f;
    const int n = width(size);
    float d[4];
    for (int i = 0; i < n; i++) d[i] = k;
    vfpu_write(vd, n, d);
    vfpu_end();
}

void psp_vmzero(uint32_t vd, int size) {
    if (!vfpu_no_prefix("vmzero")) return;
    int cols[4][4];
    matrix_cols(vd, size, cols);
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) psp_cpu.v[cols[c][r]] = 0.0f;
}

void psp_vmone(uint32_t vd, int size) {
    if (!vfpu_no_prefix("vmone")) return;
    int cols[4][4];
    matrix_cols(vd, size, cols);
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) psp_cpu.v[cols[c][r]] = 1.0f;
}

void psp_vmmov(uint32_t vd, uint32_t vs, int size) {
    if (!vfpu_no_prefix("vmmov")) return;
    int dc[4][4], sc[4][4];
    matrix_cols(vd, size, dc);
    matrix_cols(vs, size, sc);
    float tmp[4][4];
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) tmp[c][r] = psp_cpu.v[sc[c][r]];
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) psp_cpu.v[dc[c][r]] = tmp[c][r];
}

/* ---- matrix multiply and transform --------------------------------------- */

/* Scale every element of a matrix by a scalar. Orientation-independent, so
 * this one is unambiguous. */
void psp_vmscl(uint32_t vd, uint32_t vs, uint32_t vt, int size) {
    if (!vfpu_no_prefix("vmscl")) return;
    float m[4][4];
    int t[4];
    matrix_read(vs, size, m);
    psp_vfpu_regs(vt, 1, t);
    const float k = psp_cpu.v[t[0]];
    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) m[c][r] *= k;
    matrix_write(vd, size, m);
}

/* Transform a vector by a matrix: vd[r] = sum over c of M[c][r] * v[c].
 *
 * This is the column-major product a graphics pipeline wants -- the matrix
 * columns are the transformed basis vectors, so transforming (1,0,0,0) yields
 * column 0. That property is what the test pins, and it is the one that
 * distinguishes this from its transpose. */
void psp_vtfm(uint32_t vd, uint32_t vs, uint32_t vt, int size) {
    if (!vfpu_no_prefix("vtfm")) return;
    float m[4][4];
    int d[4], t[4];
    matrix_read(vs, size, m);
    psp_vfpu_regs(vt, size, t);
    psp_vfpu_regs(vd, size, d);

    float in[4], out[4];
    for (int i = 0; i < size; i++) in[i] = psp_cpu.v[t[i]];
    for (int r = 0; r < size; r++) {
        float sum = 0.0f;
        for (int c = 0; c < size; c++) sum += m[c][r] * in[c];
        out[r] = sum;
    }
    /* vd may be one of the sources, so write only after the whole result is
     * computed. */
    for (int r = 0; r < size; r++) psp_cpu.v[d[r]] = out[r];
}

/* Matrix product, composed from the same transform used above so the two
 * cannot disagree: each column of the result is a column of vt transformed by
 * vs.
 *
 * NOTE: the operand ORIENTATION here is not independently verified. The maths
 * is right for the convention stated above, and the identity/composition tests
 * hold, but those hold for the transposed convention too -- they cannot tell
 * the two apart. If recompiled geometry comes out scrambled rather than
 * absent, this is the first thing to check against an oracle. Recorded rather
 * than glossed, because a wrong orientation produces plausible output. */
void psp_vmmul(uint32_t vd, uint32_t vs, uint32_t vt, int size) {
    if (!vfpu_no_prefix("vmmul")) return;
    float a[4][4], b[4][4], out[4][4];
    matrix_read(vs, size, a);
    matrix_read(vt, size, b);

    for (int c = 0; c < size; c++)
        for (int r = 0; r < size; r++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) sum += a[k][r] * b[c][k];
            out[c][r] = sum;
        }
    matrix_write(vd, size, out);
}

/* Compare, writing one condition bit per lane plus the any/all summary bits
 * that vcmov and the bvt/bvf branches read: bits 0..n-1 the lanes, bit 4 any
 * lane, bit 5 all lanes. Lane bits n..3 keep their previous value; that is
 * our reading of the documentation, not verified on hardware -- P2's oracle
 * verifies it. The condition is the low nibble; 8..15 test vs alone. Both
 * operands go through their prefixes either way, and vcmp consumes all
 * three. */
void psp_vcmp(uint32_t cond, uint32_t vs, uint32_t vt, int size) {
    const int n = width(size);
    float s[4], t[4];
    vfpu_read(PSP_VPFX_S, vs, n, s);
    vfpu_read(PSP_VPFX_T, vt, n, t);

    const uint32_t touched = ((1u << n) - 1u) | (1u << 4) | (1u << 5);
    uint32_t cc = psp_cpu.vfpu_cc & ~touched;
    int all = 1, any = 0;
    for (int i = 0; i < n; i++) {
        const float a = s[i], b = t[i];
        int r;
        switch (cond & 0xF) {
        case 0:  r = 0;              break;   /* FL  */
        case 1:  r = (a == b);       break;   /* EQ  */
        case 2:  r = (a <  b);       break;   /* LT  */
        case 3:  r = (a <= b);       break;   /* LE  */
        case 4:  r = 1;              break;   /* TR  */
        case 5:  r = (a != b);       break;   /* NE  */
        case 6:  r = (a >= b);       break;   /* GE  */
        case 7:  r = (a >  b);       break;   /* GT  */
        case 8:  r = (a == 0.0f);                 break;   /* EZ */
        case 9:  r = isnan(a);                    break;   /* EN */
        case 10: r = isinf(a);                    break;   /* EI */
        case 11: r = isnan(a) || isinf(a);        break;   /* ES */
        case 12: r = (a != 0.0f);                 break;   /* NZ */
        case 13: r = !isnan(a);                   break;   /* NN */
        case 14: r = !isinf(a);                   break;   /* NI */
        default: r = !(isnan(a) || isinf(a));     break;   /* NS (15) */
        }
        if (r) { cc |= 1u << i; any = 1; } else { all = 0; }
    }
    if (any) cc |= 1u << 4;
    if (all) cc |= 1u << 5;
    psp_cpu.vfpu_cc = cc;
    vfpu_end();
}

/* viim / vfim -- write a single lane from an immediate encoded in the
 * instruction. There is no source operand to rewrite, but the write goes
 * through the D prefix (mask, saturate) and the op consumes all three. */
void psp_vimm(uint32_t vd, float value) {
    vfpu_write(vd, 1, &value);
    vfpu_end();
}
