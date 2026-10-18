/* Address pairs: lui/addiu and lui/ori that build a code address become seeds.
 * Each block below pins one rule of a_scan_address_pairs. Synthetic code
 * only, no game data. */
#include "analyze.h"
#include "decode.h"

#include <stdio.h>
#include <string.h>

/* The buffer straddles a 64 KiB boundary (0x08810000) so that both targets
 * with a low half >= 0x8000 and the matching carry into the high half occur. */
#define BASE   0x0880F000u
#define NBYTES 0x2000u
#define NWORDS (NBYTES / 4)
#define T(k)   (BASE + 0x400u + 4u * (uint32_t)(k))   /* decodable: the buffer is zero */

static uint32_t lui(int rt, uint32_t imm) { return 0x3C000000u | (uint32_t)rt << 16 | (imm & 0xFFFF); }
static uint32_t addiu(int rt, int rs, int imm) {
    return 0x24000000u | (uint32_t)rs << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
static uint32_t ori(int rt, int rs, uint32_t imm) {
    return 0x34000000u | (uint32_t)rs << 21 | (uint32_t)rt << 16 | (imm & 0xFFFF);
}
static uint32_t lw(int rt, int imm, int base) {
    return 0x8C000000u | (uint32_t)base << 21 | (uint32_t)rt << 16 | ((uint32_t)imm & 0xFFFF);
}
static uint32_t addu(int rd, int rs, int rt) {
    return (uint32_t)rs << 21 | (uint32_t)rt << 16 | (uint32_t)rd << 11 | 0x21u;
}
static uint32_t j(uint32_t target)   { return 0x08000000u | ((target >> 2) & 0x03FFFFFFu); }
static uint32_t jal(uint32_t target) { return 0x0C000000u | ((target >> 2) & 0x03FFFFFFu); }
static uint32_t beq_zero(int words)  { return 0x10000000u | ((uint32_t)words & 0xFFFF); }  /* b */
#define HI(a) (((a) + 0x8000u) >> 16)                 /* for lui + addiu */
#define LO(a) ((int)(int16_t)((a) & 0xFFFF))
#define JR_RA 0x03E00008u
#define NOP   0u
#define BAD   0x4C000000u                          /* opcode 0x13: no such instruction */
#define ZERO 0
#define A0 4
#define A1 5
#define T0 8

typedef struct {
    uint32_t scan_bytes;               /* 0: the whole buffer */
    uint32_t stub_addr, stub_size;
} scan_opts;

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

/* Scan `ins[0..n)` placed at the start of an otherwise zero buffer. */
static int scan(const uint32_t *ins, int n, const scan_opts *o, uint32_t *found, int max) {
    static uint32_t w[NWORDS];
    static uint8_t code[NBYTES];
    memset(w, 0, sizeof w);
    memcpy(w, ins, (size_t)n * sizeof *ins);
    for (size_t k = 0; k < NWORDS; k++)
        for (int b = 0; b < 4; b++) code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = NBYTES;
    an.scan_base = BASE;
    an.scan_size = o && o->scan_bytes ? o->scan_bytes : NBYTES;
    if (o) { an.stub_addr = o->stub_addr; an.stub_size = o->stub_size; }
    int total = a_scan_address_pairs(&an, found, max);
    CHECK(a_scan_address_pairs(&an, NULL, 0) == total, "counting pass agrees (%d)", total);
    return total;
}

static int has(const uint32_t *found, int n, uint32_t t) {
    for (int k = 0; k < n; k++) if (found[k] == t) return 1;
    return 0;
}

/* `ins` yields exactly the targets in `want` (any order). */
static void expect(const char *what, const uint32_t *ins, int n, const scan_opts *o,
                   const uint32_t *want, int nwant) {
    uint32_t found[16];
    int got = scan(ins, n, o, found, 16);
    CHECK(got == nwant, "%s: expected %d targets, got %d", what, nwant, got);
    for (int k = 0; k < nwant; k++)
        CHECK(has(found, got < 16 ? got : 16, want[k]), "%s: target 0x%08X found", what, want[k]);
}

static void test_basic_and_discovery(void) {
    /* Two pairs, one lui/addiu and one lui/ori with a different register for
     * the low half; f1 and f2 are reachable only through them. A lui whose
     * register is overwritten by a load is not a pair, and a pair that points
     * outside the scan range is dropped. */
    const uint32_t F1 = BASE + 0x40, F2 = BASE + 0x48, DATA = BASE + 0x80;
    uint32_t w[0x90 / 4];
    memset(w, 0, sizeof w);
    int i = 0;
    w[i++] = lui(A0, HI(F1));
    w[i++] = addiu(A0, A0, LO(F1));      /* -> F1 */
    w[i++] = lui(T0, F2 >> 16);
    w[i++] = ori(A1, T0, F2 & 0xFFFF);   /* -> F2 (rt != rs) */
    w[i++] = lui(A0, BASE >> 16);
    w[i++] = lw(A0, 0, A0);              /* clobbers $a0 (see test_register_tracking) */
    w[i++] = addiu(A1, A0, 0x4040);
    w[i++] = lui(A0, HI(DATA));
    w[i++] = addiu(A0, A0, LO(DATA));    /* -> DATA: outside the scan range */
    w[i++] = JR_RA;
    w[i++] = NOP;
    w[0x40 / 4] = JR_RA; w[0x44 / 4] = NOP;   /* f1 */
    w[0x48 / 4] = JR_RA; w[0x4C / 4] = NOP;   /* f2 */

    scan_opts o = { .scan_bytes = 0x50 };
    const uint32_t want[] = { F1, F2 };
    expect("basic", w, i, &o, want, 2);

    uint8_t code[sizeof w];
    for (size_t k = 0; k < sizeof w / 4; k++)
        for (int b = 0; b < 4; b++) code[k * 4 + b] = (uint8_t)(w[k] >> (8 * b));
    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = (uint32_t)sizeof code;
    an.scan_base = BASE;
    an.scan_size = 0x50;
    uint32_t seeds[3] = { BASE, F1, F2 };
    CHECK(a_discover(&an, seeds, 3) == 0, "discover");
    int got = 0;
    for (int k = 0; k < an.nfuncs; k++)
        if (an.funcs[k].addr == F1 || an.funcs[k].addr == F2) got++;
    CHECK(got == 2, "both pointer-only functions discovered (%d)", got);
    a_analysis_free(&an);
}

static void test_sign_paths(void) {
    /* Low half >= 0x8000: addiu sees a negative immediate and the high half
     * carries (this is crt0's `addiu $a1,$a1,-0x4194`). */
    const uint32_t NEG = BASE + 0x800;                 /* 0x0880F800 */
    uint32_t w1[] = { lui(A1, HI(NEG)), addiu(A1, A1, LO(NEG)) };
    CHECK(LO(NEG) < 0 && HI(NEG) == (NEG >> 16) + 1, "test setup: NEG needs the carry");
    expect("addiu negative", w1, 2, NULL, &NEG, 1);

    /* lui/ori with bit 15 set: ori zero-extends, no carry. */
    uint32_t w2[] = { lui(A1, NEG >> 16), ori(A1, A1, NEG & 0xFFFF) };
    expect("ori bit 15", w2, 2, NULL, &NEG, 1);

    /* The wrong carry convention would land 0x10000 away: not our target. */
    uint32_t w3[] = { lui(A1, NEG >> 16), addiu(A1, A1, LO(NEG)) };   /* = NEG - 0x10000 */
    uint32_t found[4];
    int n = scan(w3, 2, NULL, found, 4);
    CHECK(n == 0, "addiu without carry points below the buffer (%d)", n);
}

static void test_jump_boundary(void) {
    /* (a) lui before `jr $ra`: the delay slot is still part of the line. */
    uint32_t in_slot[] = { lui(A0, HI(T(1))), JR_RA, addiu(A0, A0, LO(T(1))) };
    expect("low half in delay slot", in_slot, 3, NULL, &(uint32_t){ T(1) }, 1);

    uint32_t past_slot[] = { lui(A0, HI(T(1))), JR_RA, NOP, addiu(A0, A0, LO(T(1))) };
    expect("low half after delay slot", past_slot, 4, NULL, NULL, 0);

    /* A call returns, so the line continues past it. */
    uint32_t over_call[] = { lui(A0, HI(T(2))), jal(T(0)), NOP, addiu(A0, A0, LO(T(2))) };
    expect("through a call", over_call, 4, NULL, &(uint32_t){ T(2) }, 1);

    /* `b` (beq $zero,$zero) is local flow, not a stop: a lui before an if/else
     * dominates both arms. */
    uint32_t over_b[] = { lui(A0, HI(T(3))), beq_zero(1), NOP, NOP, addiu(A0, A0, LO(T(3))) };
    expect("through an unconditional branch", over_b, 5, NULL, &(uint32_t){ T(3) }, 1);

    /* A lui in the delay slot of an unconditional jump feeds the jump target,
     * not the code that follows in memory. */
    uint32_t slot_j[] = { j(T(0)), lui(A0, HI(T(4))), addiu(A0, A0, LO(T(4))) };
    expect("lui in a j delay slot", slot_j, 3, NULL, NULL, 0);
    uint32_t slot_jr[] = { JR_RA, lui(A0, HI(T(4))), addiu(A0, A0, LO(T(4))) };
    expect("lui in a jr delay slot", slot_jr, 3, NULL, NULL, 0);
    /* ...but the delay slot of a call falls through to the next word. */
    uint32_t slot_jal[] = { jal(T(0)), lui(A0, HI(T(4))), addiu(A0, A0, LO(T(4))) };
    expect("lui in a jal delay slot", slot_jal, 3, NULL, &(uint32_t){ T(4) }, 1);
}

static void test_register_tracking(void) {
    /* (b) the low half replaces the high half; a later use of the same
     * register sees the new value, not the lui. */
    uint32_t replaced[] = { lui(A0, HI(T(1))), addiu(A0, A0, LO(T(1))), addiu(A1, A0, LO(T(2))) };
    expect("low half replaces the high half", replaced, 3, NULL, &(uint32_t){ T(1) }, 1);

    /* Any write to the register between lui and low half voids the pair. The
     * would-be target is in range, so only the clobber check rejects these. */
    uint32_t by_load[] = { lui(A0, HI(T(1))), lw(A0, 0, A0), addiu(A1, A0, LO(T(1))) };
    expect("clobbered by a load", by_load, 3, NULL, NULL, 0);
    uint32_t by_alu[] = { lui(A0, HI(T(2))), addu(A0, A1, 6), addiu(A1, A0, LO(T(2))) };
    expect("clobbered by an ALU op", by_alu, 3, NULL, NULL, 0);

    /* (c) a lui into a register that the low half does not overwrite can feed
     * several low halves. */
    uint32_t shared[] = { lui(T0, HI(T(1))), addiu(A0, T0, LO(T(1))), addiu(A1, T0, LO(T(2))) };
    expect("one lui, two low halves", shared, 3, NULL, (uint32_t[]){ T(1), T(2) }, 2);
}

static void test_window(void) {
    /* (d) the low half may sit up to 16 instructions after the lui. */
    for (int dist = 15; dist <= 17; dist++) {
        uint32_t w[24];
        memset(w, 0, sizeof w);
        w[0] = lui(A0, HI(T(1)));
        w[dist] = addiu(A0, A0, LO(T(1)));
        char what[32];
        snprintf(what, sizeof what, "low half at distance %d", dist);
        expect(what, w, dist + 1, NULL, &(uint32_t){ T(1) }, dist <= 16 ? 1 : 0);
    }
}

static void test_rejected(void) {
    /* (e) lui into $zero is not a value. */
    uint32_t zero[] = { lui(ZERO, HI(T(1))), addiu(A0, ZERO, LO(T(1))) };
    expect("lui $zero", zero, 2, NULL, NULL, 0);

    /* A misaligned target is not an instruction address. */
    uint32_t odd[] = { lui(A0, HI(T(1) + 2)), addiu(A0, A0, LO(T(1) + 2)) };
    expect("misaligned target", odd, 2, NULL, NULL, 0);

    /* An import stub is HLE, not code to translate. */
    scan_opts o = { .stub_addr = T(8), .stub_size = 0x40 };
    uint32_t stub[] = { lui(A0, HI(T(9))), addiu(A0, A0, LO(T(9))),     /* in the stub */
                        lui(A1, HI(T(1))), addiu(A1, A1, LO(T(1))) };   /* control */
    expect("import stub target", stub, 4, &o, &(uint32_t){ T(1) }, 1);

    /* Outside the buffer entirely. */
    uint32_t far[] = { lui(A0, 0x0900), addiu(A0, A0, 0x40) };
    expect("target outside", far, 2, NULL, NULL, 0);

    /* The window stops at the end of the scan range, even though the buffer
     * (the walkable extent) goes on. */
    scan_opts edge = { .scan_bytes = 8 };
    uint32_t cut[] = { NOP, lui(A0, HI(T(1))), addiu(A0, A0, LO(T(1))) };
    expect("low half past the scan range", cut, 3, &edge, NULL, 0);

    /* Undecodable data between the halves ends the line. */
    a_insn d;
    a_decode(BAD, 0, &d);
    CHECK(d.op == A_INVALID, "test setup: BAD is invalid");
    uint32_t bad[] = { lui(A0, HI(T(1))), BAD, addiu(A0, A0, LO(T(1))) };
    expect("invalid word between", bad, 3, NULL, NULL, 0);
}

int main(void) {
    test_basic_and_discovery();
    test_sign_paths();
    test_jump_boundary();
    test_register_tracking();
    test_window();
    test_rejected();
    printf("%s\n", failures ? "test_analyze_pairs: FAILED" : "test_analyze_pairs: ok");
    return failures ? 1 : 0;
}
