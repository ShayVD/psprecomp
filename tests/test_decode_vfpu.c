/* VFPU decoding against SNC pspas encodings. No game data. */
#include "decode.h"

#include <stdio.h>
#include <string.h>

static int failures;

static void dec(uint32_t w, a_op want, const char *name) {
    a_insn in;
    a_decode(w, 0x08900000u, &in);
    if (in.op != want || strcmp(a_mnemonic(in.op), name) != 0) {
        printf("FAIL 0x%08X: got %s, want %s\n", w, a_mnemonic(in.op), name);
        failures++;
    }
}

#define FIELD(w, expr, label)                                             \
    do { a_insn in; a_decode((w), 0x08900000u, &in);                      \
         if (!(expr)) { printf("FAIL 0x%08X: %s\n", (w), label); failures++; } } while (0)

int main(void) {
    dec(0xFFFF0000u, A_VNOP,   "vnop");
    dec(0xFFFF0320u, A_VSYNC,  "vsync");
    dec(0xFFFF040Du, A_VFLUSH, "vflush");
    dec(0x63C02000u, A_VDIV,   "vdiv");      /* vdiv.s S000, S001, S002 */
    dec(0x63828180u, A_VDIV,   "vdiv");      /* vdiv.q */
    dec(0x63890808u, A_VDIV,   "vdiv");      /* from the game */
    dec(0x61402000u, A_VSBN,   "vsbn");
    dec(0x62000000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU0 sub-op 4 is not an instruction */

    /* VFPU7 */
    dec(0xD0200000u, A_VRNDS,  "vrnds");
    dec(0xD0210000u, A_VRNDI,  "vrndi");
    dec(0xD0220000u, A_VRNDF1, "vrndf1");
    dec(0xD0238080u, A_VRNDF2, "vrndf2");
    dec(0xD0320180u, A_VF2H,   "vf2h");
    dec(0xD0330100u, A_VH2F,   "vh2f");
    dec(0xD0360100u, A_VSBZ,   "vsbz");
    dec(0xD0370100u, A_VLGB,   "vlgb");
    dec(0xD0380100u, A_VUC2I,  "vuc2i");
    dec(0xD0390100u, A_VC2I,   "vc2i");
    dec(0xD03A0100u, A_VUS2I,  "vus2i");
    dec(0xD03B0100u, A_VS2I,   "vs2i");
    dec(0xD03C8180u, A_VI2UC,  "vi2uc");
    dec(0xD03D8180u, A_VI2C,   "vi2c");
    dec(0xD03E8180u, A_VI2US,  "vi2us");
    dec(0xD03F0180u, A_VI2S,   "vi2s");

    /* VFPU9 */
    dec(0xD0408180u, A_VSRT1,  "vsrt1");
    dec(0xD0418180u, A_VSRT2,  "vsrt2");
    dec(0xD0428180u, A_VBFY1,  "vbfy1");
    dec(0xD0438180u, A_VBFY2,  "vbfy2");
    dec(0xD0448180u, A_VOCP,   "vocp");
    dec(0xD0450100u, A_VSOCP,  "vsocp");
    dec(0xD0468180u, A_VFAD,   "vfad");
    dec(0xD0478180u, A_VAVG,   "vavg");
    dec(0xD0488180u, A_VSRT3,  "vsrt3");
    dec(0xD0498180u, A_VSRT4,  "vsrt4");
    dec(0xD04A8180u, A_VSGN,   "vsgn");
    dec(0xD0508300u, A_VMFVC,  "vmfvc");
    dec(0xD0510083u, A_VMTVC,  "vmtvc");
    dec(0xD0598180u, A_VT4444, "vt4444");
    dec(0xD05A8180u, A_VT5551, "vt5551");
    dec(0xD05B8180u, A_VT5650, "vt5650");

    /* VFPU4 conversions, vcmov, VFPU6 */
    dec(0xD2038180u, A_VF2IN,  "vf2in");
    dec(0xD2208180u, A_VF2IZ,  "vf2iz");
    dec(0xD2410100u, A_VF2IU,  "vf2iu");
    dec(0xD2620100u, A_VF2ID,  "vf2id");
    dec(0xD2908180u, A_VI2F,   "vi2f");
    dec(0xD2A00100u, A_VCMOV,  "vcmov");
    dec(0xD2AE8180u, A_VCMOV,  "vcmov");
    dec(0xF2828100u, A_VCRSP,  "vcrsp");    /* .t */
    dec(0xF2828180u, A_VQMUL,  "vqmul");    /* .q, same rs */
    dec(0xF3B18180u, A_VROT,   "vrot");
    dec(0x6F028180u, A_VSGE,   "vsge");
    dec(0x6F828180u, A_VSLT,   "vslt");

    /* COP2 moves: bit 7 of the register field selects a control register. */
    dec(0x48640001u, A_MFV,    "mfv");
    dec(0x48E40066u, A_MTV,    "mtv");
    dec(0x48640083u, A_MFVC,   "mfvc");
    dec(0x48E40083u, A_MTVC,   "mtvc");

    FIELD(0x48640001u, in.rt == 4 && in.vd == 0x01, "mfv $a0, S010 fields");
    FIELD(0x48E40066u, in.rt == 4 && in.vd == 0x66, "mtv $a0, S123 fields");
    FIELD(0xF3B18180u, in.vimm5 == 0x11 && in.vsize == 4 && in.vs == 0x01, "vrot.q [-s,c,0,0] code");
    FIELD(0xF3A08100u, in.vimm5 == 0x00 && in.vsize == 3, "vrot.t [c,s,s] code");
    FIELD(0xD2AE8180u, (in.vimm5 & 0xF) == 0x0E && in.vsize == 4, "vcmovf.q cc 6");
    FIELD(0xD2A00100u, (in.vimm5 & 0xF) == 0x00 && in.vsize == 1, "vcmovt.s cc 0");
    FIELD(0xD2038180u, in.vimm5 == 3, "vf2in scale 3");
    FIELD(0xD2908180u, in.vimm5 == 16, "vi2f scale 16");
    FIELD(0xF2828100u, in.vsize == 3, "vcrsp is a triple");

    /* vcst's constant index is bits 20..16, not vs. pspas: vcst.s S000,
     * VFPU_PI is 0xD0690000; VFPU_2_PI into S010 is 0xD0650001; the game's
     * 2/pi at 0x08858B60 is 0xD0650024. */
    dec(0xD0690000u, A_VCST, "vcst");
    FIELD(0xD0690000u, in.vimm5 == 9 && in.vd == 0 && in.vs == 0, "vcst PI index in vimm5");
    FIELD(0xD0650001u, in.vimm5 == 5 && in.vd == 1 && in.vsize == 1, "vcst 2_PI into S010");
    FIELD(0xD0650024u, in.vimm5 == 5 && in.vd == 0x24, "vcst from the game");

    /* vwbn's immediate is 8 bits, 23..16 (pspas: vwbn.s S000, S000, 171). */
    dec(0xD3AB0000u, A_VWBN, "vwbn");
    FIELD(0xD3AB0000u, in.vimm8 == 0xAB, "vwbn imm8");

    /* Control registers: mfvc/mtvc vd = reg - 128; vmfvc has the control
     * register in vs, vmtvc in vd. vsize means nothing for these. */
    FIELD(0x48640083u, in.vd == 3 && in.rt == 4, "mfvc $a0, $131");
    FIELD(0x48E40083u, in.vd == 3 && in.rt == 4, "mtvc $a0, $131");
    FIELD(0xD0508301u, in.vs == 3 && in.vd == 1, "vmfvc S010, $131");
    FIELD(0xD0510183u, in.vd == 3 && in.vs == 1, "vmtvc $131, S010");

    /* vcrsp is a triple op and vqmul a quad op; nothing else is defined. */
    dec(0xF2820000u, A_VFPU_UNKNOWN, "vfpu?");   /* .s */
    dec(0xF2820080u, A_VFPU_UNKNOWN, "vfpu?");   /* .p */

    /* VFPU6 transforms: bits 25..23 = 1/2/3 give the matrix order 2/3/4. A
     * vector as wide as the order is vtfmN; one lane narrower is the
     * homogeneous vhtfmN (an implicit last lane of 1.0). Words from decode.c's
     * layout: vt 2 (bits 22..16), vs 1, vd 0, width bits 15 and 7. */
    dec(0xF1828180u, A_VTFM4,  "vtfm4");    /* .q */
    dec(0xF1828100u, A_VHTFM4, "vhtfm4");   /* .t */
    dec(0xF1028100u, A_VTFM3,  "vtfm3");    /* .t */
    dec(0xF1020180u, A_VHTFM3, "vhtfm3");   /* .p */
    dec(0xF0820180u, A_VTFM2,  "vtfm2");    /* .p */
    dec(0xF0820100u, A_VHTFM2, "vhtfm2");   /* .s */
    dec(0xF180A408u, A_VHTFM4, "vhtfm4");   /* the game's, at 0x08827738 */
    /* Any other width keeps the vtfm name; the emitter refuses it. */
    dec(0xF1820180u, A_VTFM4,  "vtfm4");    /* .p */
    dec(0xF0828180u, A_VTFM2,  "vtfm2");    /* .q */

    /* vsync carries a 16-bit immediate; 0x0000 is vnop, 0x040D vflush. */
    dec(0xFFFF0001u, A_VSYNC, "vsync");
    dec(0xFFFF8000u, A_VSYNC, "vsync");
    dec(0xFFFE0000u, A_VFPU_UNKNOWN, "vfpu?");
    dec(0xFFFF0000u, A_VNOP, "vnop");

    /* Holes in the tables stay unknown. */
    dec(0x61800000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU0 sub-op 3 */
    dec(0x62800000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU0 sub-op 5 */
    dec(0x63000000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU0 sub-op 6 */
    dec(0xD0240000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU7 rt=4 */
    dec(0xD04B0000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU9 rt=0x0B */
    dec(0xD0520000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU9 rt=0x12 */
    dec(0xD0580000u, A_VFPU_UNKNOWN, "vfpu?");   /* VFPU9 rt=0x18 */
    dec(0xD0800000u, A_VFPU_UNKNOWN, "vfpu?");   /* 0x34 rs=4 */

    /* Operand fields. */
    FIELD(0xD0202000u, in.vs == 0x20, "vrnds.s S001 seed in vs");
    FIELD(0xD0210040u, in.vd == 0x40, "vrndi.s S002 dest in vd");
    FIELD(0x63C02000u, in.vd == 0 && in.vs == 0x20 && in.vt == 0x40, "vdiv.s fields");
    dec(0x49150000u, A_BVT, "bvt");
    FIELD(0x49150000u, (in.vimm5 >> 2) == 5, "bvt 5 condition index");

    /* Disassembly shows the immediate that selects the operation. */
    {
        a_insn in; char buf[96];
        a_decode(0xD0690000u, 0x08900000u, &in);
        a_format(&in, buf, sizeof buf);
        if (!strstr(buf, "vcst") || !strstr(buf, "9")) { printf("FAIL vcst format: \"%s\"\n", buf); failures++; }
        a_decode(0xD2908180u, 0x08900000u, &in);
        a_format(&in, buf, sizeof buf);
        if (!strstr(buf, "16")) { printf("FAIL vi2f format: \"%s\"\n", buf); failures++; }
        a_decode(0xF3B18180u, 0x08900000u, &in);
        a_format(&in, buf, sizeof buf);
        if (!strstr(buf, "0x11") && !strstr(buf, "17")) { printf("FAIL vrot format: \"%s\"\n", buf); failures++; }
        a_decode(0x48640083u, 0x08900000u, &in);
        a_format(&in, buf, sizeof buf);
        if (!strstr(buf, "$131")) { printf("FAIL mfvc format: \"%s\"\n", buf); failures++; }
        a_decode(0x48E40083u, 0x08900000u, &in);
        a_format(&in, buf, sizeof buf);
        if (!strstr(buf, "$131")) { printf("FAIL mtvc format: \"%s\"\n", buf); failures++; }
    }

    printf("%s\n", failures ? "test_decode_vfpu: FAILED" : "test_decode_vfpu: ok");
    return failures ? 1 : 0;
}
