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

    printf("%s\n", failures ? "test_decode_vfpu: FAILED" : "test_decode_vfpu: ok");
    return failures ? 1 : 0;
}
