/* The trace hook sees name, arguments, return value and $ra for registered
 * and unregistered NIDs alike; the pre-call hook sees the same before the
 * handler runs. Synthetic, no game data. */
#include "psprecomp/hle.h"
#include "psprecomp/cpu.h"

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s\n", m); failures++; } } while (0)

static struct { uint32_t nid; const char *name; uint32_t args[8], ret, hi, ra; int n; } seen;

static void hook(uint32_t nid, const char *name, const uint32_t args[8], uint32_t ret, uint32_t hi, uint32_t ra) {
    seen.nid = nid; seen.name = name; memcpy(seen.args, args, sizeof seen.args);
    seen.ret = ret; seen.hi = hi; seen.ra = ra; seen.n++;
}

static struct { uint32_t nid; const char *name; uint32_t args[8], ra; int n, handled_before; } pre;
static int handled;   /* calls of fake() so far */

static void pre_hook(uint32_t nid, const char *name, const uint32_t args[8], uint32_t ra) {
    pre.nid = nid; pre.name = name; memcpy(pre.args, args, sizeof pre.args);
    pre.ra = ra; pre.handled_before = handled; pre.n++;
}

static void fake(void) {
    handled++;
    uint32_t r = psp_arg(0) + 1;
    psp_cpu.r[PSP_REG_A0] = 0;              /* handler clobbers an argument */
    psp_cpu.r[PSP_REG_V1] = 0xABCD0000u;    /* high half of a 64-bit return */
    psp_ret(r);
}

int main(void) {
    psp_cpu_reset();
    psp_hle_register(0x109F50BCu, "IoFileMgrForUser", "sceIoOpen", fake);
    psp_hle_set_trace(hook);

    psp_cpu.r[PSP_REG_A0] = 41;
    for (int i = 1; i < 4; i++) psp_cpu.r[PSP_REG_A0 + i] = 0x100 + i;
    for (int i = 0; i < 4; i++) psp_cpu.r[PSP_REG_T0 + i] = 0x200 + i;
    psp_cpu.r[PSP_REG_RA] = 0x08804100u;
    psp_hle_call(0x109F50BCu);
    CHECK(seen.n == 1, "hook called once");
    CHECK(seen.nid == 0x109F50BCu, "nid");
    CHECK(seen.name && !strcmp(seen.name, "sceIoOpen"), "name");
    CHECK(seen.args[0] == 41, "a0 is the entry value, not the clobbered one");
    for (int i = 1; i < 4; i++) CHECK(seen.args[i] == (uint32_t)(0x100 + i), "a1-a3");
    for (int i = 0; i < 4; i++) CHECK(seen.args[4 + i] == (uint32_t)(0x200 + i), "t0-t3");
    CHECK(seen.hi == 0xABCD0000u, "ret_hi is $v1");
    CHECK(seen.ret == 42, "return value");
    CHECK(seen.ra == 0x08804100u, "caller $ra");

    psp_hle_call(0xDEADBEEFu);
    CHECK(seen.n == 2 && seen.name == NULL && seen.ret == 0, "unregistered NID traced with NULL name");

    psp_hle_set_trace(NULL);
    psp_hle_call(0x109F50BCu);
    CHECK(seen.n == 2, "hook removed");

    /* The pre-call hook runs before the handler, with the entry arguments. */
    psp_hle_set_pre_call(pre_hook);
    psp_cpu.r[PSP_REG_A0] = 41;
    psp_cpu.r[PSP_REG_T3] = 0x203;
    psp_cpu.r[PSP_REG_RA] = 0x08804200u;
    int before = handled;
    psp_hle_call(0x109F50BCu);
    CHECK(pre.n == 1, "pre-call hook called once");
    CHECK(pre.handled_before == before && handled == before + 1, "pre-call hook runs before the handler");
    CHECK(pre.nid == 0x109F50BCu && pre.name && !strcmp(pre.name, "sceIoOpen"), "pre-call nid and name");
    CHECK(pre.args[0] == 41 && pre.args[7] == 0x203, "pre-call args");
    CHECK(pre.ra == 0x08804200u, "pre-call $ra");
    CHECK(seen.n == 2, "trace hook stays off");

    psp_hle_call(0xDEADBEEFu);
    CHECK(pre.n == 2 && pre.nid == 0xDEADBEEFu && pre.name == NULL, "pre-call sees unregistered NID with NULL name");

    psp_hle_set_trace(hook);
    psp_hle_call(0x109F50BCu);
    CHECK(pre.n == 3 && seen.n == 3, "both hooks on together");

    psp_hle_set_trace(NULL);
    psp_hle_set_pre_call(NULL);
    psp_hle_call(0x109F50BCu);
    CHECK(pre.n == 3, "pre-call hook removed");

    printf("%s\n", failures ? "test_hle_trace: FAILED" : "test_hle_trace: ok");
    return failures ? 1 : 0;
}
