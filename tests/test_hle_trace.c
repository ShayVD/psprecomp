/* The trace hook sees name, arguments, return value and $ra for registered
 * and unregistered NIDs alike. Synthetic, no game data. */
#include "psprecomp/hle.h"
#include "psprecomp/cpu.h"

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s\n", m); failures++; } } while (0)

static struct { uint32_t nid; const char *name; uint32_t a0, t3, ret, ra; int n; } seen;

static void hook(uint32_t nid, const char *name, const uint32_t args[8], uint32_t ret, uint32_t ra) {
    seen.nid = nid; seen.name = name; seen.a0 = args[0]; seen.t3 = args[7];
    seen.ret = ret; seen.ra = ra; seen.n++;
}

static void fake(void) { psp_ret(psp_arg(0) + 1); }

int main(void) {
    psp_cpu_reset();
    psp_hle_register(0x109F50BCu, "IoFileMgrForUser", "sceIoOpen", fake);
    psp_hle_set_trace(hook);

    psp_cpu.r[PSP_REG_A0] = 41;
    psp_cpu.r[PSP_REG_T3] = 0x1234;
    psp_cpu.r[PSP_REG_RA] = 0x08804100u;
    psp_hle_call(0x109F50BCu);
    CHECK(seen.n == 1, "hook called once");
    CHECK(seen.nid == 0x109F50BCu, "nid");
    CHECK(seen.name && !strcmp(seen.name, "sceIoOpen"), "name");
    CHECK(seen.a0 == 41 && seen.t3 == 0x1234, "args captured before the call");
    CHECK(seen.ret == 42, "return value");
    CHECK(seen.ra == 0x08804100u, "caller $ra");

    psp_hle_call(0xDEADBEEFu);
    CHECK(seen.n == 2 && seen.name == NULL && seen.ret == 0, "unregistered NID traced with NULL name");

    psp_hle_set_trace(NULL);
    psp_hle_call(0x109F50BCu);
    CHECK(seen.n == 2, "hook removed");

    printf("%s\n", failures ? "test_hle_trace: FAILED" : "test_hle_trace: ok");
    return failures ? 1 : 0;
}
