/* elf_parse must report every executable PROGBITS section (.init, .text,
 * .fini, ...) but not the import-stub section, and not data sections. */

#include "container.h"

#include <stdio.h>
#include <string.h>

static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static int failures;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, m); failures++; } } while (0)

static void shdr(uint8_t *sh, uint32_t name, uint32_t type, uint32_t flags,
                 uint32_t addr, uint32_t off, uint32_t size) {
    put32(sh + 0, name); put32(sh + 4, type); put32(sh + 8, flags);
    put32(sh + 12, addr); put32(sh + 16, off); put32(sh + 20, size);
}

int main(void) {
    static const char strtab[] =
        "\0.init\0.text\0.fini\0.sceStub.text\0.rodata\0.shstrtab";
    enum { N_INIT = 1, N_TEXT = 7, N_FINI = 13, N_STUB = 19, N_RO = 33, N_STR = 41 };

    uint8_t buf[1024];
    memset(buf, 0, sizeof buf);
    memcpy(buf, "\x7f" "ELF", 4);
    buf[4] = 1; buf[5] = 1;
    put16(buf + 16, 2);                 /* ET_EXEC */
    put32(buf + 24, 0x08804040);
    put32(buf + 28, 52);                /* phoff */
    put32(buf + 32, 0x200);             /* shoff */
    put16(buf + 42, 32);
    put16(buf + 44, 1);
    put16(buf + 46, 40);                /* shentsize */
    put16(buf + 48, 7);                 /* shnum: null + 6 */
    put16(buf + 50, 6);                 /* shstrndx */

    uint8_t *ph = buf + 52;
    put32(ph + 0, 1); put32(ph + 4, 0x100); put32(ph + 8, 0x08804018);
    put32(ph + 16, 0x100); put32(ph + 20, 0x100); put32(ph + 24, 0x5);

    memcpy(buf + 0x180, strtab, sizeof strtab);

    uint8_t *s = buf + 0x200;
    shdr(s + 40 * 1, N_INIT, 1, 0x6, 0x08804018, 0x100, 0x28);
    shdr(s + 40 * 2, N_TEXT, 1, 0x6, 0x08804040, 0x128, 0x20);
    shdr(s + 40 * 3, N_FINI, 1, 0x6, 0x08B2FE34, 0x148, 0x8);
    shdr(s + 40 * 4, N_STUB, 1, 0x6, 0x08B2FE40, 0x150, 0x10);
    shdr(s + 40 * 5, N_RO,   1, 0x2, 0x08B30000, 0x160, 0x10);
    shdr(s + 40 * 6, N_STR,  3, 0x0, 0, 0x180, sizeof strtab);

    elf_info e;
    CHECK(elf_parse(buf, sizeof buf, &e) == 0, "parse");
    /* .init and .text touch, so they merge; .fini stays separate. */
    CHECK(e.nexec == 2, "two ranges (stub and rodata excluded, init+text merged)");
    if (e.nexec == 2) {
        CHECK(e.exec_lo[0] == 0x08804018 && e.exec_hi[0] == 0x08804060, ".init+.text range");
        CHECK(e.exec_lo[1] == 0x08B2FE34 && e.exec_hi[1] == 0x08B2FE3C, ".fini range");
    }
    CHECK(e.stub_size == 0x10, "stub section still recorded separately");

    /* Adjacent executable sections merge into one range. */
    {
        uint8_t b2[2048];
        memset(b2, 0, sizeof b2);
        memcpy(b2, "\x7f" "ELF", 4);
        b2[4] = 1; b2[5] = 1;
        put16(b2 + 16, 2);
        put32(b2 + 32, 0x200);
        put16(b2 + 42, 32);
        put16(b2 + 46, 40);
        memcpy(b2 + 0x180, "\0.x\0.shstrtab", 15);
        uint8_t *t = b2 + 0x200;
        shdr(t + 40 * 1, 1, 1, 0x6, 0x08804000, 0x100, 0x40);
        shdr(t + 40 * 2, 1, 1, 0x6, 0x08804040, 0x140, 0x20);
        shdr(t + 40 * 3, 4, 3, 0x0, 0, 0x180, 15);
        put16(b2 + 48, 4);
        put16(b2 + 50, 3);
        elf_info m;
        CHECK(elf_parse(b2, sizeof b2, &m) == 0, "parse adjacent");
        CHECK(m.nexec == 1 && m.exec_overflow == 0, "adjacent sections merge");
        if (m.nexec == 1)
            CHECK(m.exec_lo[0] == 0x08804000 && m.exec_hi[0] == 0x08804060, "merged range");

        /* 17 non-adjacent executable sections overflow the table. */
        memset(b2 + 0x200, 0, 40 * 20);
        for (int i = 1; i <= 17; i++)
            shdr(t + 40 * i, 1, 1, 0x6, 0x08800000 + i * 0x1000, 0x100, 0x10);
        shdr(t + 40 * 18, 4, 3, 0x0, 0, 0x180, 15);
        put16(b2 + 48, 19);
        put16(b2 + 50, 18);
        CHECK(elf_parse(b2, sizeof b2, &m) == 0, "parse overflow");
        CHECK(m.exec_overflow == 1, "overflow flagged");
        CHECK(m.nexec == 0, "overflow leaves no partial list");
    }

    if (failures) { printf("%d check(s) failed\n", failures); return 1; }
    printf("all elf_exec checks passed\n");
    return 0;
}
