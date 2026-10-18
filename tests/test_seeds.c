/* The seeds-file reader. Synthetic files only. */
#include "seeds.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s\n", m); failures++; } } while (0)

static const char *tmpfiles[32];
static int ntmp;

/* Writes `text` to a scratch file and remembers it for cleanup. */
static const char *write_tmp(const char *name, const char *text) {
    FILE *f = fopen(name, "wb");
    CHECK(f != NULL, "can create scratch file");
    if (!f) return NULL;
    fputs(text, f);
    fclose(f);
    if (ntmp < 32) tmpfiles[ntmp++] = name;
    return name;
}

/* Expect failure: nothing handed back, count zero. */
static void expect_fail(const char *file, const char *msg) {
    uint32_t *v = (uint32_t *)0x1;
    int n = -1;
    int rc = file ? a_read_seed_file(file, &v, &n) : 0;
    CHECK(file && rc != 0, msg);
    CHECK(v == NULL && n == 0, "error leaves *out NULL and *n zero");
}

int main(void) {
    uint32_t *v = NULL;
    int n = -1;

    const char *ok = write_tmp("t_seeds_ok.txt",
        "# generated\n"
        "0x08A8BE6C\n"
        "\n"
        "   0x08804040   # with a comment\n"
        "0X0880A5DC\r\n"
        "134217728\n"
        "0\n"
        "0x08804044");                      /* last line has no newline */
    CHECK(ok && a_read_seed_file(ok, &v, &n) == 0, "valid file parses");
    CHECK(n == 6, "six seeds");
    if (v && n == 6) {
        CHECK(v[0] == 0x08A8BE6Cu && v[1] == 0x08804040u && v[2] == 0x0880A5DCu, "values in order");
        CHECK(v[3] == 134217728u, "decimal accepted");
        CHECK(v[4] == 0u, "a lone 0 is valid");
        CHECK(v[5] == 0x08804044u, "final line without newline is read");
    }
    free(v);

    expect_fail(write_tmp("t_seeds_bad.txt", "0x08804040\nnot-an-address\n"), "garbage line rejected");
    expect_fail(write_tmp("t_seeds_neg.txt", "-1\n"), "negative rejected");
    expect_fail(write_tmp("t_seeds_big.txt", "0x100000000\n"), "over 32 bits rejected");
    expect_fail(write_tmp("t_seeds_trail.txt", "0x08804040 0x08804044\n"), "two values on a line rejected");
    expect_fail(write_tmp("t_seeds_oct.txt", "010\n"), "010 is ambiguous, rejected");
    expect_fail(write_tmp("t_seeds_nox.txt", "08804040\n"), "hex missing its 0x rejected");
    expect_fail(write_tmp("t_seeds_0x.txt", "0x\n"), "bare 0x rejected");

    /* Lines longer than the buffer must not be split into fake lines. */
    {
        char longline[2400];
        strcpy(longline, "# ");
        memset(longline + 2, 'x', 1100);
        strcpy(longline + 1102, " 0x08804040\n");
        expect_fail(write_tmp("t_seeds_long1.txt", longline), "long comment line rejected");

        memset(longline, ' ', 1100);
        strcpy(longline + 1100, "0x08804040\n");
        expect_fail(write_tmp("t_seeds_long2.txt", longline), "long whitespace line rejected");
    }

    /* The line limit: 1023 characters are accepted, 1024 are not, with or
     * without a trailing newline. */
    {
        static char buf[1100];
        const char *addr = "0x08804040";
        struct { int len; const char *tail; int ok; const char *name; } cases[] = {
            { 1023, "\n", 1, "1023-char line accepted" },
            { 1023, "",   1, "1023-char final line without newline accepted" },
            { 1024, "\n", 0, "1024-char line rejected" },
            { 1024, "",   0, "1024-char final line without newline rejected" },
        };
        for (int i = 0; i < 4; i++) {
            int pad = cases[i].len - (int)strlen(addr);
            memset(buf, ' ', (size_t)pad);
            strcpy(buf + pad, addr);
            strcat(buf, cases[i].tail);
            static char name[4][32];   /* write_tmp keeps the pointer */
            snprintf(name[i], sizeof name[i], "t_seeds_edge%d.txt", i);
            const char *f = write_tmp(name[i], buf);
            if (cases[i].ok) {
                uint32_t *ev = NULL;
                int en = -1;
                CHECK(f && a_read_seed_file(f, &ev, &en) == 0 && en == 1 && ev[0] == 0x08804040u,
                      cases[i].name);
                free(ev);
            } else {
                expect_fail(f, cases[i].name);
            }
        }
    }

    /* A directory is not a seeds file. */
    expect_fail(".", "directory rejected");

    v = NULL;
    const char *empty = write_tmp("t_seeds_empty.txt", "# nothing\n\n");
    CHECK(empty && a_read_seed_file(empty, &v, &n) == 0 && n == 0, "empty file is fine");
    free(v);

    expect_fail("t_seeds_does_not_exist.txt", "missing file is an error");

    for (int i = 0; i < ntmp; i++) remove(tmpfiles[i]);

    printf("%s\n", failures ? "test_seeds: FAILED" : "test_seeds: ok");
    return failures ? 1 : 0;
}
