/* The seeds-file reader. Synthetic files only. */
#include "seeds.h"

#include <stdio.h>
#include <stdlib.h>

static int failures;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s\n", m); failures++; } } while (0)

static const char *write_tmp(const char *name, const char *text) {
    FILE *f = fopen(name, "w");
    if (!f) return NULL;
    fputs(text, f);
    fclose(f);
    return name;
}

int main(void) {
    uint32_t *v = NULL;
    int n = -1;

    const char *ok = write_tmp("t_seeds_ok.txt",
        "# generated\n"
        "0x08A8BE6C\n"
        "\n"
        "   0x08804040   # with a comment\n"
        "0x0880A5DC\r\n");
    CHECK(a_read_seed_file(ok, &v, &n) == 0, "valid file parses");
    CHECK(n == 3, "three seeds");
    CHECK(v && v[0] == 0x08A8BE6Cu && v[1] == 0x08804040u && v[2] == 0x0880A5DCu, "values in order");
    free(v);

    v = NULL;
    const char *bad = write_tmp("t_seeds_bad.txt", "0x08804040\nnot-an-address\n");
    CHECK(a_read_seed_file(bad, &v, &n) != 0, "garbage line rejected");
    CHECK(v == NULL, "nothing returned on error");

    const char *neg = write_tmp("t_seeds_neg.txt", "-1\n");
    CHECK(a_read_seed_file(neg, &v, &n) != 0, "negative rejected");
    const char *big = write_tmp("t_seeds_big.txt", "0x100000000\n");
    CHECK(a_read_seed_file(big, &v, &n) != 0, "over 32 bits rejected");
    const char *trail = write_tmp("t_seeds_trail.txt", "0x08804040 0x08804044\n");
    CHECK(a_read_seed_file(trail, &v, &n) != 0, "two values on a line rejected");

    const char *empty = write_tmp("t_seeds_empty.txt", "# nothing\n\n");
    CHECK(a_read_seed_file(empty, &v, &n) == 0 && n == 0, "empty file is fine");
    free(v);

    CHECK(a_read_seed_file("t_seeds_does_not_exist.txt", &v, &n) != 0, "missing file is an error");

    printf("%s\n", failures ? "test_seeds: FAILED" : "test_seeds: ok");
    return failures ? 1 : 0;
}
