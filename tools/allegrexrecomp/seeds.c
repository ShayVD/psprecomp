/* See seeds.h. */
#include "seeds.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Parse one token: 0x/0X + hex digits, or decimal digits. No sign, nothing
 * after it. A decimal token with a leading zero and more digits ("08804040")
 * is almost certainly hex missing its 0x and is refused as ambiguous
 * (*ambiguous set). Returns 0 and the value, or -1. */
static int parse_addr(const char *p, uint32_t *out, int *ambiguous) {
    int base = 10;
    *ambiguous = 0;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { base = 16; p += 2; }
    if (!*p) return -1;
    if (base == 10 && p[0] == '0' && p[1]) { *ambiguous = 1; return -1; }
    unsigned long long v = 0;
    for (; *p; p++) {
        int d;
        if (*p >= '0' && *p <= '9') d = *p - '0';
        else if (base == 16 && *p >= 'a' && *p <= 'f') d = *p - 'a' + 10;
        else if (base == 16 && *p >= 'A' && *p <= 'F') d = *p - 'A' + 10;
        else return -1;
        v = v * (unsigned)base + (unsigned)d;
        if (v > 0xFFFFFFFFull) return -1;
    }
    *out = (uint32_t)v;
    return 0;
}

int a_read_seed_file(const char *path, uint32_t **out, int *n) {
    *out = NULL;
    *n = 0;
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read seeds file %s\n", path); return -1; }

    uint32_t *v = NULL;
    int cnt = 0, cap = 0, lineno = 0, rc = 0;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        lineno++;
        /* A full buffer with no newline means the line may not have fit.
         * Without this check the rest of it would be read as a line of its
         * own. Peek at what follows: end of file or the newline itself means
         * the line was exactly full, which is fine. A short read without a
         * newline is the last line (or a read error, checked below). */
        if (!strchr(line, '\n') && strlen(line) == sizeof line - 1) {
            int c = fgetc(f);
            if (c != EOF && c != '\n') {
                fprintf(stderr, "%s:%d: line too long\n", path, lineno);
                rc = -1;
                break;
            }
        }
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        char *end = p + strlen(p);
        while (end > p && isspace((unsigned char)end[-1])) end--;
        *end = '\0';
        if (*p == '\0') continue;

        uint32_t val;
        int ambiguous;
        if (parse_addr(p, &val, &ambiguous) != 0) {
            if (ambiguous)
                fprintf(stderr, "%s:%d: ambiguous leading zero (use 0x for hex): '%s'\n",
                        path, lineno, p);
            else
                fprintf(stderr, "%s:%d: not an address: \"%s\"\n", path, lineno, p);
            rc = -1;
            break;
        }
        if (cnt == cap) {
            int ncap = cap ? cap * 2 : 64;
            uint32_t *nv = (uint32_t *)realloc(v, (size_t)ncap * sizeof *nv);
            if (!nv) { fprintf(stderr, "%s: out of memory\n", path); rc = -1; break; }
            v = nv;
            cap = ncap;
        }
        v[cnt++] = val;
    }
    if (!rc && ferror(f)) { fprintf(stderr, "%s: read error\n", path); rc = -1; }
    fclose(f);
    if (rc) { free(v); return -1; }
    *out = v;
    *n = cnt;
    return 0;
}
