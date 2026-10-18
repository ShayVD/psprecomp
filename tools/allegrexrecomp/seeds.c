/* See seeds.h. */
#include "seeds.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int a_read_seed_file(const char *path, uint32_t **out, int *n) {
    *out = NULL;
    *n = 0;
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read seeds file %s\n", path); return -1; }

    uint32_t *v = NULL;
    int cnt = 0, cap = 0, lineno = 0, rc = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        lineno++;
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;

        char *end = NULL;
        errno = 0;
        unsigned long long val = isdigit((unsigned char)*p) ? strtoull(p, &end, 0) : 0;
        if (end) while (isspace((unsigned char)*end)) end++;
        if (!end || end == p || *end != '\0' || errno || val > 0xFFFFFFFFull) {
            fprintf(stderr, "%s:%d: not an address\n", path, lineno);
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
        v[cnt++] = (uint32_t)val;
    }
    fclose(f);
    if (rc) { free(v); return -1; }
    *out = v;
    *n = cnt;
    return 0;
}
