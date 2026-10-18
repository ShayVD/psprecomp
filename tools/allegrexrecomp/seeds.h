/* Extra discovery seeds from a file: one address per line ("0x08A8BE6C"),
 * '#' to end of line is a comment, blank lines are ignored. khbbs generates
 * this from its bbs.toml; any project can write one by hand. */
#ifndef ALLEGREX_SEEDS_H
#define ALLEGREX_SEEDS_H

#include <stdint.h>

/* On success returns 0 and hands back a malloc'd array (free it) and its
 * length. On a missing file or a malformed line, prints the reason with the
 * line number, sets *out to NULL and returns -1. */
int a_read_seed_file(const char *path, uint32_t **out, int *n);

#endif
