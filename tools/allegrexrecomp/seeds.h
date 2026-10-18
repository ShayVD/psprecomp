/* Extra discovery seeds from a file: one address per line, hex with a 0x
 * prefix ("0x08A8BE6C") or plain decimal (a leading zero does not mean octal).
 * '#' to end of line is a comment, blank lines are ignored, lines are at most
 * 255 characters. khbbs generates this from its bbs.toml; any project can
 * write one by hand. */
#ifndef ALLEGREX_SEEDS_H
#define ALLEGREX_SEEDS_H

#include <stdint.h>

/* On success returns 0 and hands back a malloc'd array (free it) and its
 * length. On a missing or unreadable file, an over-long line or a malformed
 * line, prints the reason (with the line number and the offending text where
 * there is one), sets *out to NULL and *n to 0, and returns -1. */
int a_read_seed_file(const char *path, uint32_t **out, int *n);

#endif
