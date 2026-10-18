/* Function discovery — recursive descent over Allegrex code.
 *
 * The emitter needs to know where functions begin and end. Nothing in a
 * stripped PRX says so directly, so we recover it by following control flow
 * from a set of seeds until nothing new is reachable.
 *
 * The signals we lean on, in order of reliability:
 *
 *   1. `jal <target>` — a direct call. Its target is a function entry, full
 *      stop. This is by far the strongest signal and most of a module is
 *      discovered through it.
 *   2. `jr $ra` — a return, and therefore a function end. The decoder
 *      distinguishes this from `jr $rN` precisely so discovery can rely on it;
 *      conflating them either truncates every function at its first jump table
 *      or never closes one at all.
 *   3. The module's entry point and its exported function addresses, as seeds.
 *
 * What this deliberately does NOT do is scan linearly for function prologues.
 * That finds more, but it also finds data that happens to look like a
 * prologue, and a wrong function boundary produces C that compiles and
 * silently misbehaves. Recursive descent under-approximates, which is the
 * right direction to be wrong in: everything it reports is genuinely reachable
 * code, and what it misses shows up as unreached bytes we can go and account
 * for.
 */
#ifndef ALLEGREX_ANALYZE_H
#define ALLEGREX_ANALYZE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t addr;          /* function entry */
    /* The address range actually touched. `start` is not always `addr`: a
     * block can sit below the entry, and the emitter must iterate over
     * everything the function owns rather than assuming it all lies forward. */
    uint32_t start;
    uint32_t end;           /* one past the highest instruction reached */
    uint32_t insns;         /* instructions actually visited */
    unsigned has_return  : 1;   /* reaches a `jr $ra` */
    unsigned has_indirect: 1;   /* contains an unresolved computed jump */
    unsigned has_vfpu    : 1;   /* touches the VFPU */
} a_func;

/* A switch table recovered at a `jr $rN` site.
 *
 * The code indexes a table at `table - 4 * lo`, but its bound only lets
 * indexes lo .. lo + count - 1 through, so only those entries are recorded:
 * `table` is the address of the first reachable one. Their targets, in
 * table order (duplicates kept), are an->jt_targets[first .. first + count).
 * The targets are code in .text; where the table itself lives is not
 * checked. */
typedef struct {
    uint32_t site;      /* the jr */
    uint32_t table;     /* the first reachable entry */
    uint32_t lo;        /* its index in the table the code indexes */
    uint32_t first;
    uint32_t count;     /* reachable entries, from the sltiu/slti bound and
                         * any lower bound, corrected for offsets */
} a_jtable;

typedef struct {
    /* Input */
    const uint8_t *code;
    uint32_t base;              /* virtual address of code[0] */
    uint32_t size;
    uint32_t stub_addr, stub_size;   /* import thunks; calls here are HLE */

    /* Also harvest call targets by linearly scanning the code for `jal`.
     *
     * Necessary on PSP, not optional. A module's `module_start` typically does
     * little more than create a thread, passing the real entry point as a
     * *pointer* to sceKernelCreateThread. That pointer is built with
     * lui/addiu, so the bulk of a game is never the target of a `jal` from
     * anything reachable — pure recursive descent from the entry point finds
     * the init stub and stops.
     *
     * `jal` carries an absolute target, so a linear scan recovers those
     * targets without needing to reach the instruction by control flow. The
     * cost is that data decoding as `jal` yields bogus seeds; over a real
     * `.text` (99.8% valid instructions) that is rare, and a bogus seed
     * produces a function that fails to reach a `jr $ra`, which the report
     * counts separately. */
    int scan_calls;

    /* The linear `jal` harvest runs over this sub-range only, normally `.text`.
     *
     * The walkable extent (`base`/`size` above) covers the whole loaded image,
     * because a module can and does place executable code outside `.text` --
     * C++ static initialisers in particular. But scanning all of `.data` for
     * `jal` patterns would manufacture thousands of false seeds out of data
     * that happens to decode. Walking widely and scanning narrowly gets both:
     * control flow may lead anywhere, while the brute-force harvest stays
     * where instructions are known to live. */
    uint32_t scan_base;
    uint32_t scan_size;

    /* The whole loaded module image, not just .text. Jump tables live in
     * .rodata or .data, so resolving a computed jump means reading outside the
     * code extent. Optional: without it, tables are simply not resolved. */
    const uint8_t *image;
    uint32_t       image_base;
    uint32_t       image_size;

    /* Output */
    a_func   *funcs;
    int       nfuncs;
    uint32_t *imports;          /* distinct import stubs called */
    int       nimports;
    uint32_t *indirects;        /* sites with an unresolved computed jump */
    int       nindirects;

    /* Jump tables recovered from the `jr $rN` sites. They are resolved as
     * the walk reaches each jr, so their targets are blocks of the function
     * containing the switch (or functions of their own, if already entries),
     * and checked against every way into their code once the walk is done. */
    int       ntables;          /* tables resolved (== njtables) */
    int       ntable_targets;   /* total entries across them (== njt_targets) */
    a_jtable *jtables;          /* sorted by site */
    int       njtables;
    uint32_t *jt_targets;
    int       njt_targets;

    /* Statistics over *discovered* code only — the number that matters, as
     * opposed to coverage over a whole segment that is mostly data. */
    uint64_t insns;
    uint64_t vfpu;
    uint64_t invalid;
    uint32_t bytes_reached;
    /* Executable extent, when the container reports one. Distinct from size,
     * which spans .data and .bss too. */
    uint32_t text_size;

    /* Which function owns each word: the owning function's entry address, or
     * A_NO_OWNER. The emitter needs this because a function's instructions are
     * not necessarily contiguous — blocks get laid out apart from each other,
     * and the gaps belong either to another function or to data. Emitting the
     * whole [addr, end) range instead would pull in a neighbour's code. */
    uint32_t *owner;
    uint32_t  nwords;
} a_analysis;

#define A_NO_OWNER 0xFFFFFFFFu

/* Run discovery. `seeds` are function entry addresses to start from (the
 * module entry point and its exports). Returns 0 on success.
 * Call a_analysis_free() when done. */
int a_discover(a_analysis *an, const uint32_t *seeds, int nseeds);

/* Scan a data region for words that look like code addresses.
 *
 * This is the fallback for a **statically linked** module (ET_EXEC). A
 * relocatable PRX lists its function pointers in the relocation tables, which
 * is enumeration — the linker recorded them because they *are* addresses. A
 * static executable has those addresses baked in absolutely and its relocation
 * sections are empty, so there is nothing to enumerate and the only option is
 * to recognise them by shape.
 *
 * That makes this a **heuristic, not a fact**, and it is kept separate and
 * named accordingly so nobody later mistakes one for the other. Three filters
 * keep the false-positive rate down: the value must land inside the code
 * extent, be instruction-aligned, and point at something that actually decodes
 * as a valid instruction. A survivor that is nonetheless data shows up as a
 * function that never reaches `jr $ra`, which the report counts.
 *
 * `region` is the raw bytes to scan; `out` receives candidate addresses.
 * Returns the number found (may exceed `max`). */
int a_scan_data_pointers(const a_analysis *an,
                         const uint8_t *region, uint32_t region_len,
                         uint32_t *out, int max);

/* A pointer found by the data scan, and where it was read: the address of
 * the data word. */
typedef struct { uint32_t value, source; } a_data_seed;

/* a_scan_data_pointers, also reporting each pointer's source address.
 * `region_addr` is the virtual address of region[0]. Returns the number
 * found (may exceed `max`). */
int a_scan_data_seeds(const a_analysis *an, const uint8_t *region, uint32_t region_len,
                      uint32_t region_addr, a_data_seed *out, int max);

typedef struct {
    int dropped;    /* data seeds (words) left out of the final discovery */
    int restored;   /* dropped once, put back because their code was lost */
    int passes;     /* discovery runs */
} a_prune_stats;

/* Discovery with the data-scan seeds `ds` added to `seeds`, minus the ones
 * that are only jump-table words.
 *
 * The data scan cannot tell a function pointer from a switch table's entry:
 * both are code addresses in .rodata/.data. Seeding table entries makes
 * every case block a function of its own. So discovery runs once with all
 * seeds; then every data seed whose value was read only from the reachable
 * entries of recovered tables -- no data word outside a table holds it, and
 * it is not in `seeds` -- is dropped, and discovery runs again. A case that
 * is also a call, tail call or cross-function branch target still becomes
 * an entry through the walk. A dropped seed whose code the new run does not
 * reach as an entry or as a block of a function with a table listing it is
 * put back for good, and discovery runs again; new tables found by a run
 * can justify further drops, for a few passes. `an` holds the last run. */
int a_discover_pruned(a_analysis *an, const uint32_t *seeds, int nseeds,
                      const a_data_seed *ds, int nds, a_prune_stats *ps);

/* Scan the code (the scan range, normally .text) for addresses built in
 * registers: `lui rX, hi` followed within a few instructions by
 * `addiu rY, rX, lo` or `ori rY, rX, lo`, with rX not rewritten in between.
 *
 * This is how a statically linked module passes a function pointer: crt0
 * hands its main-thread entry to sceKernelCreateThread this way, and so do
 * callback and handler registrations. No `jal` and no data word names those
 * functions, so neither the call harvest nor the data scan sees them.
 *
 * Like a_scan_data_pointers this is a heuristic. A target is kept only if it
 * lies inside the scan range, is word-aligned, is not an import stub and
 * decodes as a valid instruction. `out` receives candidates (duplicates
 * possible); returns the number found, which may exceed `max`. */
int a_scan_address_pairs(const a_analysis *an, uint32_t *out, int max);

void a_analysis_free(a_analysis *an);

/* Is `addr` inside the discovered code extent? */
int a_in_range(const a_analysis *an, uint32_t addr);

/* The table recovered for the jr at `site`, or NULL. */
const a_jtable *a_find_jtable(const a_analysis *an, uint32_t site);

#ifdef __cplusplus
}
#endif

#endif /* ALLEGREX_ANALYZE_H */
