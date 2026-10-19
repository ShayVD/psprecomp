# Tests are self-contained: no game data, no ROMs, no disc images. Every input
# is either a hand-written instruction encoding or a synthetic buffer built in
# the test itself.

add_executable(test_decode test_decode.c)
target_link_libraries(test_decode PRIVATE allegrex_core)
add_test(NAME decode COMMAND test_decode)

add_executable(test_runtime test_runtime.c)
target_link_libraries(test_runtime PRIVATE psprecomp)
add_test(NAME runtime COMMAND test_runtime)

add_executable(test_container test_container.c)
target_link_libraries(test_container PRIVATE allegrex_core)
add_test(NAME container COMMAND test_container)

add_executable(test_elf_exec test_elf_exec.c)
target_link_libraries(test_elf_exec PRIVATE allegrex_core)
add_test(NAME elf_exec COMMAND test_elf_exec)

# Published standard vectors only (FIPS-197, SP 800-38A, RFC 4493) — no PSP
# key material, no game data.
add_executable(test_crypto test_crypto.c)
target_link_libraries(test_crypto PRIVATE allegrex_core)
add_test(NAME crypto COMMAND test_crypto)

add_executable(test_kirk test_kirk.c)
target_link_libraries(test_kirk PRIVATE allegrex_core)
add_test(NAME kirk COMMAND test_kirk)

add_executable(test_emit test_emit.c)
target_link_libraries(test_emit PRIVATE allegrex_core)
add_test(NAME emit COMMAND test_emit)

add_executable(test_hle test_hle.c)
target_link_libraries(test_hle PRIVATE psprecomp allegrex_core)
add_test(NAME hle COMMAND test_hle)

add_executable(test_vfpu test_vfpu.c)
target_link_libraries(test_vfpu PRIVATE psprecomp)
add_test(NAME vfpu COMMAND test_vfpu)

add_executable(test_ctors test_ctors.c)
target_link_libraries(test_ctors PRIVATE psprecomp)
add_test(NAME ctors COMMAND test_ctors)

add_executable(test_raster test_raster.c)
target_link_libraries(test_raster PRIVATE psprecomp)
add_test(NAME raster COMMAND test_raster)

add_executable(test_emit_vfim test_emit_vfim.c)
target_link_libraries(test_emit_vfim PRIVATE allegrex_core)
add_test(NAME emit_vfim COMMAND test_emit_vfim)

add_executable(test_emit_split test_emit_split.c)
target_link_libraries(test_emit_split PRIVATE allegrex_core)
add_test(NAME emit_split COMMAND test_emit_split)

add_executable(test_hle_trace test_hle_trace.c)
target_link_libraries(test_hle_trace PRIVATE psprecomp)
add_test(NAME hle_trace COMMAND test_hle_trace)

add_executable(test_analyze_pairs test_analyze_pairs.c)
target_link_libraries(test_analyze_pairs PRIVATE allegrex_core)
add_test(NAME analyze_pairs COMMAND test_analyze_pairs)

add_executable(test_seeds test_seeds.c)
target_link_libraries(test_seeds PRIVATE allegrex_core)
add_test(NAME seeds COMMAND test_seeds)

add_executable(test_switch test_switch.c)
target_link_libraries(test_switch PRIVATE allegrex_core)
add_test(NAME switch COMMAND test_switch)

# The switch test leaves the C it generated next to itself; compile it, so an
# undefined label or a duplicate case value fails here and not in a game build.
set_tests_properties(switch PROPERTIES FIXTURES_SETUP switch_gen)
foreach(_g IN ITEMS t_switch t_switch_entry t_switch_back t_switch_slot
                    t_switch_slot_bh t_switch_slot_lh t_switch_slot_jl
                    t_switch_slot_jj t_switch_slot_ja t_switch_slot_jr
                    t_always_b t_always_beq t_always_bgez t_always_beql t_runs_off
                    t_slot_target t_slot_target_o t_strlen
                    t_always_blez t_always_bgezl t_always_blezl t_alias_br t_alias_jal t_j_entry)
    add_test(NAME switch_cc_${_g}
             COMMAND ${CMAKE_C_COMPILER} -std=c11 -fsyntax-only
                     -I${CMAKE_SOURCE_DIR}/include ${_g}_funcs.c)
    set_tests_properties(switch_cc_${_g} PROPERTIES FIXTURES_REQUIRED switch_gen)
endforeach()

add_executable(test_prune test_prune.c)
target_link_libraries(test_prune PRIVATE allegrex_core)
add_test(NAME prune COMMAND test_prune)

add_executable(test_decode_vfpu test_decode_vfpu.c)
target_link_libraries(test_decode_vfpu PRIVATE allegrex_core)
add_test(NAME decode_vfpu COMMAND test_decode_vfpu)

add_executable(test_emit_vfpu test_emit_vfpu.c)
target_link_libraries(test_emit_vfpu PRIVATE allegrex_core)
add_test(NAME emit_vfpu COMMAND test_emit_vfpu)

add_executable(test_vfpu_prefix test_vfpu_prefix.c)
target_link_libraries(test_vfpu_prefix PRIVATE psprecomp)
add_test(NAME vfpu_prefix COMMAND test_vfpu_prefix)

add_executable(test_vfpu_ops test_vfpu_ops.c)
target_link_libraries(test_vfpu_ops PRIVATE psprecomp)
add_test(NAME vfpu_ops COMMAND test_vfpu_ops)

add_executable(test_fpu_cvt test_fpu_cvt.c)
target_link_libraries(test_fpu_cvt PRIVATE psprecomp)
add_test(NAME fpu_cvt COMMAND test_fpu_cvt)
