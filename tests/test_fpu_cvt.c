/* psp_cvt_w: the FPU's float-to-int conversions. Synthetic values only. */
#include "psprecomp/recomp_rt.h"

#include <math.h>
#include <stdio.h>

static int failures;
static void eq(double x, int rm, int32_t want, const char *what) {
    int32_t got = psp_cvt_w(x, rm);
    if (got != want) { printf("FAIL %s: cvt(%.9g, %d) = %d, want %d\n", what, x, rm, got, want); failures++; }
}

int main(void) {
    eq(2.5, 0, 2, "round half to even (down)");
    eq(3.5, 0, 4, "round half to even (up)");
    eq(-2.5, 0, -2, "round -2.5");
    eq(-0.5, 0, 0, "round -0.5");
    eq(0.49999997f, 0, 0, "round just below a half");
    eq(2.7, 1, 2, "trunc");
    eq(-2.7, 1, -2, "trunc negative");
    eq(2.1, 2, 3, "ceil");
    eq(-2.7, 2, -2, "ceil negative");
    eq(-2.1, 3, -3, "floor negative");
    eq(2147483520.0f, 1, 2147483520, "largest float below 2^31");
    eq(2147483648.0f, 1, INT32_MAX, "2^31 saturates");
    eq(-2147483648.0f, 1, INT32_MIN, "-2^31 is exact");
    eq(1e10, 0, INT32_MAX, "big positive");
    eq(-1e10, 0, INT32_MIN, "big negative");
    eq(2147483647.5, 1, INT32_MAX, "trunc just inside 2^31");
    eq(-2147483648.5, 1, INT32_MIN, "trunc just inside -2^31-1");
    eq(2147483648.0, 1, INT32_MAX, "trunc 2^31 (double)");
    eq(-2147483649.0, 1, INT32_MIN, "trunc -2^31-1 (double)");
    eq(-2147483648.0, 1, INT32_MIN, "trunc -2^31 (double)");
    eq(-0.9, 1, 0, "trunc -0.9");
    eq(NAN, 0, INT32_MAX, "NaN");
    eq(INFINITY, 3, INT32_MAX, "+inf");
    eq(-INFINITY, 2, INT32_MIN, "-inf");
    printf("%s\n", failures ? "test_fpu_cvt: FAILED" : "test_fpu_cvt: ok");
    return failures ? 1 : 0;
}
