#include "surfaces.h"

/*
 * Rosenbrock function.
 * Global minimum at (1, 1) where f = 0.
 * The classic banana-shaped valley makes this a tough test for optimisers
 * because the global minimum sits inside a long, narrow, parabolic valley.
 * Most gradient methods find the valley quickly but crawl along its floor.
 *
 * f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
 */
double rosenbrock(double x, double y) {
    double a = 1.0 - x;
    double b = y - x * x;
    return a * a + 100.0 * b * b;
}

/*
 * Beale function.
 * Global minimum at (3, 0.5) where f = 0.
 * Has sharp ridges and a flat region that punishes optimisers with
 * fixed step sizes -- momentum methods overshoot the ridges while
 * adaptive methods can navigate them.
 *
 * f(x, y) = (1.5 - x + x*y)^2
 *         + (2.25 - x + x*y^2)^2
 *         + (2.625 - x + x*y^3)^2
 */
double beale(double x, double y) {
    double y2 = y * y;
    double y3 = y2 * y;
    double t1 = 1.5 - x + x * y;
    double t2 = 2.25 - x + x * y2;
    double t3 = 2.625 - x + x * y3;
    return t1 * t1 + t2 * t2 + t3 * t3;
}

/*
 * Himmelblau function.
 * Four identical minima at:
 *   (3, 2), (-2.8051, 3.1313), (-3.7793, -3.2832), (3.5844, -1.8481)
 * All have f = 0.
 * The symmetry makes it useful for testing whether an optimiser is
 * sensitive to starting position -- different starts converge to
 * different minima.
 *
 * f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
 */
double himmelblau(double x, double y) {
    double t1 = x * x + y - 11.0;
    double t2 = x + y * y - 7.0;
    return t1 * t1 + t2 * t2;
}

/*
 * Bowl (sum of squares).
 * Global minimum at (0, 0) where f = 0.
 * Perfectly conditioned -- all optimisers converge here.
 * Useful as a sanity check and baseline.
 *
 * f(x, y) = x^2 + y^2
 */
double bowl(double x, double y) {
    return x * x + y * y;
}

/*
 * Monkey saddle with a regularising bowl term.
 *
 * The pure monkey saddle f(x,y) = x^3 - 3xy^2 has a degenerate
 * critical point at the origin where the Hessian is identically zero.
 * That makes it uninteresting for optimisation since there's no
 * curvature signal at all.
 *
 * Adding 0.5*(x^2 + y^2) creates a surface that still has a saddle-like
 * character near the origin (three valleys radiating outward at 120 degrees)
 * but with a nonzero Hessian. The gradient is still zero at the origin,
 * so it's a genuine critical point. First-order methods stall because
 * the gradient vanishes. Second-order methods can detect that the cubic
 * term creates directions of negative curvature just off-center.
 *
 * f(x, y) = x^3 - 3*x*y^2 + 0.5*(x^2 + y^2)
 */
double monkey_saddle(double x, double y) {
    return x * x * x - 3.0 * x * y * y + 0.5 * (x * x + y * y);
}

/*
 * Evaluate a loss surface over a uniform grid.
 *
 * The grid spans [x_min, x_max] x [y_min, y_max] with the given
 * number of rows (y-axis) and cols (x-axis). Output is row-major:
 * out[i * cols + j] = surface(x_j, y_i).
 *
 * surface_id selects the function:
 *   0 = rosenbrock
 *   1 = beale
 *   2 = himmelblau
 *   3 = bowl
 *   4 = monkey_saddle
 */
void eval_grid(double *out,
               double x_min, double x_max,
               double y_min, double y_max,
               int rows, int cols,
               int surface_id) {

    /* Pick the surface function pointer once, then loop. */
    double (*surface)(double, double);
    switch (surface_id) {
        case 0: surface = rosenbrock;    break;
        case 1: surface = beale;         break;
        case 2: surface = himmelblau;    break;
        case 3: surface = bowl;          break;
        case 4: surface = monkey_saddle; break;
        default: surface = bowl;         break;
    }

    /* Step sizes for the grid. Handle the degenerate case of 1 row/col. */
    double dx = (cols > 1) ? (x_max - x_min) / (cols - 1) : 0.0;
    double dy = (rows > 1) ? (y_max - y_min) / (rows - 1) : 0.0;

    for (int i = 0; i < rows; i++) {
        double y = y_min + i * dy;
        for (int j = 0; j < cols; j++) {
            double x = x_min + j * dx;
            out[i * cols + j] = surface(x, y);
        }
    }
}
