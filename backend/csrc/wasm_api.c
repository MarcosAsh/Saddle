/*
 * WASM entry points for Saddle.
 *
 * Thin wrappers around the existing C functions, annotated with
 * EMSCRIPTEN_KEEPALIVE so they survive dead-code elimination.
 * The native build ignores this file entirely.
 */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include "surfaces.h"
#include "adam.h"

EMSCRIPTEN_KEEPALIVE
void wasm_eval_grid(double *out,
                    double x_min, double x_max,
                    double y_min, double y_max,
                    int rows, int cols,
                    int surface_id)
{
    eval_grid(out, x_min, x_max, y_min, y_max, rows, cols, surface_id);
}

EMSCRIPTEN_KEEPALIVE
void wasm_adam_optimise(double x0, double y0,
                        int num_steps,
                        double lr, double beta1, double beta2, double eps,
                        int surface_id,
                        double *trajectory_x,
                        double *trajectory_y,
                        double *trajectory_loss)
{
    adam_optimise(x0, y0, num_steps, lr, beta1, beta2, eps, surface_id,
                 trajectory_x, trajectory_y, trajectory_loss);
}

EMSCRIPTEN_KEEPALIVE
double wasm_rosenbrock(double x, double y) { return rosenbrock(x, y); }

EMSCRIPTEN_KEEPALIVE
double wasm_beale(double x, double y) { return beale(x, y); }

EMSCRIPTEN_KEEPALIVE
double wasm_himmelblau(double x, double y) { return himmelblau(x, y); }

EMSCRIPTEN_KEEPALIVE
double wasm_bowl(double x, double y) { return bowl(x, y); }

EMSCRIPTEN_KEEPALIVE
double wasm_monkey_saddle(double x, double y) { return monkey_saddle(x, y); }
