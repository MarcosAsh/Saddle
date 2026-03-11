#include "adam.h"
#include "surfaces.h"
#include <math.h>

/*
 * Single Adam update step -- the hot inner loop.
 *
 * Everything is a tight scalar loop over n elements. No branching inside
 * the loop, no memory allocation, no function calls (sqrt is typically
 * inlined by the compiler at -O2).
 *
 * Bias correction factors are computed once outside the loop since they
 * only depend on the step count, not the element index.
 */
void adam_step(double *params,
               double *m,
               double *v,
               const double *grads,
               int n,
               int step,
               double lr,
               double beta1,
               double beta2,
               double eps) {

    /* Bias correction denominators, computed once per step. */
    double bc1 = 1.0 - pow(beta1, (double)step);
    double bc2 = 1.0 - pow(beta2, (double)step);

    for (int i = 0; i < n; i++) {
        double g = grads[i];

        /* Update biased first and second moment estimates. */
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

        /* Bias-corrected estimates. */
        double m_hat = m[i] / bc1;
        double v_hat = v[i] / bc2;

        /* Parameter update. */
        params[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

/*
 * Numerical gradient via central differences.
 *
 * For a 2D surface f(x, y), we compute:
 *   df/dx ~ (f(x+h, y) - f(x-h, y)) / (2h)
 *   df/dy ~ (f(x, y+h) - f(x, y-h)) / (2h)
 *
 * h = 1e-7 is a reasonable choice for double precision: small enough
 * for accuracy, large enough to avoid catastrophic cancellation.
 */
static void numerical_grad(double (*surface)(double, double),
                           double x, double y,
                           double *gx, double *gy) {
    const double h = 1e-7;
    *gx = (surface(x + h, y) - surface(x - h, y)) / (2.0 * h);
    *gy = (surface(x, y + h) - surface(x, y - h)) / (2.0 * h);
}

/*
 * Full Adam optimisation loop in C.
 *
 * Runs num_steps of Adam on a 2D loss surface, recording the full
 * trajectory. The gradient is computed numerically since we don't
 * have autodiff in C -- this is intentional; the point of the C
 * implementation is to benchmark the update arithmetic, not the
 * gradient computation.
 */
void adam_optimise(double x0, double y0,
                   int num_steps,
                   double lr, double beta1, double beta2, double eps,
                   int surface_id,
                   double *trajectory_x,
                   double *trajectory_y,
                   double *trajectory_loss) {

    /* Select surface function. */
    double (*surface)(double, double);
    switch (surface_id) {
        case 0: surface = rosenbrock; break;
        case 1: surface = beale;      break;
        case 2: surface = himmelblau; break;
        case 3: surface = bowl;       break;
        default: surface = bowl;      break;
    }

    /* Working state: params and Adam moments. */
    double params[2] = {x0, y0};
    double m[2] = {0.0, 0.0};
    double v[2] = {0.0, 0.0};
    double grads[2];

    /* Record starting position. */
    trajectory_x[0] = params[0];
    trajectory_y[0] = params[1];
    trajectory_loss[0] = surface(params[0], params[1]);

    for (int step = 1; step <= num_steps; step++) {
        /* Compute gradient numerically. */
        numerical_grad(surface, params[0], params[1], &grads[0], &grads[1]);

        /* Adam update. */
        adam_step(params, m, v, grads, 2, step, lr, beta1, beta2, eps);

        /* Record trajectory. */
        trajectory_x[step] = params[0];
        trajectory_y[step] = params[1];
        trajectory_loss[step] = surface(params[0], params[1]);
    }
}
