#ifndef SADDLE_ADAM_H
#define SADDLE_ADAM_H

/*
 * Pure C implementation of the Adam optimiser update step.
 *
 * Operates on flat arrays of doubles. No dependencies beyond libc math.
 * Designed to be called from Python via ctypes for benchmarking against
 * JIT-compiled JAX.
 */

/*
 * Single Adam update step.
 *
 * params, m, v are updated in place.
 * grads is read-only.
 * step is the 1-indexed iteration count (used for bias correction).
 * n is the number of parameters.
 *
 * The arithmetic per element:
 *   m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
 *   v[i] = beta2 * v[i] + (1 - beta2) * grads[i]^2
 *   m_hat = m[i] / (1 - beta1^step)
 *   v_hat = v[i] / (1 - beta2^step)
 *   params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
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
               double eps);

/*
 * Run multiple Adam steps on a loss surface.
 *
 * This is the full optimisation loop in C: evaluate gradients numerically
 * (central differences) and apply Adam updates for num_steps iterations.
 *
 * trajectory_x, trajectory_y, trajectory_loss are pre-allocated arrays
 * of size (num_steps + 1) that get filled with the full path, starting
 * from the initial position at index 0.
 *
 * surface_id: 0=rosenbrock, 1=beale, 2=himmelblau, 3=bowl
 */
void adam_optimise(double x0, double y0,
                   int num_steps,
                   double lr, double beta1, double beta2, double eps,
                   int surface_id,
                   double *trajectory_x,
                   double *trajectory_y,
                   double *trajectory_loss);

#endif
