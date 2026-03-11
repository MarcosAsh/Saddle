#ifndef SADDLE_SURFACES_H
#define SADDLE_SURFACES_H

/*
 * Loss surface evaluators for Saddle.
 *
 * Each function takes (x, y) and returns the scalar loss value.
 * Grid versions fill a pre-allocated buffer of size rows*cols,
 * evaluating the surface over a linearly spaced grid.
 */

/* Rosenbrock: f(x,y) = (a - x)^2 + b*(y - x^2)^2, with a=1, b=100 */
double rosenbrock(double x, double y);

/* Beale: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2 */
double beale(double x, double y);

/* Himmelblau: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 */
double himmelblau(double x, double y);

/* Bowl (sum of squares): f(x,y) = x^2 + y^2 */
double bowl(double x, double y);

/*
 * Grid evaluation: evaluates a surface over an evenly spaced grid.
 *
 * out     - pre-allocated buffer of size rows * cols (row-major)
 * x_min, x_max - x-axis bounds
 * y_min, y_max - y-axis bounds
 * rows, cols   - grid dimensions
 * surface_id   - 0=rosenbrock, 1=beale, 2=himmelblau, 3=bowl
 */
void eval_grid(double *out,
               double x_min, double x_max,
               double y_min, double y_max,
               int rows, int cols,
               int surface_id);

#endif
