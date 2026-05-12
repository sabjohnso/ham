# Worked examples

Each example is a complete HAM solve of a problem with a known
closed-form or Taylor-recurrence reference. The library ships them
under `examples/` as both importable modules (with factory functions
`build_problem`, `solve_to`, `analyze`, `is_convergent`) and runnable
scripts:

```sh
poetry run python examples/quadratic_drag.py
poetry run python examples/logistic.py
poetry run python examples/volterra.py
```

The narrative pages below walk through each example end-to-end:
problem statement, HAM setup, deformation chain, diagnostics, the
interesting observations that fell out of the run.

## The three shipped examples

| Example | ODE | Reference | Highlight |
| --- | --- | --- | --- |
| [Quadratic drag](quadratic-drag.md) | \(v'(t) = 1 - v(t)^2\), \(v(0) = 0\) | \(v(t) = \tanh(t)\) (closed form) | At order 7 the L²-optimal \(\hbar\) is \(-\tfrac{1}{2}\), not \(-1\) — the adaptive-ℏ advantage at work. |
| [Logistic](logistic.md) | \(u'(t) = u(t)(1 - u(t))\), \(u(0) = \tfrac{1}{2}\) | \(u(t) = 1/(1 + e^{-t})\) (closed form) | First example with a *non-zero initial guess* (\(u_0 = 1/2\)); the sigmoid Taylor converges fast enough that \(\hbar = -1\) cleanly dominates. |
| [Volterra](volterra.md) | \(u'(t) = \kappa\,u\,(1 - u - \int_0^t u\,d\tau)\), \(u(0) = \alpha\) | Taylor recurrence (no closed form) | First example with an **integro-differential** \(N\); HAM polynomial degree grows by 2 per step; residual non-monotone in \(M\) (Liao's theorem only guarantees limit convergence). |

The three examples surface different regimes: a closed-form
hyperbolic-tangent benchmark, an initial-guess-with-non-zero-value
case, and an integro-differential problem requiring the Stage 9a
`Integral` branch of the nonlinear compiler.

## Deferred — Blasius

The original PLAN listed three benchmark problems; the Blasius
boundary-layer equation is the last remaining deferral:

- **Blasius boundary-layer equation** — canonical HAM benchmark with
  asymptotic BC \(f'(\infty) = 1\). The library's
  `BoundaryCondition` accepts only *point* BCs at this time;
  Blasius requires either a domain-truncation hack
  (\(f'(\eta_{\max}) = 1\) for large \(\eta_{\max}\)) or a library
  extension supporting genuine asymptotic conditions.

Stage 10 (in progress) probes whether the truncated form is
tractable with the current library; if not, the asymptotic-BC
extension becomes its own stage.
