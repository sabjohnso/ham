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
poetry run python examples/blasius.py
```

The narrative pages below walk through each example end-to-end:
problem statement, HAM setup, deformation chain, diagnostics, the
interesting observations that fell out of the run.

## The four shipped examples

| Example | ODE | Reference | Highlight |
| --- | --- | --- | --- |
| [Quadratic drag](quadratic-drag.md) | \(v'(t) = 1 - v(t)^2\), \(v(0) = 0\) | \(v(t) = \tanh(t)\) (closed form) | At order 7 the L²-optimal \(\hbar\) is \(-\tfrac{1}{2}\), not \(-1\) — the adaptive-ℏ advantage at work. |
| [Logistic](logistic.md) | \(u'(t) = u(t)(1 - u(t))\), \(u(0) = \tfrac{1}{2}\) | \(u(t) = 1/(1 + e^{-t})\) (closed form) | First example with a *non-zero initial guess* (\(u_0 = 1/2\)); the sigmoid Taylor converges fast enough that \(\hbar = -1\) cleanly dominates. |
| [Volterra](volterra.md) | \(u'(t) = \kappa\,u\,(1 - u - \int_0^t u\,d\tau)\), \(u(0) = \alpha\) | Taylor recurrence (no closed form) | First example with an **integro-differential** \(N\); HAM polynomial degree grows by 2 per step; residual non-monotone in \(M\) (Liao's theorem only guarantees limit convergence). |
| [Blasius](blasius.md) | \(f'''(\eta) + \tfrac{1}{2} f(\eta) f''(\eta) = 0\), \(f(0) = f'(0) = 0\), \(f'(\eta_{\max}) = 1\) | Howarth's \(f''(0) \approx 0.4696\) | First example where \(\hbar = -1\) **diverges** and the L² residual norm has a **false plateau**; the per-problem validity gate compares \(f''(0)\) to the reference, not residual-to-zero. |

The four examples surface different regimes: a closed-form
hyperbolic-tangent benchmark, an initial-guess-with-non-zero-value
case, an integro-differential problem requiring the Stage 9a
`Integral` branch of the nonlinear compiler, and a higher-order BVP
where the convergence-control parameter must be tuned away from
\(\hbar = -1\) to recover the solution.

## What remains scope-deferred

- **Exponential-basis Blasius.** Liao's canonical
  *Beyond Perturbation* Ch. 14 treatment uses
  \(f(\eta) \approx \gamma_0 + \gamma_1 \eta + \sum c_k(\alpha) e^{-k\alpha\eta}\)
  with a second free parameter \(\alpha\). Implementing this requires
  an exponential-basis-aware `LinearOperator` and a two-parameter
  optimisation pathway, which is a Stage-11-scale extension rather
  than another worked example.
- **True asymptotic boundary conditions.** The Blasius example uses
  a truncated finite \(\eta_{\max} = 10\). A library-level
  `BoundaryCondition` extension to support genuine
  \(f'(\infty) = 1\) (and the corresponding inverter) would enable
  Liao's exponential-basis treatment cleanly.
