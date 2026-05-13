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
poetry run python examples/blasius_exponential.py
```

The narrative pages below walk through each example end-to-end:
problem statement, HAM setup, deformation chain, diagnostics, the
interesting observations that fell out of the run.

## The five shipped examples

| Example | ODE | Reference | Highlight |
| --- | --- | --- | --- |
| [Quadratic drag](quadratic-drag.md) | \(v'(t) = 1 - v(t)^2\), \(v(0) = 0\) | \(v(t) = \tanh(t)\) (closed form) | At order 7 the L²-optimal \(\hbar\) is \(-\tfrac{1}{2}\), not \(-1\) — the adaptive-ℏ advantage at work. |
| [Logistic](logistic.md) | \(u'(t) = u(t)(1 - u(t))\), \(u(0) = \tfrac{1}{2}\) | \(u(t) = 1/(1 + e^{-t})\) (closed form) | First example with a *non-zero initial guess* (\(u_0 = 1/2\)); the sigmoid Taylor converges fast enough that \(\hbar = -1\) cleanly dominates. |
| [Volterra](volterra.md) | \(u'(t) = \kappa\,u\,(1 - u - \int_0^t u\,d\tau)\), \(u(0) = \alpha\) | Taylor recurrence (no closed form) | First example with an **integro-differential** \(N\); HAM polynomial degree grows by 2 per step; residual non-monotone in \(M\) (Liao's theorem only guarantees limit convergence). |
| [Blasius (polynomial)](blasius.md) | \(f'''(\eta) + \tfrac{1}{2} f(\eta) f''(\eta) = 0\), \(f(0) = f'(0) = 0\), \(f'(\eta_{\max}) = 1\) | Howarth's \(f''(0) \approx 0.4696\) | First example where \(\hbar = -1\) **diverges** and the L² residual norm has a **false plateau**; the per-problem validity gate compares \(f''(0)\) to the reference, not residual-to-zero. Truncated domain. |
| [Blasius (exponential)](blasius-exponential.md) | Same as above, but with **true** \(f'(\infty) = 1\) | Howarth's \(f''(0) \approx 0.4696\) | Liao's recommended basis. Asymptotic BC via `BoundaryCondition(point=sp.oo)`, custom inverter filtering growing-exp branches, **two free parameters** \((\hbar, \alpha)\) optimised jointly via `optimal_parameters`. At M = 2 the joint optimum hits within \(4 \times 10^{-4}\) of Howarth — 50× better than polynomial basis at M = 5. |

The five examples surface different regimes:

- A **closed-form hyperbolic-tangent** benchmark.
- An **initial-guess-with-non-zero-value** case.
- An **integro-differential** problem requiring the Stage 9a
  `Integral` branch of the nonlinear compiler.
- A **higher-order BVP** where the convergence-control parameter
  must be tuned away from \(\hbar = -1\) to recover the solution.
- An **exponential-basis** treatment of the same higher-order BVP
  using a custom inverter and Liao's Rule 1 (solution expression).

## What remains scope-deferred

- **Closed-form basis-aware inverter for the exponential-basis L.**
  The current Blasius-exponential example uses sympy.dsolve at each
  HAM step with symbolic \(\alpha\); M = 3 takes ≈ 12 s and grows
  super-linearly. A faster inverter that decomposes the RHS over
  \(\{\eta^j e^{-k\alpha\eta}\}\) and applies a closed-form L^{-1}
  per basis element would scale to the M ≈ 30 needed to reach
  Liao's six-decimal accuracy on \(f''(0)\).
