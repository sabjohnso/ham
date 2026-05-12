# Worked examples

Each example is a complete HAM solve of a problem with a known
closed-form solution. The library ships them under `examples/` as both
importable modules (with factory functions `build_problem`,
`solve_to`, `analyze`, `is_convergent`) and runnable scripts:

```sh
poetry run python examples/quadratic_drag.py
poetry run python examples/logistic.py
```

The narrative pages below walk through each example end-to-end:
problem statement, HAM setup, deformation chain, diagnostics, the
interesting observations that fell out of the run.

## The two shipped examples

| Example | ODE | Exact solution | Highlight |
| --- | --- | --- | --- |
| [Quadratic drag](quadratic-drag.md) | \(v'(t) = 1 - v(t)^2\), \(v(0) = 0\) | \(v(t) = \tanh(t)\) | At order 7 the L²-optimal \(\hbar\) is \(-\tfrac{1}{2}\), not \(-1\) — the adaptive-ℏ advantage at work. |
| [Logistic](logistic.md) | \(u'(t) = u(t)(1 - u(t))\), \(u(0) = \tfrac{1}{2}\) | \(u(t) = 1/(1 + e^{-t})\) | First example with a *non-zero initial guess* (\(u_0 = 1/2\)); the sigmoid Taylor converges fast enough that \(\hbar = -1\) cleanly dominates. |

The two examples are complementary by design: between them they
exercise every code path in the library and surface two contrasting
behaviours that anyone using HAM in practice will encounter.

## Deferred — Volterra and Blasius

The original PLAN listed three examples; Volterra and Blasius are
deferred:

- **Volterra population model** — polynomial \(N\), Liao Ch. 10.
  Deferred not for technical reasons but for scope.
- **Blasius boundary-layer equation** — canonical HAM benchmark with
  asymptotic BC \(f'(\infty) = 1\). The library's
  `BoundaryCondition` accepts only *point* BCs at this time;
  Blasius requires a library extension to represent asymptotic
  conditions in a form the dsolve-backed inverter can consume.

Both are in scope for a future stage. The infrastructure for adding
worked examples — `examples/<name>.py` + `tests/examples/test_<name>.py`
— is the path of least resistance for new contributions.
