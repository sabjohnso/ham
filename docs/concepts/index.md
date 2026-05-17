# Concepts

The math behind HAM, written from scratch. Read in order — each page
builds on the previous — or jump to the topic you need.

- **[The homotopy equation](homotopy-equation.md)** — why HAM works:
  a continuous deformation between an initial guess and the exact
  solution, controlled by an embedding parameter \(q \in [0, 1]\) and
  a free convergence parameter \(\hbar\).
- **[The deformation chain](deformation-chain.md)** — how to extract
  the m-th order coefficient \(u_m(x)\): expand the homotopy series in
  \(q\), match powers, solve a chain of *linear* ODEs that all involve
  the same auxiliary operator \(L\).
- **[Convergence](convergence.md)** — Liao's Theorem 2.1, the three
  fundamental rules for choosing the auxiliary operator and base
  functions, and the diagnostics that justify reporting the partial
  sum as a solution.
- **[Padé acceleration](pade-acceleration.md)** — when the formal
  series in \(q\) has radius of convergence less than 1, the bare
  partial sum at \(q = 1\) diverges; the homotopy-Padé approximant
  analytically continues past that radius and often gives the right
  answer anyway.
- **[Substrates: where the coefficients live](substrates.md)** — the
  homotopy chain doesn't care whether each \(u_k(x)\) is a sympy
  expression or a grid vector. The library is generic over a
  substrate abstraction; the symbolic-HAM and Spectral-HAM modes are
  two concrete substrates sharing the same solver loop.

The references throughout are Liao's book
*Beyond Perturbation* (Chapman & Hall/CRC 2003) and the 2009 review
paper *Notes on the homotopy analysis method* (CNSNS 14: 983–997).
The substrate page additionally references Motsa, Sibanda, & Shateyi
(CNSNS 15, 2010, the SHAM origin paper) and Trefethen, *Spectral
Methods in MATLAB* (SIAM 2000).
