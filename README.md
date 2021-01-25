# ContextualSubspaceVQE

ContextualSubspaceVQE provides a classical simulation implementation of the hybrid quantum-classical algorithm called *contextual subspace variational quantum eigensolver* (CS-VQE), as described in [https://arxiv.org/abs/2011.10027](https://arxiv.org/abs/2011.10027).

## What is included

- Test for contextuality of a Hamiltonian.
- Construction of quasi-quantized (classical) models for noncontextual Hamiltonians.
- Heuristics to approximate optimal noncontextual sub-Hamiltonians given arbitrary target Hamiltonians.
- Classical simulation of quantum correction to a noncontextual approximation.
- Classical simulation of CS-VQE.

## How to cite

When you use ContextualSubspaceVQE in a publication or other work, please cite as:

> William M. Kirby, Andrew Tranter, and Peter J. Love, *Contextual Subspace Variational Quantum Eigensolver*, arXiv preprint (2020), [arXiv:2011.10027](https://arxiv.org/abs/2011.10027).

## How to use

Take a look through [CS_VQE_how_to_use.ipynb](https://github.com/wmkirby1/ContextualSubspaceVQE/blob/main/CS_VQE_how_to_use.ipynb) for examples of the usage of all of the main functions.
