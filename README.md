# Local topological order parameters in dimerized spin systems via quantum algorithms

The present repository implements various codes to calculate the Berry phase of dimerized spin systems in 1D chains and 2D grids. The Berry phase serves as a local topological order parameter. To understand the theoretical and numerical tools in the scripts, please read `research-notes.pdf` and `poster.pdf`. The scripts are organized as follows:

- `dimer_wilson.py` ~ Numerical diagonalization and integral for Berry phase in a 1D dimerized chain.
- `dimer_qcircuit.py` ~ Quantum algorithm for Berry phase in a 1D dimerized chain.
- `tetra_wilson.py` ~ Numerical diagonalization and integral for Berry phase in a 2D dimerized grid.
- `tetra_unitary.py` ~ Sparse matrix aproach to quantum algorithm to calculate the Berry phase in a 2D dimerized grid.
- `tetra_qcircuit.py` ~ Quantum algorithm for Berry phase in a 2D dimerized grid.

Please note that the tetramerized (2D dimerized) quantum circuit is not simulatable on a laptop, since ground state preparation takes too long of a time. It is a worthy research topic to find a way to start a ground state for a system as this one. The 1D quantum circuit is not runnable on hardware either, since there are too many operations in place, and the system will becomes noise before the algorithm can run.
