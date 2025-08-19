# Local topological order parameters in dimerized spin systems via quantum algorithms

The present repository implements various codes to calculate the Berry phase of dimerized spin systems in 1D chains and 2D grids. The Berry phase serves as a local topological order parameter. To understand the theoretical and numerical tools in the scripts, please read `research-notes.pdf` and `poster.pdf`. The scripts are organized as follows:

- `dimer_wilson.py` ~ Numerical diagonalization and integral for Berry phase in a 1D dimerized chain.
- `dimer_qcircuit.py` ~ Quantum algorithm for Berry phase in a 1D dimerized chain.
- `tetra_wilson.py` ~ Numerical diagonalization and integral for Berry phase in a 2D dimerized grid.
- `tetra_unitary.py` ~ Sparse matrix aproach to quantum algorithm to calculate the Berry phase in a 2D dimerized grid.
- `tetra_qcircuit.py` ~ Quantum algorithm for Berry phase in a 2D dimerized grid using state preparation with Qiskit built-ins.
- `tetra_tensors.py` ~ Same quantum algorithm as above, but state preparation is done by encoding the probability distribution into a tensor network via a quantics tensor train, which is then mapped to gates which prepare the state. Does not work as well as Qiskit's state prep.

Please note that the quantum circuits are thousands of gates long (circuit depth ~100k), so they are not runnable on NISQ-era devices.