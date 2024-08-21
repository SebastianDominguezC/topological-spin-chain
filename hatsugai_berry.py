import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import multiprocessing as mp
import time

# Pauli matrices
pauli_x = np.matrix([[0, 1], [1, 0]])
pauli_y = np.matrix([[0, -1j], [1j, 0]])
pauli_z = np.matrix([[1, 0], [0, -1]])


# A function to build the matrix with correct order of kronecker products
def matrix_gen(i, j, n, I, J, periodic=True):
    """
    Generates n-by-n matrix of kronecker products, placing I at "position" i and J at "position" j.
    i must be smaller than j and n.

    Parameters:
    i (int): position to tensor I
    j (int): position to tensor J
    n (int): size of square matrix
    I (np.array): square matrix
    J (np.array): square matrix
    periodic (bool): boundary condition of interaction
    """

    if i == j:
        raise Exception("i and j should not be the same")
    if i > j or i > n:
        raise Exception("i should be smaller than j and n")

    # "0D" matrix to tensor into
    M = np.identity(1)

    # If j should "loop back" i.e a periodic condition
    while (j > n or j == n) and periodic:
        j -= n

    # Kronecker product of matrices in place
    for k in range(0, n):
        if i == k:
            M = np.kron(M, I)
            continue

        if j == k:
            M = np.kron(M, J)
            continue

        M = np.kron(M, np.identity(2))

    return M


# For time evolution
def propagator(H, t0, tf, steps):
    dt = (tf - t0) / steps
    L = H.shape[0]
    U = np.empty((L, L, steps), dtype="complex_")

    for i, t in enumerate(np.arange(t0, tf, dt)):
        U[:, :, i] = sp.linalg.expm(-1.0j * H * t)

    return U


# Gets an eigenvector from numpy object
def extract_eigenvector(col, eigenvectors):
    n = eigenvectors.shape[0]
    vec = np.zeros(n, dtype=np.complex_)

    for i, row in enumerate(eigenvectors):
        vec[i] = row[col][0]

    return vec


# Gets eigenstate with min eigenvalue of a matrix
def ground_state(H):
    E = np.linalg.eig(H)
    i = np.where(E.eigenvalues == E.eigenvalues.min())

    return extract_eigenvector(i, E.eigenvectors)


# <r|M|v> ~ matrix entry
def matrix_entry(row_vec, col_vec, M):
    mid = np.matmul(row_vec, M)
    return np.matmul(mid, col_vec)


def interaction_chain_nearest_neighbors(n, J, alpha):
    interactions = np.zeros(n)

    for i in range(n):
        if i % 2 == 0:
            interactions[i] = J
        else:
            interactions[i] = alpha

    return interactions


def ground_state_optimized(H, show_states=False):
    E = sp.sparse.linalg.eigsh(H, k=2, which="SA")
    i = np.where(E[0] == E[0].min())

    if E[0][0] == E[0][1]:
        print("WARNING: Degenerate ground state")
        if not show_states:
            print(E[0])

    if show_states:
        print(E[0])

    return extract_eigenvector(i, E[1])


# Twistyyy
def twisted_nn_heisenberg_chain(n_qubits, interactions, twist, twist_loc):
    N = 2**n_qubits
    H = np.zeros((N, N), dtype=np.complex_)

    for i in range(n_qubits):
        # Gets bond strength
        coef = interactions[i]

        # Si.Sj interactions
        XX = matrix_gen(i, i + 1, n_qubits, pauli_x, pauli_x)
        YY = matrix_gen(i, i + 1, n_qubits, pauli_y, pauli_y)
        ZZ = matrix_gen(i, i + 1, n_qubits, pauli_z, pauli_z)

        # Twist
        if i == twist_loc:
            XY = matrix_gen(i, i + 1, n_qubits, pauli_x, pauli_y)
            YX = matrix_gen(i, i + 1, n_qubits, pauli_y, pauli_x)

            H += coef * (np.cos(twist) * (XX + YY) - np.sin(twist) * (XY - YX) + ZZ)
            continue

        # No twist, normal S.S interaction
        H += coef * (XX + YY + ZZ)
    return H


def berry_phase_hatsugai(n_qubits, site, C, bonds, threshold, feedback=False):
    if feedback:
        print(f"running site {site}")

    # Wislon loop operator
    wilson = 1.0 + 0.0j

    # Ground / reference state
    Hg = twisted_nn_heisenberg_chain(n_qubits, bonds, 0, site) / 4
    ref_state = ground_state_optimized(Hg)
    prev_state = ref_state

    for twist in C:
        # Twist hamiltonian
        Ht = twisted_nn_heisenberg_chain(n_qubits, bonds, twist, site) / 4

        # Calculate new "ground state" wave fx
        next_state = ground_state_optimized(Ht, show_states=True)

        # Wilson loop operator
        c1 = np.vdot(ref_state, prev_state)
        c2 = np.vdot(prev_state, next_state)
        c3 = np.vdot(next_state, ref_state)

        wilson *= c1 * c2 * c3

        prev_state = next_state

    if feedback:
        print(f"twist site {site} finished")

    # Store berry phase for lattice site
    berry_phase = np.abs(np.angle(wilson))

    if berry_phase < threshold:
        berry_phase = 0

    return (site, berry_phase)


def print_barrier():
    print("-------------------------------------------------------------------")


def sim_start_info(n_qubits, J, alpha, N, threshold):
    print_barrier()
    print("STARTING BERRY PHASE CALCULATION")
    print_barrier()
    print("Calculation data:")
    print(f"-- System size: {n_qubits}")
    print(f"-- 1st bond strength: {J}")
    print(f"-- 2nd bond strength: {alpha}")
    print(f"-- Closed path steps: {N}")
    print(f"-- 0 threshold: {threshold}")
    print(f"-- Processors to use: {mp.cpu_count()}")
    print_barrier()


def sim_end_info(start_time, bonds, berry_phases):
    print_barrier()
    print("RESULTS")
    print_barrier()
    print("Dimer pattern:")
    print(bonds)
    print_barrier()
    print("Berry phases:")
    print(berry_phases)
    print_barrier()
    print("Simulation time")
    end_time = time.time()
    print(f"{end_time - start_time} seconds")
    print_barrier()


if __name__ == "__main__":
    # Timer
    start_time = time.time()

    # Sim data
    n_qubits = 6
    J = 1
    g = 0.01
    alpha = J / g
    bonds = interaction_chain_nearest_neighbors(n_qubits, J, alpha)

    threshold = 1.0e-10
    N = 16
    dC = 2 * np.pi / N

    C = np.arange(dC, 2 * np.pi, dC)

    # Multi-processors
    pool = mp.Pool(mp.cpu_count())
    berry_phases = np.zeros(n_qubits)

    # Callback for async data
    def async_data(result):
        i = result[0]
        phase = result[1]
        berry_phases[i] = phase

    # Print info
    sim_start_info(n_qubits, J, alpha, N, threshold)

    # Start twist process for every site
    for site in range(1):
        pool.apply_async(
            berry_phase_hatsugai,
            args=(n_qubits, site, C, bonds, threshold, True),
            callback=async_data,
        )

    # Close processors
    pool.close()
    pool.join()

    # Print info
    sim_end_info(start_time, bonds, berry_phases)
