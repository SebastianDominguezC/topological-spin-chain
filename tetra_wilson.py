import multiprocessing as mp
import time

import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import eigsh


E0 = np.array([0, 0, 0])
E1 = np.array([2 * np.pi, 0, 0])
E2 = np.array([0, 2 * np.pi, 0])
E3 = np.array([0, 0, 2 * np.pi])
G = (E0 + E1 + E2 + E3) / 4


def path_up(T, N):
    dt = T / N
    t = np.arange(0, T, dt) + dt

    theta = 2 * np.pi * t / T

    return theta


def path_up_down(T, N):
    dt = T / N
    t = np.arange(0, T, dt) + dt
    theta = 2 * np.pi * t / T

    i = N // 4

    theta[i:] = 2 * np.pi * (1 - t[i:] / T) / 3

    return theta


def path_down(T, N):
    dt = T / N
    t = np.arange(0, T, dt) + dt

    theta = 2 * np.pi * (1 - 3 * t / T)

    i = N // 4

    theta[i:] = 2 * np.pi * (1 - t[i:] / T) / 3

    return theta


# Parameter space paths
def L1(T, N):
    theta_1 = path_up(T, N)
    theta_2 = path_up_down(T, N)
    theta_3 = path_up_down(T, N)

    return [theta_1, theta_2, theta_3]


def L2(T, N):
    theta_1 = path_down(T, N)
    theta_2 = path_up(T, N)
    theta_3 = path_up_down(T, N)

    return [theta_1, theta_2, theta_3]


def L3(T, N):
    theta_1 = path_up_down(T, N)
    theta_2 = path_down(T, N)
    theta_3 = path_up(T, N)

    return [theta_1, theta_2, theta_3]


def L4(T, N):
    theta_1 = path_up_down(T, N)
    theta_2 = path_up_down(T, N)
    theta_3 = path_down(T, N)

    return [theta_1, theta_2, theta_3]


def joined_paths(T, N):
    l1 = L1(T, N)
    l2 = L2(T, N)
    l3 = L3(T, N)
    l4 = L4(T, N)

    theta_1 = np.concatenate((l1[0], l2[0], l3[0], l4[0]))
    theta_2 = np.concatenate((l1[1], l2[1], l3[1], l4[1]))
    theta_3 = np.concatenate((l1[2], l2[2], l3[2], l4[2]))

    return [theta_1, theta_2, theta_3]


# Operational utilities
pauli_x = csr_matrix(np.matrix([[0, 1], [1, 0]], dtype=np.complex_))
pauli_y = csr_matrix(np.matrix([[0, -1j], [1j, 0]], dtype=np.complex_))
pauli_z = csr_matrix(np.matrix([[1, 0], [0, -1]], dtype=np.complex_))
identity = csr_matrix(np.matrix([[1, 0], [0, 1]], dtype=np.complex_))


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

    # if i == j: raise Exception("i and j should not be the same")
    # if i > j or i > n : raise Exception("i should be smaller than j and n")

    # "0D" matrix to tensor into
    M = 1.0

    # If j should "loop back" i.e a periodic condition
    while (j > n or j == n) and periodic:
        j -= n

    # Kronecker product of matrices in place
    for k in range(0, n):
        if i == k:
            M = kron(M, I, format="csr")
            continue

        if j == k:
            M = kron(M, J, format="csr")
            continue

        M = kron(M, identity, format="csr")

    return M


# Gets an eigenvector from numpy object
def extract_eigenvector(col, eigenvectors):
    n = eigenvectors.shape[0]
    vec = np.zeros(n, dtype=np.complex_)

    for i, row in enumerate(eigenvectors):
        vec[i] = row[col][0]

    return vec


# Ground state with lancoz algorithm
def ground_state_optimized(H, show_states=False):
    E = eigsh(H, k=2, which="SA")
    i = np.where(E[0] == E[0].min())
    j = (i[0][0] + 1) % 2
    E0 = E[0][i[0][0]]
    E1 = E[0][j]

    if E[0][0] == E[0][1]:
        print("WARNING: Degenerate ground state")
        if not show_states:
            print(E[0])

    if show_states:
        print(E[0])

    return [extract_eigenvector(i, E[1]), E0, E1]


# <r|M|v> ~ matrix entry
def matrix_entry(row_vec, col_vec, M):
    mid = np.matmul(row_vec, M)
    return np.matmul(mid, col_vec)


# Builds the 2d dimer lattice ~ tetramerized :D
def tetramerized_lattice(N, J, a, twist_locations, twists, feedback=False):
    H = 0
    n_qubits = N**2

    for y in range(N):  # y coord
        for x in range(N):  # x coord
            current_spin = y * N + x
            right_neighbor = y * N + ((x + 1) % N)
            down_neighbor = ((y + 1) % N) * N + x

            Jx = J
            Jy = J

            if x % 2 == 1:
                Jx = a
            if y % 2 == 1:
                Jy = a

            # x dir interactions
            XX1 = matrix_gen(current_spin, right_neighbor, n_qubits, pauli_x, pauli_x)
            YY1 = matrix_gen(current_spin, right_neighbor, n_qubits, pauli_y, pauli_y)
            ZZ1 = matrix_gen(current_spin, right_neighbor, n_qubits, pauli_z, pauli_z)

            if (current_spin, right_neighbor) in twist_locations:
                twist_index = twist_locations.index((current_spin, right_neighbor))
                twist = twists[twist_index]
                if feedback:
                    print(
                        f"twisting {current_spin}-{right_neighbor} by {twist} strength {Jx}"
                    )

                XY1 = matrix_gen(
                    current_spin, right_neighbor, n_qubits, pauli_x, pauli_y
                )
                YX1 = matrix_gen(
                    current_spin, right_neighbor, n_qubits, pauli_y, pauli_x
                )

                H += Jx * (
                    np.cos(twist) * (XX1 + YY1) - np.sin(twist) * (XY1 - YX1) + ZZ1
                )

            else:
                H += Jx * (XX1 + YY1 + ZZ1)

            # y dir interactions
            XX2 = matrix_gen(current_spin, down_neighbor, n_qubits, pauli_x, pauli_x)
            YY2 = matrix_gen(current_spin, down_neighbor, n_qubits, pauli_y, pauli_y)
            ZZ2 = matrix_gen(current_spin, down_neighbor, n_qubits, pauli_z, pauli_z)

            if (current_spin, down_neighbor) in twist_locations:
                twist_index = twist_locations.index((current_spin, down_neighbor))
                twist = twists[twist_index]
                if feedback:
                    print(
                        f"twisting {current_spin}-{down_neighbor} by {twist} strength {Jy}"
                    )

                XY2 = matrix_gen(
                    current_spin, down_neighbor, n_qubits, pauli_x, pauli_y
                )
                YX2 = matrix_gen(
                    current_spin, down_neighbor, n_qubits, pauli_y, pauli_x
                )

                H += Jy * (
                    np.cos(twist) * (XX2 + YY2) - np.sin(twist) * (XY2 - YX2) + ZZ2
                )
            else:
                H += Jy * (XX2 + YY2 + ZZ2)

    return H


def print_barrier():
    print("-------------------------------------------------------------------")


def sim_start_info(size, n_qubits, J, alphas, path_string, total_steps, threshold):
    print_barrier()
    print("STARTING BERRY PHASE CALCULATION")
    print_barrier()
    print("Calculation data:")
    print(f"-- Parameter path: {path_string}")
    print(f"-- # of steps: {total_steps}")
    print(f"-- 0 threshold: {threshold}")
    print(f"-- Processors to use: {mp.cpu_count()}")
    print(f"-- System size: {size}x{size}")
    print(f"-- # of qubits: {n_qubits}")
    print(f"-- Bond plaquette 1: {J}")
    print(f"-- Bonds plaquette 2: {alphas}")
    print_barrier()


def sim_end_info(start_time, berry_phases):
    print_barrier()
    print("RESULTS")
    print_barrier()
    print("Berry phases:")
    print(berry_phases)
    print_barrier()
    print("Simulation time")
    end_time = time.time()
    print(f"{end_time - start_time} seconds")
    print_barrier()


# Run the algorithm
def berry_phase_tetramerized(
    size, J, alpha, twist_locations, parameter_paths, total_steps, threshold, identifier
):
    # Get relevant twist
    twist_1 = parameter_paths[0]

    twists = [0, 0, 0, 0]

    H = tetramerized_lattice(size, J, alpha, twist_locations, twists)
    ref_state = ground_state_optimized(H, True)[0]
    prev_state = ref_state

    wilson = 1.0 + 0j

    for i in range(total_steps):
        # Twist hamiltonian
        next_twists = [twist_1[i], twist_1[i], 0, 0]

        # Calculate new "ground state" wave fx
        Ht = tetramerized_lattice(size, J, alpha, twist_locations, next_twists)
        [next_state, e0, e1] = ground_state_optimized(Ht)

        if np.round(e0, 6) == np.round(e1, 6):
            print(f"Near degeneracy found at alpha={alpha}")
            print(f"e0: {e0}")
            print(f"e1: {e1}")
            print(f"At time step {i}")
            print(f"Corresponding to twists: {next_twists}")

        # Wilson loop operator
        wilson *= np.vdot(prev_state, next_state)

        if i == total_steps - 1:
            wilson *= np.vdot(next_state, ref_state)

        prev_state = next_state

    # Get berry phase :)
    berry_phase = np.abs(np.angle(wilson))

    if berry_phase < threshold:
        berry_phase = 0

    return (identifier, berry_phase)


if __name__ == "__main__":
    # Timer
    start_time = time.time()

    # Threshold
    threshold = 1.0e-10

    # Evolution parameters
    T = 10
    N = 20
    parameter_paths = L1(T, N)
    path_string = "All"
    total_steps = len(parameter_paths[0])

    # System parameters
    size = 4
    n_qubits = size**2
    J = 1
    alphas = np.arange(0, 2.1, 0.1)

    # Twist site
    twist_locations = [(0, 1), (0, 4), (1, 5), (4, 5)]

    # Multi-processing
    pool = mp.Pool(mp.cpu_count())
    berry_phases = np.zeros(len(alphas))

    # Callback for async data
    def async_data(result):
        i = result[0]
        phase = result[1]
        berry_phases[i] = phase

    # Print info
    sim_start_info(size, n_qubits, J, alphas, path_string, total_steps, threshold)

    # Start twist process for every site
    for id, alpha in enumerate(alphas):
        pool.apply_async(
            berry_phase_tetramerized,
            args=(
                size,
                J,
                alpha,
                twist_locations,
                parameter_paths,
                total_steps,
                threshold,
                id,
            ),
            callback=async_data,
        )

    # Close processors
    pool.close()
    pool.join()

    # Print info
    sim_end_info(start_time, berry_phases)
