import time

import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import eigsh, expm


### Parameter space stuff (not all is used, just taken from Hatsugai)
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


### "Classical" stuff

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


### Unitary evolution stuff
swap = (
    kron(identity, identity, format="csr")
    + kron(pauli_x, pauli_x, format="csr")
    + kron(pauli_y, pauli_y, format="csr")
    + kron(pauli_z, pauli_z, format="csr")
) / 2


def left_shift(n_qubits, i, j, A=1):
    if i == j:
        return A

    G = 1

    for _ in range(j - 1):
        G = kron(G, identity, format="csr")

    G = kron(G, swap, format="csr")

    for _ in range(n_qubits - j - 1):
        G = kron(G, identity, format="csr")

    if i == j - 1:
        return G * A
    else:
        return left_shift(n_qubits, i, j - 1, G * A)


def right_shift(n_qubits, i, j, A=1):
    if i == j:
        return A

    G = 1

    for _ in range(i):
        G = kron(G, identity, format="csr")

    G = kron(G, swap, format="csr")

    for _ in range(i + 2, n_qubits):
        G = kron(G, identity, format="csr")

    if i + 1 == j:
        return G * A
    else:
        return right_shift(n_qubits, i + 1, j, G * A)


def two_qubit_interaction(n_qubits, i, j, A, psi):
    if i == j:
        raise Exception("Two qubits cannot have same indices... lol check your logic")
    if j < i:
        print(f"weird: {i}-{j}")

    M = 1
    for _ in range(i):
        M = kron(M, identity, format="csr")

    M = kron(M, A, format="csr")

    for _ in range(n_qubits - i - 2):
        M = kron(M, identity, format="csr")

    psi = left_shift(n_qubits, i + 1, j) * psi
    psi = M * psi
    psi = right_shift(n_qubits, i + 1, j) * psi

    return psi


def rxx(theta):
    XX = kron(pauli_x, pauli_x, format="csr")

    return expm(-1j * theta * XX)


def ryy(theta):
    YY = kron(pauli_y, pauli_y, format="csr")

    return expm(-1j * theta * YY)


def rzz(theta):
    ZZ = kron(pauli_z, pauli_z, format="csr")

    return expm(-1j * theta * ZZ)


def rxy(theta):
    XY = kron(pauli_x, pauli_y, format="csr")
    YX = kron(pauli_y, pauli_x, format="csr")

    return expm(-1j * theta * (XY - YX))


### "Quantum" stuff
def spin_interaction(n_qubits, i, j, time_rot, psi):
    M = rxx(time_rot) * ryy(time_rot) * rzz(time_rot)

    return two_qubit_interaction(n_qubits, i, j, M, psi)


# Commuting twist interactions
def spin_twist_interaction(n_qubits, i, j, time_rot, twist_angle, psi):
    coef = np.cos(twist_angle)
    M = rxx(coef * time_rot) * ryy(coef * time_rot) * rzz(time_rot)

    return two_qubit_interaction(n_qubits, i, j, M, psi)


def non_commuting_twist(n_qubits, i, j, rotation, psi):
    M = rxy(rotation)

    return two_qubit_interaction(n_qubits, i, j, M, psi)


# indexes are pairs of qubits in list format []
def bd_step(n_qubits, b_indexes, d_indexes, alpha, dt, psi):
    time_rot = alpha * dt

    for [i, j] in b_indexes:
        psi = spin_interaction(n_qubits, i, j, time_rot / 4, psi)

    for [i, j] in d_indexes:
        psi = spin_interaction(n_qubits, i, j, time_rot / 2, psi)

    for [i, j] in b_indexes:
        psi = spin_interaction(n_qubits, i, j, time_rot / 4, psi)

    return psi


# indeces are pairs of qubits in list format []
def a_split(n_qubits, a_indexes, twist_loc, twist, J, dt, psi):
    twist_rot = -J * dt * np.sin(twist)
    time_rot = J * dt

    psi = non_commuting_twist(n_qubits, twist_loc[0], twist_loc[1], twist_rot / 4, psi)

    for [i, j] in a_indexes:

        if [i, j] == twist_loc:
            psi = spin_twist_interaction(n_qubits, i, j, time_rot / 2, twist, psi)
            continue

        psi = spin_interaction(n_qubits, i, j, time_rot / 2, psi)

    psi = non_commuting_twist(n_qubits, twist_loc[0], twist_loc[1], twist_rot / 4, psi)

    return psi


# indeces are pairs of qubits in list format []
def c_split(n_qubits, c_indexes, twist_loc, twist, J, dt, psi):
    twist_rot = -J * dt * np.sin(twist)
    time_rot = J * dt

    psi = non_commuting_twist(n_qubits, twist_loc[0], twist_loc[1], twist_rot / 2, psi)

    for [i, j] in c_indexes:
        if [i, j] == twist_loc:
            psi = spin_twist_interaction(n_qubits, i, j, time_rot, twist, psi)
            continue

        psi = spin_interaction(n_qubits, i, j, time_rot, psi)

    psi = non_commuting_twist(n_qubits, twist_loc[0], twist_loc[1], twist_rot / 2, psi)

    return psi


def ac_step(n_qubits, a_indexes, c_indexes, twist_indices, twist, J, dt, psi):
    a_twist_loc = twist_indices[0]
    c_twist_loc = twist_indices[1]

    psi = a_split(n_qubits, a_indexes, a_twist_loc, twist, J, dt, psi)
    psi = c_split(n_qubits, c_indexes, c_twist_loc, twist, J, dt, psi)
    psi = a_split(n_qubits, a_indexes, a_twist_loc, twist, J, dt, psi)

    return psi


def evolution_circuit(L, J, alpha, dt, dc, twist_indices, psi):
    n_qubits = L**2

    a_links = []
    b_links = []
    c_links = []
    d_links = []

    for y in range(L):
        for x in range(L):
            qubit = x + L * y
            right_qubit = y * L + ((x + 1) % L)
            down_qubit = ((y + 1) % L) * L + x

            # "x-dir" interactions
            if x % 2 == 0:
                a_links.append(sorted([qubit, right_qubit]))
            else:
                b_links.append(sorted([qubit, right_qubit]))

            # "y-dir" interactions
            if y % 2 == 0:
                c_links.append(sorted([qubit, down_qubit]))
            else:
                d_links.append(sorted([qubit, down_qubit]))

    # Forward propagation
    for j in range(round(N / 2)):
        twist = (j + 1 / 2) * dc
        psi = bd_step(n_qubits, b_links, d_links, alpha, dt, psi)
        psi = ac_step(n_qubits, a_links, c_links, twist_indices, twist, J, dt, psi)
        psi = bd_step(n_qubits, b_links, d_links, alpha, dt, psi)

    # Backward propagation
    for j in range(round(N / 2)):
        twist = np.pi + (j + 1 / 2) * dc
        psi = bd_step(n_qubits, b_links, d_links, alpha, -dt, psi)
        psi = ac_step(n_qubits, a_links, c_links, twist_indices, twist, J, -dt, psi)
        psi = bd_step(n_qubits, b_links, d_links, alpha, -dt, psi)

    return psi


def calculate_berry_phase(L, J, alpha, twist_indices, t0, tf, N, identifier, threshold):
    print(f"Starting calculation: {identifier}")

    # Discretization
    T = tf - t0
    dt = T / N
    dc = 2 * np.pi / N

    # Circuit
    Ht = tetramerized_lattice(L, J, alpha, [], [0, 0, 0, 0])
    psi_0 = np.transpose(ground_state_optimized(Ht)[0])

    # Evolution
    psi_f = evolution_circuit(L, J, alpha, dt, dc, twist_indices, psi_0)

    # Berry phase extraction
    dotty = np.vdot(psi_0, psi_f)
    berry_phase = np.abs(np.angle(dotty))

    if berry_phase < threshold:
        berry_phase = 0

    return [identifier, berry_phase]


def print_barrier():
    print("-------------------------------------------------------------------")


def sim_start_info(L, J, alpha, twist_locs, N, threshold):
    print_barrier()
    print("STARTING BERRY PHASE CALCULATION")
    print_barrier()
    print("Calculation data:")
    print(f"-- System size: {L}x{L}")
    print(f"-- Plaquette I strength: {J}")
    print(f"-- Plaquette II strength: {alpha}")
    print(f"-- Twisting links: {twist_locs}")
    print(f"-- Closed path steps: {N}")
    print(f"-- 0 threshold: {threshold}")
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


if __name__ == "__main__":
    # Timer
    start_time = time.time()

    # Grid & discretization
    L = 4
    t0 = 0
    tf = 20
    N = 150

    # Interactions
    J = 1.0
    alphas = np.arange(0, 2.1, 0.1)
    berry_phases = np.zeros(len(alphas))

    # Twists First one hast to be in A second in C
    twist_indices = [[0, 1], [0, 4]]

    # Threshold
    threshold = 1.0e-10

    # Print info
    sim_start_info(L, J, alphas, twist_indices, N, threshold)

    # Start twist process for different link strenghts
    for i in range(len(alphas)):
        [_, phase] = calculate_berry_phase(
            L,
            J,
            alphas[i],
            twist_indices,
            t0,
            tf,
            N,
            i,
            threshold,
        )
        berry_phases[i] = phase

    # Print info
    sim_end_info(start_time, berry_phases)
