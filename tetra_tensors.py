import numpy as np
import scipy as sp
import multiprocessing as mp
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    RXXGate,
    RYYGate,
    RZZGate,
    XXPlusYYGate,
)
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate

from qiskit_aer.noise import (
    NoiseModel,
    pauli_error,
)

import multiprocessing as mp
import time

import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import eigsh

import tntorch as tn
import torch

RXX = RXXGate
RYY = RYYGate
RZZ = RZZGate


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
pauli_x = csr_matrix(np.matrix([[0, 1], [1, 0]], dtype=np.complex128))
pauli_y = csr_matrix(np.matrix([[0, -1j], [1j, 0]], dtype=np.complex128))
pauli_z = csr_matrix(np.matrix([[1, 0], [0, -1]], dtype=np.complex128))
identity = csr_matrix(np.matrix([[1, 0], [0, 1]], dtype=np.complex128))


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
    vec = np.zeros(n, dtype=np.complex128)

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


### "Quantum" stuff
def spin_interaction(time_rot):
    qc = QuantumCircuit(2)

    qc.rxx(time_rot, 0, 1)
    qc.ryy(time_rot, 0, 1)
    qc.rzz(time_rot, 0, 1)

    return qc.to_gate()


# Commuting twist interactions
def spin_twist_interaction(time_rot, twist_angle):
    qc = QuantumCircuit(2)

    qc.rxx(time_rot * np.cos(twist_angle), 0, 1)
    qc.ryy(time_rot * np.cos(twist_angle), 0, 1)
    qc.rzz(time_rot, 0, 1)

    return qc.to_gate()


def non_commuting_twist(rotation):
    qc = QuantumCircuit(2)

    qc.append(XXPlusYYGate(rotation, -np.pi / 2), [0, 1])

    return qc.to_gate()


# indexes are pairs of qubits in list format []
def bd_step(n_qubits, b_indexes, d_indexes, alpha, dt):
    qc = QuantumCircuit(n_qubits)

    for pair in b_indexes:
        time_rot = alpha * dt / 2
        qc.append(spin_interaction(time_rot), pair)

    for pair in d_indexes:
        time_rot = alpha * dt
        qc.append(spin_interaction(time_rot), pair)

    for pair in b_indexes:
        time_rot = alpha * dt / 2
        qc.append(spin_interaction(time_rot), pair)

    return qc.to_gate()


# indeces are pairs of qubits in list format []
def a_split(n_qubits, a_indexes, twist_loc, twist, J, dt):
    qc = QuantumCircuit(n_qubits)

    twist_rot = -J * dt * np.sin(twist)
    qc.append(non_commuting_twist(twist_rot), twist_loc)

    for pair in a_indexes:
        time_rot = J * dt

        if pair == twist_loc:
            qc.append(spin_twist_interaction(time_rot, twist), pair)
            continue

        qc.append(spin_interaction(time_rot), pair)

    qc.append(non_commuting_twist(twist_rot), twist_loc)

    return qc.to_gate()


# indeces are pairs of qubits in list format []
def c_split(n_qubits, c_indexes, twist_loc, twist, J, dt):
    qc = QuantumCircuit(n_qubits)

    twist_rot = -2 * J * dt * np.sin(twist)
    qc.append(non_commuting_twist(twist_rot), twist_loc)

    for pair in c_indexes:
        time_rot = 2 * J * dt

        if pair == twist_loc:
            qc.append(spin_twist_interaction(time_rot, twist), pair)
            continue

        qc.append(spin_interaction(time_rot), pair)

    qc.append(non_commuting_twist(twist_rot), twist_loc)

    return qc.to_gate()


def ac_step(n_qubits, a_indexes, c_indexes, twist_indices, twist, J, dt):
    qc = QuantumCircuit(n_qubits)
    qubits = range(n_qubits)

    a_twist_loc = twist_indices[0]
    c_twist_loc = twist_indices[1]

    qc.append(a_split(n_qubits, a_indexes, a_twist_loc, twist, J, dt), qubits)
    qc.append(c_split(n_qubits, c_indexes, c_twist_loc, twist, J, dt), qubits)
    qc.append(a_split(n_qubits, a_indexes, a_twist_loc, twist, J, dt), qubits)

    return qc.to_gate()


def evolution_circuit(L, J, alpha, dt, dc, twist_indices):
    n_qubits = L**2
    qc = QuantumCircuit(n_qubits)

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
                a_links.append([qubit, right_qubit])
            else:
                b_links.append([qubit, right_qubit])

            # "y-dir" interactions
            if y % 2 == 0:
                c_links.append([qubit, down_qubit])
            else:
                d_links.append([qubit, down_qubit])

    qubits = range(n_qubits)

    # Forward propagation
    for j in range(round(N / 2)):
        twist = (j + 1 / 2) * dc
        qc.append(bd_step(n_qubits, b_links, d_links, alpha, dt), qubits)
        qc.append(
            ac_step(n_qubits, a_links, c_links, twist_indices, twist, J, dt), qubits
        )
        qc.append(bd_step(n_qubits, b_links, d_links, alpha, dt), qubits)

    # Backward propagation
    for j in range(round(N / 2)):
        twist = np.pi + (j + 1 / 2) * dc
        qc.append(bd_step(n_qubits, b_links, d_links, alpha, -dt), qubits)
        qc.append(
            ac_step(n_qubits, a_links, c_links, twist_indices, twist, J, -dt), qubits
        )
        qc.append(bd_step(n_qubits, b_links, d_links, alpha, -dt), qubits)

    return qc


# Ground state preparation
def quantum_circuit_from_tt(tensor_network):
    n_qubits = len(tensor_network.cores)
    cores = tensor_network.cores

    W = []

    G = cores[0]
    nk = G.shape[-1]
    M = G.reshape(2, nk)
    U, S, V = np.linalg.svd(M)
    S = torch.tensor(np.diag(S), dtype=torch.complex128)

    R = (S @ V).clone().detach()
    G_next = cores[1]

    cores[1] = np.einsum("ij,jkl->ikl", R, G_next)

    W.append(U)

    prev_lk = 1

    for k in range(1, n_qubits):
        G = cores[k]
        nk = G.shape[-1]
        mk = int(2 * 2 ** min(k, prev_lk))
        prev_lk = np.log2(nk)
        M = G.reshape(mk, nk)
        U, S, V = np.linalg.svd(M)
        S = torch.tensor(np.diag(S))

        R = (S @ V).clone().detach()
        W.append(U)

        if k + 1 == n_qubits:
            continue

        G_next = cores[k + 1]

        cores[k + 1] = np.einsum("ij,jkl->ikl", R, G_next)

    return W


def create_circuit(W_list: list, n: int):
    circ = QuantumCircuit(n)
    for i, unitary in enumerate(W_list):
        circ.append(UnitaryGate(unitary), range(i, int(np.log2(len(unitary))) + i))

    return circ


# Parameters and sim run
L = 4
n_qubits = L**2
noisy = False

t0 = 0
tf = 20
N = 300

T = tf - t0
dt = T / N
dc = 2 * np.pi / N

changes = np.arange(0, 1.1, 0.1)
ones = np.ones(len(changes))
alphas = np.concatenate([changes, ones])
Js = np.concatenate([ones, changes[::-1]])
runs = len(Js)
# alphas = np.arange(0, 2.1, 0.1)
# runs = len(alphas)
# Js = np.ones(runs)

twist_indices = [[0, 1], [0, 4]]

berry_phases = []

for k in range(runs):
    print(f"running run number: {k} ----------")
    # Circuit
    Ht = tetramerized_lattice(L, Js[k], alphas[k], [(0, 1), (0, 4)], [0, 0, 0, 0])
    psi = ground_state_optimized(Ht)[0]

    start_time = time.time()

    # Preparation
    qc = QuantumCircuit(n_qubits + 1, 1)

    def probability_dist(x):
        results = []

        for bitstring in x:
            index = int("0b" + "".join([str(int(val.item())) for val in bitstring]), 2)
            results.append(psi[index])

        return torch.tensor(results)

    sample = [torch.arange(0, 2) for _ in range(n_qubits)]

    tensor_state = tn.cross(
        function=probability_dist,
        domain=sample,
        function_arg="matrix",
        ranks_tt=n_qubits,
    )

    state_preparation_gates = quantum_circuit_from_tt(tensor_state)

    state_preparation_circuit = create_circuit(state_preparation_gates[::-1], n_qubits)

    qc.append(state_preparation_circuit, range(1, n_qubits + 1))
    qc.h(0)
    qc.barrier()

    # Evolution
    evolution_gate = evolution_circuit(
        L, Js[k], alphas[k], dt, dc, twist_indices
    ).control(1)

    qc.append(evolution_gate, range(n_qubits + 1))

    # Hadamard test
    qc.h(0)
    qc.measure(0, 0)

    time_interval = time.time() - start_time
    print(f"finished applying gates in {time_interval}")

    print("Transpiling now")

    start_time = time.time()

    # Noisy simulator
    simulator = AerSimulator()

    if noisy:
        # Example error probabilities
        p_gate1 = 0.0000001

        # QuantumError objects
        error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)

        # Add errors to noise model
        noise_bit_flip = NoiseModel()
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

        simulator = AerSimulator(noise_model=noise_bit_flip)

    # Transpile for simulator
    circ = transpile(qc, simulator, optimization_level=3)

    time_interval = time.time() - start_time
    print(f"finished transpiling in {time_interval}")
    print("Operations: ", circ.count_ops())
    print("# of gates: ", sum(circ.count_ops().values()))
    print("Depth: ", circ.depth())
    start_time = time.time()

    # Run and get counts
    result = simulator.run(circ, shots=100_000).result()

    zeros = result.data()["counts"].get("0x0", 0)
    ones = result.data()["counts"].get("0x1", 0)
    total = zeros + ones

    p0 = zeros / total

    berry = 2 * np.arccos(np.sqrt(p0))

    time_interval = time.time() - start_time

    print(f"Finished running in {time_interval}")
    print("Berry phase is:")
    print(berry)
    berry_phases.append(berry)

print("-------------------")
print("Finished running simulation")
print(f"J: {Js}")
print(f"a: {alphas}")
print(f"B: {berry_phases}")
