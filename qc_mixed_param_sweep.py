import numpy as np
import scipy as sp
import multiprocessing as mp
import time
import sys
import io
import os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    RXXGate,
    RYYGate,
    RZZGate,
    XXPlusYYGate,
)
from qiskit_aer import AerSimulator

RXX = RXXGate
RYY = RYYGate
RZZ = RZZGate


### "Classical" stuff
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


pauli_x = np.matrix([[0, 1], [1, 0]])
pauli_y = np.matrix([[0, -1j], [1j, 0]])
pauli_z = np.matrix([[1, 0], [0, -1]])


def twisted_dimerized_chain(L, interactions, twist, location):
    N = 2**L
    H = np.zeros((N, N), dtype=np.complex_)

    for i in range(L):
        # Gets bond strength
        coef = interactions[i]

        # S.S interactions
        XX = matrix_gen(i, i + 1, L, pauli_x, pauli_x)
        YY = matrix_gen(i, i + 1, L, pauli_y, pauli_y)
        ZZ = matrix_gen(i, i + 1, L, pauli_z, pauli_z)

        # Twist
        if i == location:
            XY = matrix_gen(i, i + 1, L, pauli_x, pauli_y)
            YX = matrix_gen(i, i + 1, L, pauli_y, pauli_x)

            H += coef * (np.cos(twist) * (XX + YY) - np.sin(twist) * (XY - YX) + ZZ)
            continue

        H += coef * (XX + YY + ZZ)
    return H


# Gets an eigenvector from numpy object
def extract_eigenvector(col, eigenvectors):
    n = eigenvectors.shape[0]
    vec = np.zeros(n, dtype=np.complex_)

    for i, row in enumerate(eigenvectors):
        vec[i] = row[col][0]

    return vec


# Gets ground state using scipy eigsh
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


# Dimmer pattern
def interaction_chain_nn(n, J, alpha):
    interactions = np.zeros(n)

    for i in range(n):
        if i % 2 == 0:
            interactions[i] = J
        else:
            interactions[i] = alpha

    return interactions


### "Quantum" stuff


# Regular spin interactions exp(-i t S.S)
def spin_interaction(time_rot):
    qc = QuantumCircuit(2)

    qc.rxx(time_rot, 0, 1)
    qc.ryy(time_rot, 0, 1)
    qc.rzz(time_rot, 0, 1)

    return qc.to_gate()


# Commuting spin and twist interactions
def spin_twist_interaction(time_rot, theta):
    qc = QuantumCircuit(2)

    qc.rxx(time_rot * np.cos(theta), 0, 1)
    qc.ryy(time_rot * np.cos(theta), 0, 1)
    qc.rzz(time_rot, 0, 1)

    return qc.to_gate()


# Loopy index
def next_qubit_index(i, L):
    k = i + 1
    if k >= L:
        k -= L

    return k


# Gate that generates unitary evolution of twisted dimer
def twisted_dimer_evolution(L, J, alpha, twist_loc, t0, tf, N):
    if L % 2 != 0:
        raise Exception("Dimer chain ust be even.")

    qc = QuantumCircuit(L)

    # Time period
    T = tf - t0

    # Time and path discretization
    dt = T / N
    dC = 2 * np.pi / N

    # A and C groups (even or odd)
    a_links = range(0, L, 2)
    b_links = range(1, L, 2)

    Ja = J
    Jb = alpha

    # If twist loc is even, then A must be the odd part of circuit (reverse for above)
    if twist_loc % 2 == 0:
        a_links = range(1, L, 2)
        b_links = range(0, L, 2)

        Ja = alpha
        Jb = J

    # Forward propagation
    for j in range(round(N / 2)):
        # Theta
        twist = (j + 1 / 2) * dC
        # print(twist)

        # a interactions
        for i in a_links:
            # No interaction
            if Ja == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = dt * Ja

            # Gates
            qc.append(spin_interaction(time_rot), [i, k])

        # b interactions
        for i in b_links:
            if Jb == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = dt * Jb

            # Gates
            if i == twist_loc:
                qc.append(spin_twist_interaction(time_rot, twist), [i, k])
            else:
                qc.append(spin_interaction(time_rot), [i, k])

        # T interaction (twist)
        qc.append(
            XXPlusYYGate(-4 * dt * Jb * np.sin(twist), -np.pi / 2),
            [twist_loc, twist_loc + 1],
        )

        # b interactions
        for i in b_links:
            if Jb == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Rotations
            time_rot = dt * Jb

            # Gates
            if i == twist_loc:
                qc.append(spin_twist_interaction(time_rot, twist), [i, k])
            else:
                qc.append(spin_interaction(time_rot), [i, k])

        # a interactions
        for i in a_links:
            # No interaction
            if Ja == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = dt * Ja

            # Gates
            qc.append(spin_interaction(time_rot), [i, k])

    # Backward propagation
    for j in range(round(N / 2)):
        # Theta
        twist = np.pi + (j + 1 / 2) * dC
        # print(twist)

        # a interactions
        for i in a_links:
            # No interaction
            if Ja == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = -dt * Ja

            # Gates
            qc.append(spin_interaction(time_rot), [i, k])

        # b interactions
        for i in b_links:
            if Jb == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = -dt * Jb

            # Gates
            if i == twist_loc:
                qc.append(spin_twist_interaction(time_rot, twist), [i, k])
            else:
                qc.append(spin_interaction(time_rot), [i, k])

        # T interaction (twist)
        qc.append(
            XXPlusYYGate(-4 * -dt * Jb * np.sin(twist), -np.pi / 2),
            [twist_loc, twist_loc + 1],
        )

        # b interactions
        for i in b_links:
            if Jb == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Rotations
            time_rot = -dt * Jb

            # Gates
            if i == twist_loc:
                qc.append(spin_twist_interaction(time_rot, twist), [i, k])
            else:
                qc.append(spin_interaction(time_rot), [i, k])

        # a interactions
        for i in a_links:
            # No interaction
            if Ja == 0:
                continue

            # Next qubit index, could loop back
            k = next_qubit_index(i, L)

            # Time rotation
            time_rot = -dt * Ja

            # Gates
            qc.append(spin_interaction(time_rot), [i, k])

    return qc.to_gate()


def berry_phase_circuit(n_qubits, J, alpha, twist_loc, t0, tf, time_steps, i, N):
    # Visual logs
    # print(f"Running sim {i + 1} / {N} with J = {J} and dimer = {alpha}")
    # print("-------")

    # Numerical hamiltonian for ground state preparation
    links = interaction_chain_nn(n_qubits, J, alpha)
    H = twisted_dimerized_chain(n_qubits, links, 0, twist_loc)
    psi_0 = ground_state_optimized(H)

    # Circuit
    qc = QuantumCircuit(n_qubits + 1, 1)

    # Initial state
    qc.prepare_state(psi_0, range(1, n_qubits + 1))
    qc.barrier()

    qc.h(0)

    # Evolution
    evolution_gate = twisted_dimer_evolution(
        n_qubits, J, alpha, twist_loc, t0, tf, time_steps
    ).control(1)

    qc.append(evolution_gate, range(n_qubits + 1))

    qc.h(0)

    qc.measure(0, 0)

    simulator = AerSimulator()

    # Transpile for simulator
    circ = transpile(qc, simulator, optimization_level=3)

    # Run and get counts
    result = simulator.run(circ).result()

    zeros = result.data()["counts"].get("0x0", 0)
    ones = result.data()["counts"].get("0x1", 0)
    total = zeros + ones

    p0 = zeros / total

    berry = 2 * np.arccos(np.sqrt(p0))

    return (i, berry, circ.count_ops(), circ.depth())


# Some utilities
def print_barrier():
    print("-------------------------------------------------------------------")


def sim_start_info(n_qubits, J, alpha, N, final_time, twist_loc, threshold):
    print_barrier()
    print("STARTING BERRY PHASE CALCULATION")
    print_barrier()
    print("Calculation data:")
    print(f"-- System size: {n_qubits}")
    print(f"-- Twist location: {twist_loc}")
    print(f"-- 1st link strength: {J}")
    print(f"-- 2nd link strengths: {alpha}")
    print(f"-- Time steps: {N}")
    print(f"-- Evolving from: 0 -> {final_time} ")
    print(f"-- 0 threshold: {threshold}")
    print(f"-- Processors to use: {mp.cpu_count()}")
    print_barrier()


def sim_end_info(start_time, J, alphas, berry_phases):
    print_barrier()
    print("RESULTS")
    print_barrier()
    print("Delta:")
    print(alphas - J)
    print_barrier()
    print("Berry phases:")
    print(berry_phases)
    print_barrier()
    print("Simulation time")
    end_time = time.time()
    print(f"{end_time - start_time} seconds")
    print_barrier()


def capture_and_clear_console():
    # Redirect stdout to capture the printed output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Get the printed output as a string
    printed_output = new_stdout.getvalue()

    # Restore original stdout
    sys.stdout = old_stdout

    # Clear the console
    # Check for the platform and use the appropriate command
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix-based (Linux, macOS)
        os.system("clear")

    return printed_output


def save_string_to_file(filename, string_data):
    # Open the file in write mode ('w')
    with open(filename, "w") as file:
        # Write the string data to the file
        file.write(string_data)


if __name__ == "__main__":
    # Clear console
    capture_and_clear_console()

    # Final times and n steps
    ft_list = range(10, 21, 1)
    n_list = range(100, 210, 10)

    tf_len = len(ft_list)
    n_len = len(n_list)

    # Constants in each sim data
    n_qubits = 4
    t0 = 0
    twist_loc = 0
    J = 1
    alphas = np.arange(0, 2.1, 0.1)
    threshold = 1.0e-10

    # Comparison metrics
    alpha_len = len(alphas)
    exact_phases = np.zeros(alpha_len)
    exact_phases[int(alpha_len / 2 + 1) :] = np.pi

    mse = np.zeros((n_len, tf_len))
    gates = np.zeros((n_len, tf_len))
    depths = np.zeros((n_len, tf_len))

    # Data saving path
    root_path = "./qc-data/sweep-mixed-decomp"

    for x, tf in enumerate(ft_list):
        for y, n_steps in enumerate(n_list):
            log_file = f"{root_path}/log-{x}-{y}.txt"
            sys.stdout = open(log_file, "wt")

            # Timer
            start_time = time.time()

            # Sim data
            time_steps = n_steps

            # Data saving
            berry_phases = np.zeros(alpha_len)
            circuit_gates = np.zeros(alpha_len)
            circuit_depths = np.zeros(alpha_len)

            # Multi-processing pool (probably souldnt use all processors lol)
            pool = mp.Pool(mp.cpu_count() - 2)

            # Callback for async data ~ order it
            def async_data(result):
                i = result[0]
                phase = result[1]
                berry_phases[i] = phase
                circuit_gates[i] = sum(result[2].values())
                circuit_depths[i] = result[3]

            # Print info
            sim_start_info(n_qubits, J, alphas, time_steps, tf, twist_loc, threshold)

            # Start twist process for different link strenghts
            for i in range(alpha_len):
                pool.apply_async(
                    berry_phase_circuit,
                    args=(
                        n_qubits,
                        J,
                        alphas[i],
                        twist_loc,
                        t0,
                        tf,
                        time_steps,
                        i,
                        alpha_len,
                    ),
                    callback=async_data,
                )

            # Close processors
            pool.close()
            pool.join()

            # Print info
            sim_end_info(start_time, J, alphas, berry_phases)

            # Comparison metrics
            square_error = np.zeros(alpha_len)

            for i in range(alpha_len):
                square_error[i] = (exact_phases[i] - berry_phases[i]) ** 2

            mse[x][y] = sum(square_error) / alpha_len
            gates[x][y] = max(circuit_gates)
            depths[x][y] = max(circuit_depths)

            # Data saving
            phase_file = f"{root_path}/phase-{x}-{y}.csv"
            alpha_file = f"{root_path}/alpha-{x}-{y}.csv"
            square_error_file = f"{root_path}/sqe-{x}-{y}.csv"

            np.savetxt(phase_file, berry_phases, delimiter=",", fmt="%f")
            np.savetxt(alpha_file, alphas, delimiter=",", fmt="%f")
            np.savetxt(square_error_file, square_error, delimiter=",", fmt="%f")

    # Data saving
    gate_file = f"{root_path}/gates.csv"
    depth_file = f"{root_path}/depths.csv"
    mse_file = f"{root_path}/mses.csv"

    np.savetxt(gate_file, gates, delimiter=",", fmt="%f")
    np.savetxt(depth_file, depths, delimiter=",", fmt="%f")
    np.savetxt(mse_file, mse, delimiter=",", fmt="%f")
