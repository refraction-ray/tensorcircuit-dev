"""
One-site TDVP for H = J sum_<ij> (X_i X_j + Y_i Y_j) on an open rectangle.

Run from the repository root: python examples/one_site_tdvp_2d.py
For a quick exact comparison: add --rows 2 --cols 3 --chi 8.
The default MPS cap is chi=128; use --chi 64 for faster iteration.
X and Y are Pauli matrices. For spin-1/2 exchange with strength Js,
set coupling=Js/4 in the MPO and preparation helpers.

The snake MPO has bond dimension 2 * cols + 2. Its channels remember an X
or Y at the most recently visited site in each column, plus start/finish
channels. This represents H exactly. In contrast, MPS chi bounds the
state's Schmidt rank across each snake cut and controls projection error.
Open-boundary MPS
tensors have actual boundary dimension one, with no padded virtual states.
The local TDVP kernels are shared with the one-dimensional example and
compiled once per tensor shape; the short spatial sweeps run in Python.

A symmetric two-site gate step prepares the Neel state at t = warmup and
establishes its bond space. Subsequent evolution is strictly one-site TDVP:
it cannot increase these bond dimensions. Vary chi, dt, and warmup to check
convergence; conserved norm and energy alone do not certify accuracy.

GPU SVD diagnostic anchor: with the unpatched default JAX SVD, the 7x7,
chi=128, warmup=0.01 preparation first fails at logical gate 14, internal
adjacent operation 123, the reverse SWAP on bond (7, 8). Its finite split
matrix has shape (256, 256) and dtype complex128; compare raw GPU SVD with
the LAX QR SVD algorithm at this point.
"""

import argparse
from functools import partial
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from one_site_tdvp import (
    bond_tdvp_step,
    left_canonicalize,
    local_tdvp_step,
    right_canonicalize,
    update_L,
    update_R,
)
import tensorcircuit as tc


def snake_lattice(rows, cols):
    """Return coordinates in MPS order and each undirected lattice edge once."""
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("Use a positive rectangle containing at least two sites.")
    coordinates = [
        (r, c)
        for r in range(rows)
        for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))
    ]
    index = {rc: i for i, rc in enumerate(coordinates)}
    edges = []
    for i, (r, c) in enumerate(coordinates):
        for neighbor in ((r + 1, c), (r, c + 1)):
            if neighbor in index:
                edges.append(tuple(sorted((i, index[neighbor]))))
    return coordinates, sorted(edges)


def build_xy_mpo(rows, cols, coupling=1.0):
    """
    Build exact W[left, physical_out, physical_in, right] snake tensors.

    Channel 0 has not started a term; channel D-1 has completed it.
    Channels 1+2*c and 2+2*c carry X and Y from a source in column c.
    A source opens with J*P, propagates with I, and closes with P at each
    future neighbor. A column's channel is reused only after its old source
    has reached its last neighbor. No compression or dense H is needed.
    """
    coordinates, edges = snake_lattice(rows, cols)
    dimension = 2 * cols + 2
    tensors = np.zeros((len(coordinates), dimension, 2, 2, dimension), dtype=tc.npdtype)
    identity = np.eye(2)
    paulis = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]))
    tensors[:, 0, :, :, 0] = identity
    tensors[:, -1, :, :, -1] = identity
    for source, (_, column) in enumerate(coordinates):
        targets = [j for i, j in edges if i == source]
        if not targets:
            continue
        for kind, pauli in enumerate(paulis):
            channel = 1 + 2 * column + kind
            tensors[source, 0, :, :, channel] = coupling * pauli
            tensors[source + 1 : max(targets), channel, :, :, channel] = identity
            for target in targets:
                tensors[target, channel, :, :, -1] += pauli
    mpo = list(tensors)
    mpo[0] = mpo[0][0:1]
    mpo[-1] = mpo[-1][..., -1:]
    return tuple(tc.backend.convert_to_tensor(w) for w in mpo)


def prepare_neel_mps(rows, cols, chi, warmup, coupling=1.0):
    """
    Apply one symmetric, truncated gate step to the checkerboard Neel state.

    The preparation approximates exp(-i H warmup), with a second-order
    splitting plus MPS truncation. Its physical duration counts in the
    reported time. It avoids freezing a rank-one product state under TDVP.
    """
    if chi < 2 or warmup <= 0:
        raise ValueError("Use chi >= 2 and warmup > 0 to initialize the bond space.")
    coordinates, edges = snake_lattice(rows, cols)
    circuit = tc.MPSCircuit(len(coordinates), split={"max_singular_values": chi})
    for i, (r, c) in enumerate(coordinates):
        if (r + c) % 2 == 0:
            circuit.x(i)
    x, y = tc.gates.x().tensor, tc.gates.y().tensor
    hbond = coupling * (tc.backend.kron(x, x) + tc.backend.kron(y, y))
    gate = tc.gates.exp(unitary=hbond, theta=warmup / 2)
    for i, j in edges + edges[::-1]:
        circuit.apply_general_gate(gate, i, j)
    circuit.position(0)
    mps = list(circuit.get_tensors())
    mps[0] = mps[0] / tc.backend.norm(mps[0])
    return tuple(mps)


def make_tdvp_step(mpo, krylov_dim=12):
    """
    Return a symmetric one-site step for a right-canonical open MPS.

    Local center tensors evolve forward and intervening bond centers evolve
    backward. The returned MPS again has its orthogonality center at site 0.
    Boundary shapes taper naturally, so QR uses the physical Hilbert space.
    """
    if krylov_dim < 2:
        raise ValueError("Use at least two Krylov vectors.")
    site_step = tc.backend.jit(partial(local_tdvp_step, krylov_dim=krylov_dim))
    bond_step = tc.backend.jit(partial(bond_tdvp_step, krylov_dim=krylov_dim))
    left_qr = tc.backend.jit(left_canonicalize)
    right_qr = tc.backend.jit(right_canonicalize)
    left_env = tc.backend.jit(update_L)
    right_env = tc.backend.jit(update_R)
    boundary = tc.backend.ones((1, 1, 1), dtype=tc.dtypestr)

    def step(mps, dt):
        mps = list(mps)
        n = len(mps)
        right = [None] * n + [boundary]
        for i in range(n - 1, 0, -1):
            right[i] = right_env(right[i + 1], mpo[i], mps[i])
        left = [boundary]
        for i in range(n - 1):
            center = site_step(left[i], mpo[i], right[i + 1], mps[i], dt / 2)
            mps[i], bond = left_qr(center)
            left.append(left_env(left[i], mpo[i], mps[i]))
            bond = bond_step(left[i + 1], right[i + 1], bond, dt / 2)
            mps[i + 1] = tc.backend.einsum("ab,bcd->acd", bond, mps[i + 1])
        mps[-1] = site_step(left[-1], mpo[-1], boundary, mps[-1], dt)
        env = boundary
        for i in range(n - 1, 0, -1):
            bond, mps[i] = right_qr(mps[i])
            env = right_env(env, mpo[i], mps[i])
            bond = bond_step(left[i], env, bond, dt / 2)
            center = tc.backend.einsum("abc,cd->abd", mps[i - 1], bond)
            mps[i - 1] = site_step(left[i - 1], mpo[i - 1], env, center, dt / 2)
        return tuple(mps)

    return step


def measure(mps, mpo):
    """Compute norm, energy, and local Z without forming a full statevector."""
    boundary = tc.backend.ones((1, 1, 1), dtype=tc.dtypestr)
    env = boundary
    for a, w in zip(mps, mpo):
        env = update_L(env, w, a)
    energy = tc.backend.real(env[0, 0, 0])
    # The right-canonical suffix contracts to identity at every site.
    norm = tc.backend.real(tc.backend.sum(tc.backend.conj(mps[0]) * mps[0]))
    density = tc.backend.ones((1, 1), dtype=tc.dtypestr)
    z = tc.gates.z().tensor
    magnetization = []
    for a in mps:
        magnetization.append(
            tc.backend.real(
                tc.backend.einsum("ab,asr,st,btr->", density, tc.backend.conj(a), z, a)
            )
            / norm
        )
        density = tc.backend.einsum("ab,asr,bst->rt", density, tc.backend.conj(a), a)
    return norm, energy / norm, tc.backend.stack(magnetization)


def mps_to_state(mps):
    """Small-system reference only: contract an open MPS to a statevector."""
    state = mps[0][0]
    for a in mps[1:]:
        state = tc.backend.reshape(
            tc.backend.einsum("pa,asb->psb", state, a), (-1, a.shape[2])
        )
    return state[:, 0]


def exact_xy_hamiltonian(rows, cols, coupling=1.0):
    """Sparse reference: XX+YY exchanges unequal bits with amplitude 2J."""
    n = rows * cols
    if n > 12:
        raise ValueError("The statevector reference is restricted to at most 12 sites.")
    _, edges = snake_lattice(rows, cols)
    basis = np.arange(2**n)
    sources, targets = [], []
    for i, j in edges:
        bi, bj = 1 << (n - 1 - i), 1 << (n - 1 - j)
        selected = basis[((basis & bi) != 0) != ((basis & bj) != 0)]
        sources.extend(selected)
        targets.extend(selected ^ bi ^ bj)
    return coo_matrix(
        (np.full(len(sources), 2 * coupling), (targets, sources)), shape=(2**n, 2**n)
    ).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--chi", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.01)
    args = parser.parse_args()
    if args.dt <= 0 or args.steps < 1:
        parser.error("dt and steps must be positive")
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    coordinates, _ = snake_lattice(args.rows, args.cols)
    mpo = build_xy_mpo(args.rows, args.cols)
    start = time.perf_counter()
    mps = prepare_neel_mps(args.rows, args.cols, args.chi, args.warmup)
    tc.backend.numpy(mps[0])
    print(f"Preparation: {time.perf_counter() - start:.3f}s")
    print(
        f"{args.rows}x{args.cols}, MPO D={2 * args.cols + 2}, "
        f"MPS max chi={max(a.shape[2] for a in mps)}, J=1 (Pauli convention)"
    )
    step = make_tdvp_step(mpo)
    observe = tc.backend.jit(measure)
    signs = np.array([(-1) ** (r + c) for r, c in coordinates])
    exact = None
    if len(coordinates) <= 12:
        hamiltonian = exact_xy_hamiltonian(args.rows, args.cols)
        index = sum(
            1 << (len(coordinates) - 1 - i) for i, s in enumerate(signs) if s == 1
        )
        initial = np.eye(1, 2 ** len(coordinates), index, dtype=complex).ravel()
        exact = expm_multiply(-1j * args.warmup * hamiltonian, initial)
    print(
        "time       norm          energy         mean Z       staggered Z    step seconds"
    )
    for k in range(args.steps + 1):
        elapsed = 0.0
        if k:
            start = time.perf_counter()
            mps = step(mps, args.dt)
            tc.backend.numpy(mps[0])
            elapsed = time.perf_counter() - start
            if exact is not None:
                exact = expm_multiply(-1j * args.dt * hamiltonian, exact)
        norm, energy, z = (tc.backend.numpy(x) for x in observe(mps, mpo))
        print(
            f"{args.warmup + k * args.dt:7.3f}  {norm:12.9f}  {energy:13.6e}  "
            f"{np.mean(z):12.6e}  {np.mean(signs * z):12.8f}  {elapsed:10.3f}"
        )
        if exact is not None:
            state = tc.backend.numpy(mps_to_state(mps))
            fidelity = abs(np.vdot(exact, state)) ** 2 / norm
            print(f"  Fidelity to exact Neel evolution: {fidelity:.10f}")
    grid = np.empty((args.rows, args.cols))
    for rc, value in zip(coordinates, z):
        grid[rc] = value
    print("Final <Z> on the original lattice:")
    print(np.array2string(grid, precision=4, suppress_small=True))
    print("The first TDVP step includes compilation; later steps reuse local kernels.")


if __name__ == "__main__":
    main()
