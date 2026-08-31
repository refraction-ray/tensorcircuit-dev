"""
Reproduction of "Towards Practical Quantum Neural Network Diagnostics with Neural Tangent Kernels"
Link: https://arxiv.org/abs/2503.01966

Description:
This script reproduces Figure 2 from the paper using TensorCircuit-NG.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from matplotlib import pyplot as plt
import numpy as np
import tensorcircuit as tc

plt.switch_backend("Agg")

NQUBITS = 6
# The paper extends beyond the visible 620-parameter axis; this gallery artifact
# keeps the six depth points that fit within that plotted range.
DEPTHS = np.arange(5, 31, 5, dtype=np.int64)
SEEDS = range(3)
BACKENDS = ("jax", "pytorch")
DTYPE = "complex128"
ENCODING_SCALE = np.pi
RCOND = 1.0e-10
ARCHITECTURES = ("low_hva", "high_hva", "low_hea", "high_hea")
METRICS = ("lambda_min", "inverse_lambda_max", "inverse_condition", "r2")

FIELDS = np.arange(-5.0, 5.0, 0.5, dtype=np.float64)
SCALED_INPUTS = np.linspace(-0.95, 0.95, FIELDS.size, dtype=np.float64)
TRAIN_FIELDS = np.array(
    [-5.0, -3.5, -2.5, -1.0, -0.5, 0.0, 0.5, 1.5, 4.0, 4.5],
    dtype=np.float64,
)
TRAIN_INDICES = np.flatnonzero(np.isin(FIELDS, TRAIN_FIELDS))
TEST_INDICES = np.flatnonzero(~np.isin(FIELDS, TRAIN_FIELDS))

# The paper does not publish its approximate VQE labels as a table. These
# values are digitized from the marker centers in the vector datasets figure.
LABELS = np.array(
    [
        -0.823305712,
        -0.821116171,
        -0.817501222,
        -0.813012704,
        -0.804953841,
        -0.791620098,
        -0.767481527,
        -0.711551981,
        -0.542529333,
        -0.258214671,
        0.002376346,
        0.256895775,
        0.549635215,
        0.711407665,
        0.767629197,
        0.792108976,
        0.804991024,
        0.812604221,
        0.817524780,
        0.820904512,
    ],
    dtype=np.float64,
)

STYLES = {
    "low_hva": {"color": "#482060", "linestyle": "--", "label": r"low-$\omega$ + HVA"},
    "high_hva": {"color": "#922b6c", "linestyle": "-", "label": r"high-$\omega$ + HVA"},
    "low_hea": {"color": "#d84b5f", "linestyle": "--", "label": r"low-$\omega$ + HEA"},
    "high_hea": {"color": "#ed845b", "linestyle": "-", "label": r"high-$\omega$ + HEA"},
}


def parse_args() -> argparse.Namespace:
    """Parse the backend used for the reproduction."""

    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2 with a TensorCircuit-NG backend."
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="pytorch",
        help="Backend for the reproduction (default: pytorch).",
    )
    return parser.parse_args()


def parameter_count(architecture: str, depth: int) -> int:
    """Return the trainable parameter count for one circuit."""

    ansatz = architecture.rsplit("_", maxsplit=1)[1]
    parameters_per_layer = 2 * NQUBITS if ansatz == "hva" else 3 * NQUBITS
    return parameters_per_layer * depth


def initialize_parameters(seed: int, count: int) -> np.ndarray:
    """Draw one reproducible Uniform(0, 2*pi) initialization."""

    generator = np.random.default_rng(seed)
    return generator.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float64)


def apply_encoding(
    circuit: tc.Circuit,
    x_value: Any,
    depth: int,
    layer: int,
    frequency: str,
) -> None:
    """Apply the low- or high-frequency data embedding."""

    if frequency == "low":
        angle = ENCODING_SCALE * x_value / (depth - layer)
        for qubit in range(NQUBITS):
            circuit.rx(qubit, theta=angle)
        return
    for qubit in range(NQUBITS):
        circuit.ry(qubit, theta=ENCODING_SCALE * x_value)
        circuit.rz(qubit, theta=ENCODING_SCALE * x_value * x_value)


def apply_cnot_ring(circuit: tc.Circuit) -> None:
    """Apply the periodic nearest-neighbor CNOT ring."""

    for qubit in range(NQUBITS):
        circuit.cnot(qubit, (qubit + 1) % NQUBITS)


def apply_hva(circuit: tc.Circuit, layer_parameters: Any) -> None:
    """Apply one Hamiltonian variational ansatz layer."""

    for qubit in range(NQUBITS):
        circuit.rx(qubit, theta=layer_parameters[0, qubit])
    for qubit in range(NQUBITS):
        circuit.rzz(
            qubit,
            (qubit + 1) % NQUBITS,
            theta=layer_parameters[1, qubit],
        )


def apply_hea(circuit: tc.Circuit, layer_parameters: Any) -> None:
    """Apply one hardware-efficient ansatz layer."""

    rotations = (circuit.rx, circuit.rz, circuit.rx)
    for sublayer, rotation in enumerate(rotations):
        for qubit in range(NQUBITS):
            rotation(qubit, theta=layer_parameters[sublayer, qubit])
        apply_cnot_ring(circuit)


def qnn_output(
    parameters: Any,
    x_value: Any,
    depth: int,
    architecture: str,
) -> Any:
    """Evaluate the representative first-qubit transverse-X expectation."""

    frequency, ansatz = architecture.split("_")
    ansatz_width = 2 if ansatz == "hva" else 3
    shaped_parameters = tc.backend.reshape(parameters, [depth, ansatz_width, NQUBITS])
    circuit = tc.Circuit(NQUBITS)
    for layer in range(depth):
        apply_encoding(circuit, x_value, depth, layer, frequency)
        if ansatz == "hva":
            apply_hva(circuit, shaped_parameters[layer])
        else:
            apply_hea(circuit, shaped_parameters[layer])
    return tc.backend.real(circuit.expectation_ps(x=[0]))


def batch_outputs(
    parameters: Any,
    inputs: Any,
    depth: int,
    architecture: str,
) -> Any:
    """Evaluate a shared-parameter QNN on all scalar inputs."""

    def scalar_output(parameter_values: Any, x_value: Any) -> Any:
        return qnn_output(parameter_values, x_value, depth, architecture)

    mapped_output = tc.backend.vmap(scalar_output, vectorized_argnums=1)
    return tc.backend.jit(mapped_output)(parameters, inputs)


@contextlib.contextmanager
def runtime_context(backend: str) -> Iterator[Any]:
    """Set the selected backend and precision for one complete sweep."""

    with tc.runtime_backend(backend) as active_backend, tc.runtime_dtype(DTYPE):
        yield active_backend


def build_transforms(
    active_backend: Any, depth: int, architecture: str
) -> tuple[Callable[[Any, Any], Any], Callable[[Any, Any], Any]]:
    """Build JIT/vectorized output and Jacobian transforms for one circuit shape."""

    def scalar_output(parameters: Any, x_value: Any) -> Any:
        return qnn_output(parameters, x_value, depth, architecture)

    output_function = active_backend.jit(
        active_backend.vmap(scalar_output, vectorized_argnums=1)
    )
    if active_backend.name == "pytorch":
        # TCNG 1.9.1's generic jacrev is incompatible with this PyTorch functorch path.
        # For this scalar output, grad returns the same parameter Jacobian.
        scalar_jacobian = active_backend.grad(scalar_output, argnums=0)
    else:
        scalar_jacobian = active_backend.jacrev(scalar_output, argnums=0)
    jacobian_function = active_backend.jit(
        active_backend.vmap(scalar_jacobian, vectorized_argnums=1)
    )
    return output_function, jacobian_function


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Return the coefficient of determination."""

    residual = np.sum(np.square(targets - predictions))
    total = np.sum(np.square(targets - np.mean(targets)))
    return float(1.0 - residual / total)


def kernel_diagnostics(jacobian: np.ndarray) -> dict[str, float]:
    """Compute the four diagnostics plotted in Figure 2."""

    train_jacobian = jacobian[TRAIN_INDICES]
    test_jacobian = jacobian[TEST_INDICES]
    kernel = train_jacobian @ train_jacobian.T
    kernel = 0.5 * (kernel + kernel.T)
    eigenvalues = np.linalg.eigvalsh(kernel)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1.0e-10 * scale:
        raise ValueError(f"QNTK is not PSD: minimum eigenvalue {eigenvalues[0]}")
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])
    inverse = np.linalg.pinv(kernel, rcond=RCOND, hermitian=True)
    cross_kernel = test_jacobian @ train_jacobian.T
    predictions = cross_kernel @ inverse @ LABELS[TRAIN_INDICES]
    return {
        "lambda_min": lambda_min,
        "inverse_lambda_max": 1.0 / lambda_max,
        "inverse_condition": lambda_min / lambda_max,
        "r2": r2_score(LABELS[TEST_INDICES], predictions),
    }


def run_sweep(backend: str) -> dict[str, dict[str, np.ndarray]]:
    """Run all four architectures, six depths, and three initializations."""

    results: dict[str, dict[str, np.ndarray]] = {}
    with runtime_context(backend) as active_backend:
        input_tensor = active_backend.convert_to_tensor(SCALED_INPUTS, dtype="float64")
        transforms: dict[
            tuple[str, int], tuple[Callable[[Any, Any], Any], Callable[[Any, Any], Any]]
        ] = {}
        for architecture in ARCHITECTURES:
            architecture_metrics = {
                metric: np.empty((len(SEEDS), DEPTHS.size), dtype=np.float64)
                for metric in METRICS
            }
            for depth_index, depth_value in enumerate(DEPTHS):
                depth = int(depth_value)
                transform_key = architecture, depth
                transforms[transform_key] = build_transforms(
                    active_backend, depth, architecture
                )
                output_function, jacobian_function = transforms[transform_key]
                count = parameter_count(architecture, depth)
                for seed in SEEDS:
                    start = time.perf_counter()
                    parameters = initialize_parameters(seed, count)
                    parameter_tensor = active_backend.convert_to_tensor(
                        parameters, dtype="float64"
                    )
                    outputs = output_function(parameter_tensor, input_tensor)
                    jacobian = jacobian_function(parameter_tensor, input_tensor)
                    output_values = np.asarray(
                        active_backend.numpy(outputs), dtype=np.float64
                    )
                    if output_values.shape != SCALED_INPUTS.shape:
                        raise ValueError(
                            f"Unexpected output shape {output_values.shape}; "
                            f"expected {SCALED_INPUTS.shape}."
                        )
                    diagnostics = kernel_diagnostics(
                        np.asarray(active_backend.numpy(jacobian), dtype=np.float64)
                    )
                    for metric in METRICS:
                        architecture_metrics[metric][seed, depth_index] = diagnostics[
                            metric
                        ]
                    elapsed = time.perf_counter() - start
                    print(
                        f"{architecture}: depth={depth} seed={seed} "
                        f"R2={diagnostics['r2']:.6f} elapsed={elapsed:.2f}s",
                        flush=True,
                    )
            results[architecture] = architecture_metrics
    return results


def draw_break_marks(top: Any, bottom: Any) -> None:
    """Draw diagonal marks for a broken vertical axis."""

    diagonal = 0.018
    style = {"color": "#333333", "clip_on": False, "linewidth": 0.9}
    top.plot(
        (-diagonal, diagonal), (-diagonal, diagonal), transform=top.transAxes, **style
    )
    top.plot(
        (1 - diagonal, 1 + diagonal),
        (-diagonal, diagonal),
        transform=top.transAxes,
        **style,
    )
    bottom.plot(
        (-diagonal, diagonal),
        (1 - diagonal, 1 + diagonal),
        transform=bottom.transAxes,
        **style,
    )
    bottom.plot(
        (1 - diagonal, 1 + diagonal),
        (1 - diagonal, 1 + diagonal),
        transform=bottom.transAxes,
        **style,
    )


def plot_results(results: dict[str, dict[str, np.ndarray]]) -> None:
    """Render the four Figure 2 diagnostics."""

    labels = (
        r"$\lambda_{\min}$",
        r"$\lambda_{\max}^{-1}$",
        r"$\kappa^{-1}$",
        r"QNTK test $R^2$",
    )
    figure = plt.figure(figsize=(16.0, 4.7))
    grid = figure.add_gridspec(
        2,
        4,
        left=0.065,
        right=0.99,
        bottom=0.17,
        top=0.80,
        hspace=0.08,
        wspace=0.35,
    )
    axis_a_top = figure.add_subplot(grid[0, 0])
    axis_a_bottom = figure.add_subplot(grid[1, 0], sharex=axis_a_top)
    axis_b = figure.add_subplot(grid[:, 1])
    axis_c_top = figure.add_subplot(grid[0, 2])
    axis_c_bottom = figure.add_subplot(grid[1, 2], sharex=axis_c_top)
    axis_d = figure.add_subplot(grid[:, 3])
    metric_axes = (
        (axis_a_top, axis_a_bottom),
        (axis_b,),
        (axis_c_top, axis_c_bottom),
        (axis_d,),
    )
    for architecture in ARCHITECTURES:
        style = STYLES[architecture]
        x_values = np.asarray(
            [parameter_count(architecture, int(depth)) for depth in DEPTHS]
        )
        for axes, metric in zip(metric_axes, METRICS):
            mean = np.mean(results[architecture][metric], axis=0)
            standard_deviation = np.std(results[architecture][metric], axis=0)
            for axis_index, axis in enumerate(axes):
                axis.plot(
                    x_values,
                    mean,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2,
                    marker="o",
                    markersize=3.4,
                    label=(
                        style["label"]
                        if metric == "lambda_min" and axis_index == 0
                        else None
                    ),
                )
                axis.fill_between(
                    x_values,
                    mean - standard_deviation,
                    mean + standard_deviation,
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0.0,
                )
    top_axes = (axis_a_top, axis_b, axis_c_top, axis_d)
    for index, (axis, label) in enumerate(zip(top_axes, labels)):
        axis.set_ylabel(label)
        axis.text(
            -0.08,
            1.05,
            f"{chr(ord('a') + index)})",
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
        )
    all_axes = (
        axis_a_top,
        axis_a_bottom,
        axis_b,
        axis_c_top,
        axis_c_bottom,
        axis_d,
    )
    for axis in all_axes:
        axis.set_xlim(0.0, 620.0)
        axis.grid(alpha=0.22)
    for axis in (axis_a_bottom, axis_b, axis_c_bottom, axis_d):
        axis.set_xlabel("trainable parameters")
    axis_a_bottom.set_ylim(0.0, 0.5)
    axis_a_top.set_ylim(0.5, 5.8)
    axis_b.set_ylim(bottom=0.0)
    axis_c_bottom.set_ylim(0.0, 0.05)
    axis_c_top.set_ylim(0.05, 0.72)
    axis_d.set_ylim(-0.15, 1.08)
    axis_d.axhline(0.0, color="#555555", linewidth=0.8)
    for top, bottom in ((axis_a_top, axis_a_bottom), (axis_c_top, axis_c_bottom)):
        top.spines["bottom"].set_visible(False)
        bottom.spines["top"].set_visible(False)
        top.tick_params(labeltop=False, bottom=False, labelbottom=False)
        bottom.xaxis.tick_bottom()
        draw_break_marks(top, bottom)
    figure.legend(
        *axis_a_top.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=4,
        frameon=False,
    )
    output_path = Path(__file__).resolve().parent / "outputs" / "result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_path}")


def main() -> None:
    """Configure the backend, run the sweep, and save the figure."""

    args = parse_args()
    print(f"TensorCircuit backend={args.backend}")
    plot_results(run_sweep(args.backend))


if __name__ == "__main__":
    main()
