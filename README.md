# resource-superstaq
Infleqtion's resource estimation project for fault tolerant quantum computing

## Example Usage
```python
import cirq
import resource_estimation as res

# Prepare Input Circuit
qubits = 3
circuit = cirq.Circuit(
    cirq.MatrixGate(
        cirq.testing.random_unitary(dim=2**qubits, random_state=7)
    ).on(*cirq.LineQubit.range(qubits))
)

# Two Stage Compile to Clifford + T
cliff_rz_circuit = res.cliff_rz.compile_cliff_rz(circuit)
cliff_t_circuit = res.clifford_t.compile_cirq_to_clifford_t(cliff_rz_circuit, eps=.001)

# Prepare Architecture and Layout
arch = res.architecture.DefaultMovement(d=11)
layout = res.layout.MovementLayout(input_circuit=cliff_t_circuit, num_t_factories=5)

# FT Compile
primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=True)

# Estimate Resources
estimator = res.estimate.ResourceEstimator(arc=arch)
gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=True)
circuit_time = estimator.parallel_circuit_time(primitive_circuit)
physical_qubits = estimator.physical_qubits(primitive_circuit)
```

## Getting Started
Check out the [example notebook](https://github.com/Infleqtion/resource-superstaq/blob/main/notebooks/hello_estimate.ipynb)

## Architectures
This table describes general gate implementation
Name | Moniker | S style | T Style | CNOT Style | Correlated Decoding
|----------|----------|----------|----------|----------|----------|
|Single Species Movement | SSM | Fold Transversal | Yale or Gidney | Transversal | Yes
|Dual Species No Movement | DSNM | Gidney Teleportation | Gidney | Lattice Surgery | No
|Dual Species with Movement | DSM | Fold Transversal | Yale or Gidney | Transversal | Yes
|Measure Zones Only | MZO | Fold Transversal | Yale or Gidney | Transversal | Yes
|Superconducting | SSOQ | Gidney Teleportation | Gidney | Lattice Surgery | No

This table describes the difference between movement architectures. CZ here is different from CNOT above to indicate that these are physical operations rather than logical ones.
Name | Readout Style | Readout Cost | CZ Style | CZ Cost | 
|----|---------------|--------------|----------|---------|
| SSM | Readout Zone| 2 Moves | Interaction Zone | 2 Moves |
| MZO | Readout Zone | 2 Moves | In Place Entanglement | 1 Move<sup>*</sup> |
| DSM | In Place Readout | 0 Moves | In Place Entanglement | 1 Move<sup>*</sup> |

<sup>*</sup> Moves happen between alleys and interactions happen between nearest neighbor qubits, so some interactions are cheap, and others are expensive depending on the situation. In this context 1 is just an approximation.

This table describes the expected number of repeats of the T state cultivation circuit for two cultivation styles and two error levels. The repetition factor describes the number of times we expect to run the cultivation circuit (end to end) before successfully preparing a T state of the desired fidelity.
Cultivation Style | Logical Error Rate | Repetition Factor |
|-----------------|--------------------|-------------------|
| Gidney (nearest neighbor) | 10<sup>-6</sup> | 5 |
| Gidney (nearest neighbor) | 10<sup>-9</sup> | 100 |
| Yale (nonlocal) | 10<sup>-6</sup> | 1 |
| Yale (nonlocal) | 10<sup>-9</sup> | 10 |
