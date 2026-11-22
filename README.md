# Hybrid Quantum–Classical Catastrophe Bond Pricing via Space-Time Block Encoding and QSVT

A prototype implementation of a hybrid quantum–classical solver for multi-country, multi-peril catastrophe bond pricing. The framework unrolls the Forward Kolmogorov Equation into a global space-time linear system ($\mathcal{A}\vert{}\Pi\rangle = \vert{}b\rangle$), encodes the non-unitary operator $\mathcal{A}$ via an SVD-free Linear Combination of Unitaries (LCU) (`PREPARE` + `SELECT`), and wraps the block encoding in a Quantum Singular Value Transformation (QSVT) polynomial sequence.

---
* `Quantum-Classical.py`
* `requirements.txt`

---

## Environment Specifications and Dependencies

* **Operating System:** Windows 10/11, macOS, or Linux
* **Python Version:** Python 3.10 or higher
* **Core Libraries:**
  * `qiskit >= 1.0.0`
  * `qiskit-aer >= 0.13.0`
  * `numpy >= 1.24.0`
  * `scipy >= 1.10.0`

---

AI was used in formulating the codes
