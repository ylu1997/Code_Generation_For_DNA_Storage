# Version3 Code Overview

## Introduction
This document provides an overview of the code in the `version3` directory. The code is primarily written in Python and involves various functionalities related to ECG algorithm and codebook generation.

## Files and Their Purpose

### `BaseUtils.py`
This file contains basic utility functions that are used across the project.

### `CFS.py`
This file is responsible for preprocessing and editing information. It includes functions for serializing and deserializing data structures used in the project.

### `ECG_mat.py`
This file deals with the dynamic programming matrix used in ECG. It includes the `ECG_Matrix` class and related methods for matrix operations.

### `ECG_alg.py`
This file contains algorithms for ECG.

#### Key Functions:
- `augment_ecg(seq1, seq2, augAmount, mat, startRow, s_a, p_s, i_l, m_l, msks)`: Augments ECG sequences and updates the matrix.
- `batch_seq_match(seq, codebook, residue, s_a, p_s, i_l, m_l, msks)`: Matches a sequence against a codebook in parallel and returns the best match.

### `ECG_CodeGen.py`
This file contains the implementation of the codebook generation algorithm for ECG.

#### Key Functions:
- `cgg(booksize, increment_option, dep, wid, arrlen, p_s, i_l, m_l, msks, s_a)`: Generates a codebook by iteratively augmenting sequences and updating matrices until a stopping condition is met.

## Usage
To run the main ECG processing script, execute the `ECG.py` file. Ensure that the necessary dependencies, such as PyTorch, are installed in your environment.

```bash
python ECG.py
