# Version3 Code Overview

## Introduction
This document provides an overview of the code in the `version3` directory. The code is primarily written in Python and involves various functionalities related to ECG (Electrocardiogram) data processing and sequence matching.

## Files and Their Purpose

### `ECG.py`
This file contains the main logic for processing ECG data. It includes functions for initializing matrices, matching sequences, and converting decimal values to binary. The main function demonstrates how to use these functionalities with a sample codebook and sequence.

#### Key Functions:
- `decode_line(seqs, codebook, ccs)`: Decodes a given sequence using a specified codebook and correction feature sequence.
- `seq_match(item, codebook, residue, chStr, mask, shiftAmount)`: Matches a sequence item with the codebook and updates the residue matrix.
- `decimal_to_binary(a, bit_len)`: Converts a decimal number to a binary string of specified bit length.
- `cgg(booksize, increment_option, dep, wid, arrlen, p_s, i_l, m_l, msks, s_a)`: Generates a codebook by iteratively augmenting sequences and updating matrices until a stopping condition is met.
 
#### Key Sections:
- Matrix Indexing: Functions and examples for indexing and manipulating matrices.
- PyTorch Operations: Code for generating large datasets and testing data transfer times between CPU and GPU.

## Usage
To run the main ECG processing script, execute the `ECG.py` file. Ensure that the necessary dependencies, such as PyTorch, are installed in your environment.

```bash
python ECG.py
