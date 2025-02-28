# Reflections and Regrets

Looking back, I feel a sense of regret. Many years ago, I implemented a solution for multi-pattern regex matching on multicore CPUs. At that time, adapting the algorithm for GPU computation required extensive modifications and effort. However, as I see the current state of research—like in the HybridSA paper—the need for those heavy modifications on GPUs has largely diminished.

It's bittersweet: I once invested significant time optimizing for CPU architectures, and now the landscape has shifted. While I wish I had ventured into GPU-based solutions sooner, I’m also grateful to see that the innovations in GPU computation have simplified what once was a daunting challenge.

---

# Code_Generation_For_DNA_Storage

This GitHub repository is dedicated to exploring and implementing a Codebook generation method for DNA storage. Currently, it serves as a repository for storing experimental data and code used for verification purposes.

The arXiv link to the article is as follows: [Unrestricted Error-Type Codebook Generation for Error Correction Code in DNA Storage Inspired by NLP](https://arxiv.org/abs/2401.15915) 
## Purpose

The main goal of this project is to develop a robust encoding method for DNA storage that can effectively correct errors arising from substitutions, insertions, and deletions simultaneously. The emphasis is on the generation of a codebook that enables error correction in the context of DNA data storage.

## Contents
- **Data:** Store some experimental results and comparative data, as well as codebooks for other adversarial scenarios.
- **Codebook Generation Scheme** The initial version of the code for ECG and codebook generation in the research paper. This folder will not be made public until the paper is published. 
 

## License

This project is released under the MIT License - see the [LICENSE](LICENSE) file for details.

If you are interested in collaborating or have suggestions, please：
[2230501004@cnu.edu.cn](mailto:2230501004@cnu.edu.cn)
