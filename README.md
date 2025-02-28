# Reflections and Regrets

Looking back, I feel a deep sense of regret. Many years ago, I developed an algorithm for multi-pattern regex matching that ran solely on multicore CPUs. At that time, adapting it for GPU computation meant making extensive modifications and facing many challenges. Now, with the advances shown in papers like [HybridSA: GPU Acceleration of Multi-pattern Regex Matching using Bit Parallelism](https://dl.acm.org/doi/pdf/10.1145/3689771), I realize that much of the heavy work needed for GPU adaptation has been significantly simplified.

In a way, it's bittersweet—I spent so much effort optimizing for CPU architectures when the field has now moved forward. I can't help but wish I had explored GPU-based solutions earlier. Nonetheless, I’m grateful to see these innovations and to learn that the difficulties I once faced are no longer as daunting.


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
