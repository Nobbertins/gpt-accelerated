Project still in progress.

I don’t know much about AI, but I came across Andrej Karpathy’s [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) a few months ago and I thought it was really cool. He implements GPT-2 (one of the first LLMs) and runs training and inference on it in only 200 lines of Python with no external libraries. That last part is my favorite: everything was done from scratch. This lets someone like me who can read Python but doesn’t know much about AI fully understand every small detail about how an LLM works fundamentally.

I wanted to combine this knowledge with my current interest in systems and hardware, so rather than explore more powerful model architectures, I want to try to accelerate GPT-2. I came up with 4 possible avenues to explore: 
- (C++ implementation on CPU) maximizing CPU performance with a C++ implementation
- (write CUDA kernels to use GPUs) use CUDA with the C++ implementation to run intensive operations on a GPU
- (design an accelerator on an FPGA) write an accelerator in Verilog to run intensive operations on an FPGA
- (use Qiskit to use a QNN in the algorithm) Use a QNN for part of GPT-2 and run on Quantum Inspire’s quantum computers
