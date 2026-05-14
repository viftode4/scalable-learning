# Notebooks

Cheap, fast exploration. The MNIST 2-layer reproduction of Figure 2 belongs here — runs in minutes on a laptop, no FL framework, just a Python loop with N model copies and manual averaging. It is the cleanest validation of the core RoLoRA mechanism and should pass before any GPU-hour is spent.

Heavier reproduction work (RoBERTa-Large / Llama) does **not** belong here — use `experiments/` + `slurm/` for that.
