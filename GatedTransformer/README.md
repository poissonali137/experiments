# Gated Transformer Model

- This idea behind this was a simple one. Similar to how parts of the brain selectively (primarily hippocampus and prefrontal cortex) communicate with each other with a base frequency + a phase on top, I was wondering what would happen if we do the same thing with transformers if we artificially impose a gating mechanism that is a learnable function of the hidden states of a layer, allowing it to selectively contribute based on whether or not this is useful.
- The results were actually straightforward, no major improvement over baseline GPT2 loss but no decrease either! I suspect something like this is much more useful in MoE models.
