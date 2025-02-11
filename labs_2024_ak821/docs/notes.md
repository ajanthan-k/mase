# Lab 0

Canonical training parameters
- batch_size: how many inputs to proceed concurrently
- learning_rate: the critical parameter for your optimizer
- max_epochs: maximum number of epochs to train
- dropout: the dropout rate (regularisation strength)

LoRA parameters
- rank: the rank of the low-rank matrices
- alpha: the weight of the adapter

Quantisation
- Quantisation Aware Training (QAT): quantise first, then train => try recover some accuracy loss
- Post Training quantisation: train normally and quantise after

Pruning
- Unstructured pruning: remove inidividual weights from mode
- Structured pruning: remove entire neurons / layers / channels
Unstructured typically less popular due to (limited) hardware support for sparse computation as is required.
