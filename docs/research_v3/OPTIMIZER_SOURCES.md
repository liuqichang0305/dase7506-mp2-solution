# Optimizer sources and implementation contract

Accessed 2026-10-30, Asia/Shanghai. These are training-method sources, not additional training data or pretrained assets.

## Primary evidence actually read

Yury Gorishniy, Ivan Rubachev, Dmitrii Feoktistov and Artem Babenko, **Benchmarking Optimizers for MLPs in Tabular Deep Learning**, arXiv:2604.15297v1, 16 April 2026, preprint. We read the relevant paper body: §3.1 (separate optimizer tuning), §3.2–3.3 (architecture dependence), §4 (cost/limitations), Appendix A.1 (Muon plus EMA), and Appendix C.4/Table 8 (optimizer spaces). Its tabular benchmark motivates testing Muon and EMA; it does not establish an MP2 gain. The reported recipe independently tunes optimizer learning rates, and EMA is less consistent on advanced architectures. Our task loss, narrow architectures, nested question folds and finite budget are adaptations. [Paper body](https://arxiv.org/html/2604.15297v1)

Author repository actual commit: **`c34e33152db9eea8ed4b2d742b9bef5628d71fb5`**, verified with `git ls-remote` and pinned GitHub reads. License: **Apache-2.0**. We read `README.md`, `pyproject.toml`, and relevant implementation in `bin/ffn.py`: parameter grouping at lines 166–278, EMA creation at 442–450, update order at 584–586, and checkpoint state. Hidden backbone matrices receive Muon; head, biases, normalization and feature embedding parameters use auxiliary updates. The actual training code averages buffers and updates EMA after the optimizer. The README sketch contains undefined Muon settings, so it was not copied as runnable code. We reuse the design, not the benchmark runner, datasets, or vendored optimizer source. [Pinned implementation](https://github.com/yandex-research/tabular-dl-optimizers/blob/c34e33152db9eea8ed4b2d742b9bef5628d71fb5/bin/ffn.py), [license](https://github.com/yandex-research/tabular-dl-optimizers/blob/c34e33152db9eea8ed4b2d742b9bef5628d71fb5/LICENSE)

Installed dependency: **PyTorch `2.15.0+cpu`**, source commit **`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`** from `torch.version.git_version`. The official stable documentation redirected to version 2.15; the web reader could not render that page, so the installed class signature, docstring, `_adjust_lr` source and EMA utility were inspected directly. Native `torch.optim.Muon` is available and was actually exercised on CPU. This is an explicit native-PyTorch adaptation, not a claim to reproduce the authors' vendored implementation byte for byte. [Muon docs](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Muon.html), [exact source](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/optim/_muon.py)

Installed source SHA-256:

- `torch/optim/_muon.py`: `adb402c6dfd653daf6d9144d57600bfe93469d36cd57ef1bd7eb1f9ecad999f7`
- `torch/optim/swa_utils.py`: `3f05da8e70bc97670f21127124bfa3b16bcb0ab9754a7d261a92b3a779e9aa88`

PyTorch's top-level BSD-style license and the wheel's bundled third-party licenses apply to that dependency. The wheel reports the combined license expression `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib`; it should not be described as having only one dependency license. We call the installed package and do not vendor its source. [Exact upstream license](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/LICENSE)

## Project choices and API

`build_optimizers(model, config)` returns an `OptimizerBundle` with `zero_grad`, `step`, `state_dict`, `load_state_dict`, `param_groups`, `group_report`, and `step_count`. The report retains every parameter name, shape, role, learning rate and weight decay. Each trainable tensor is assigned exactly once, in deterministic named-parameter order. Frozen parameters are ignored; tied aliases shared with a head remain excluded from Muon.

The Muon group contains only hidden two-dimensional `weight` tensors whose dimensions exceed one. Head/output/classifier paths, normalization, feature embeddings, bias, input scaling, TabM member `r`/`s` adapters and three-dimensional member heads stay auxiliary. This rule is tied to the explicitly reviewed model naming contract, not a universal detector for arbitrary architectures. Custom TabM `blocks.i.weight` and RealMLP `layers.i.weight` are eligible. Biases, normalization/embedding parameters and vectors have zero decay in both AdamW and Muon comparisons. Matrix adapters remain AdamW with the explicitly recorded decay.

Starting points proposed before MP2 optimizer results: AdamW `lr=3e-4`, `weight_decay=1e-3`; Muon hidden `muon_lr=0.01` or `0.02`, auxiliary AdamW `lr=3e-4`, momentum `0.95`, five Newton–Schulz steps and `adjust_lr_fn='original'`. EMA decay `0.99`. These are project starting configurations, not paper-optimal settings or verified improvements. Muon and auxiliary learning rates are independent. RealMLP's optional `beta2=.95`, `input_scale_lr_multiplier=6` and `bias_lr_multiplier=.1` apply only when explicitly registered. Default multipliers are one.

`EMA(model, decay)` exposes a detached evaluation model as `.module`. Call `.update(model)` **after** every optimizer step. First update copies the post-step state; later updates average floating state and copy integer buffers. The source object identity is checked, preventing accidental averaging across separately initialized model objects. The trainer must additionally keep preprocessing coordinates and fold provenance fixed. BatchNorm models are rejected until fitting-partition-only statistic recalibration is explicitly implemented. Current models use no BatchNorm.

Optimizer and EMA checkpoint schemas record configuration, named groups and counters. Resume rejects incompatible settings. `EMA.export_state_dict()` returns only model tensors; optimizer moments, EMA history and training checkpoints must never enter the inference bundle. Early stopping, fitting-partition scaler selection and final refitting belong to the trainer and must use the same registered live/EMA evaluation policy.

## Actual verification

`code/tests/test_optimizers_v3.py` initially passed **8/8** synthetic tests in **0.864 seconds** on one CPU compute thread. Evidence: `results/research_v3/optimizer-synthetic-tests.log`. Checks cover disjoint grouping, tied heads, TabM adapters, explicit RealMLP rates, exact direct-AdamW parity, finite native-Muon updates, exact next-step optimizer resume, EMA agreement with `AveragedModel`, EMA resume, and rejection of nonfinite gradients before either optimizer mutates weights. These synthetic checks establish implementation behavior, not improved task accuracy.

No official train experiment, validation bank or test bank was opened for this optimizer subtask. Actual research performance belongs to the separately scheduled common-fold experiments. AI wrote the project wrapper/tests and performed this source review; this is internal AI verification, not teacher or independent human certification.
