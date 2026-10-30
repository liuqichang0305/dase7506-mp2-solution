# V2 rules, evaluation boundary, and research attribution

Checked online on 2026-10-29, approximately 16:04 UTC+8, using read-only GitHub connector fetches and primary paper/documentation pages. Machine-readable retrieved evidence is in `results/research_v2/rules_evidence.json`. No test references, test predictions, or other students' MP2 implementations were accessed for this review.

## Official rules and scope of the remaining gate

The current official MP2 `main` is still `bb801f4f2e7629d270df765ab12c333b2dc616bb`; the course website `main` is `692eb414d868720f68b54843c50f0b889541c5b0`.

The [official Guide, section 3](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/guide/GUIDE.md#3-constraints) assigns supplied train data to fitting, validation to development and selection, and test references to final scoring/reproduction. It forbids test-based training, tuning, selection and lookup tables. It permits CPU or one GPU no more powerful than RTX 4090 for training; complete CPU inference plus independent scoring is limited to 30 seconds on Xeon Platinum 8457C with four total threads. All additional inference assets together must be at most 16 MiB uncompressed. A local timing does not certify reference-machine eligibility.

The [official Code README, section 2](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/code/README.md#2-develop-your-method) instructs authors to freeze settings before final test evaluation. The [Guide's submission section](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/guide/GUIDE.md#4-submission-and-peer-review) and the [MP2 website form](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/692eb414d868720f68b54843c50f0b889541c5b0/courses/dase7506/mp2/index.html) allow updates using the same account and student ID. The website [project configuration](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/692eb414d868720f68b54843c50f0b889541c5b0/courses/dase7506/projects.mjs) implements the cutoff as 2026-11-01 00:00 UTC+8, corresponding to the stated deadline of 31 October 23:59.

These sources describe general development and submission updates. They do **not explicitly specify an applicable procedure for redevelopment after seeing an earlier test score, followed by another formal test evaluation**. This is a scoped absence in the sources checked, not a claim that the instructor expressly forbids all later research. In particular, a score-update mechanism is not permission to select models using repeated test results.

Under the user's V2 specification, research on train/CV and the preregistered validation confirmation proceeds now. New formal test evaluation and course score updates remain disabled pending explicit course confirmation of the applicable process. The old score, 348/467 (74.51820128479658%), was already known when V2 began; V2 cannot be described as an experiment wholly unexposed to test feedback. No new test score or rank is claimed. Old Issue #294 and its immutable checkpoint links remain unchanged.

The publication metadata has no MP2 release entry at this check. Peer review therefore remains pending the instructor's actual release, regardless of calendar estimates.

## Architectural sources and implementation scope

| Research family | Primary source | What is used here | Author implementation license checked |
|---|---|---|---|
| Small MLP / residual MLP | Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959), NeurIPS 2021 | Motivation for serious small MLP/residual baselines and shared evaluation protocols; project-specific widths, LayerNorm, objective and four-candidate head | [rtdl-revisiting-models LICENSE](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/LICENSE): Apache-2.0 |
| Deep Sets style comparator | Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114), NeurIPS 2017 | Shared candidate encoder and symmetric within-question pooling, followed by a per-candidate contextual head; equivariant scores rather than one invariant set prediction | [DeepSets LICENSE](https://github.com/manzilzaheer/DeepSets/blob/master/LICENSE): Apache-2.0 |
| Optional candidate attention | Lee et al., [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), ICML 2019 | If executed: attention among four candidates without position embeddings. No inducing points, pooling decoder, or claim of full paper reproduction | [set_transformer LICENSE](https://github.com/juho-lee/set_transformer/blob/master/LICENSE): MIT |

These are architectural inspirations. The V2 modules are written for this project; author repositories, datasets, checkpoints and pretrained parameters are not imported. If code is later copied rather than independently implemented, preserve its applicable license, copyright and modification notices. Optional architecture listings are not evidence that an experiment ran; the experiment registry is authoritative. No TabM implementation or score is claimed unless a corresponding completed experiment is recorded.

Substantive AI assistance includes design interpretation, new implementation, experiment orchestration, evidence analysis and this independent agent review. This is an AI code review, not instructor approval or a human peer review. The student must understand the method and conclusions.

## Dependency recommendation and evidence limits

The [official PyTorch local-installation documentation](https://pytorch.org/get-started/locally/) supports Windows/Pip and Python 3.12 within its documented Windows Python range. For the initial pilot, use a CPU binary wheel and record the installed package version; no torchvision, torchaudio or pretrained weights are needed for these numerical models. The rendered page's static version selector and newer body text are not fully synchronized, so it is not used to assert a latest version. Exact installed versions, hardware, thread settings, installation output and real pilot costs belong in the run records. NumPy-only exported inference must be checked against the trained forward pass and run in a clean environment.

## Final read-only recheck

At **2026-10-29 16:08:19 UTC (2026-10-30 00:08:19 UTC+8)**, the official MP2 repository still points to `bb801f4f2e7629d270df765ab12c333b2dc616bb`. The website repository has advanced to `63ee1cb984b9d3229196a14fc0d2312ba7704dd6` with a leaderboard-snapshot commit. Five relevant files were fetched at this new immutable commit and compared by Git blob SHA: MP2 form HTML, project deadlines, MP2 form logic, submission-workflow README and publication metadata. **All five are byte-identical to the starting versions**. Publication still has only MP1, with no MP2 review release. No new applicable formal-test procedure was identified. New leaderboard values were not used for research or selection. See `results/research_v2/rules_final_online_check.json` and `rules_final_workflow_check.json`.
