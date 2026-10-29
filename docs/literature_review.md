# Literature review and reading record

Checked on 29 October 2026 (Asia/Shanghai). This is an AI-assisted reading record, not a claim of human review. Only primary publications were used. No external dataset, pretrained weight, or answer collection was downloaded. Paper results are not estimates of performance on the course bank.

## Cobbe et al. (2021)

**Source:** [Training Verifiers to Solve Math Word Problems, arXiv v2](https://arxiv.org/html/2110.14168v2). Bibliographic details were checked on the article. **Read:** abstract; sections 4.2 and 4.3 on verification and ablations; section 5.1 on compute. Unrelated appendices and example solutions were not systematically reviewed.

The paper trains a scorer of candidate correctness using final-answer supervision, then selects a highly ranked generated solution. It explicitly distinguishes final-answer success from sound intermediate reasoning. Its ablations discuss solution-level versus token-level prediction and auxiliary objectives.

**Course adaptation:** fit a small shared scorer of the supplied whole-trace embeddings; label every correct candidate positive, using only official training references. Compare this learned scorer with answer voting. **Limits:** no generator training, new samples, large verifier, token-level labels, or imported weights are used here. Whole-trace embeddings do not justify a claim of process verification. **Hypothesis:** a learned scorer can improve selection on disagreement groups; an empirical failure would also be informative.

## Wang et al. (ICLR 2023)

**Source:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models, arXiv v4](https://arxiv.org/html/2203.11171v4). **Read:** abstract, introduction, and section 2, including the unweighted answer-aggregation rule and discussion of weighted aggregation. Full experimental appendices were not reviewed.

The method samples multiple reasoning paths and aggregates their final answers. This motivates evaluating agreement as evidence, without assuming agreement establishes correctness.

**Course adaptation:** parse the four existing answers with the official parser and select from the largest valid answer group. Exclude parsing failures from valid-answer counts; predefine deterministic tie handling. Agreement counts can also become features for a learned verifier. **Limits:** the course method does not perform the paper's sampling procedure and has only four supplied candidates; correlated wrong answers can win. No reported gain from the paper transfers to this bank. **Hypothesis:** agreement helps overall, while disagreement cases leave room for learned comparison. Compare full validation and the predefined hard subset, retaining their denominators.

## Guo et al. (ICML 2017)

**Sources:** [PMLR publication record](https://proceedings.mlr.press/v70/guo17a.html) and [paper PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf). **Read:** section 2 on calibration/reliability diagrams/ECE and section 4 on post-processing, especially temperature scaling; selected discussion of binning limitations. No claim of a full supplementary-material review.

Calibration concerns whether predicted confidence matches empirical correctness. Reliability diagrams compare mean confidence and accuracy in bins; ECE weights their absolute differences by bin frequency. Temperature scaling adjusts confidence without changing the winning class in its multiclass setting.

**Course adaptation:** define the event as correctness of the selected final answer. Fit any confidence mapping on grouped training OOF predictions or a training-only holdout, then evaluate on validation. Report Brier score, a declared ECE bin rule, bin counts, and risk versus coverage. **Limits:** four candidates can have multiple correct answers; the largest four-way softmax value is not automatically the probability of the selected answer being correct. Repeated validation selection and small bins limit interpretation. **Hypothesis:** confidence is useful only if measured discrimination and calibration support it.

## Design consequences

These sources motivate a restricted, testable sequence: deterministic voting; a regularized embedding scorer; candidate-comparison features and an objective retaining all positives; and a separate reliability analysis. The multi-positive listwise objective in the workflow is a project design choice, not claimed as an implementation of any of these papers. All learned preprocessing, model fitting, fusion, and calibration must remain within the official training split. Final test scoring follows the recorded freeze.

Publication metadata is in [references.bib](references.bib). The implementation and experiment log determine which adaptations were actually run; this reading record does not claim unrun experiments.
