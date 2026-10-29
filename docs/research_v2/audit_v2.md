# V2 实现审计与可检验假设

审计日期：2026-10-29（Asia/Shanghai）。研究从 `d9585ce4ea52877a448ff146fac5d942ac71b7a1` 的独立 `research/performance-v2` 分支开始。与已提交版本 `419e1158ba2f83b8c08796429e635f1bf8ab2977` 比较，下列旧模型源码、旧冻结资产、`PRE_TEST_FREEZE.json`、旧开发和可靠性结果没有差异。启动时已存在的未跟踪榜单审查文件属于原工作，予以保留。

本审计完整读取用户的 `DASE7506_MP2_Performance_Upgrade.txt`。用户此次明确允许 train/CV/validation 再研究；它不授权以 test 调参。V2 开发前已知旧正式提交的 **348/467** 总成绩，因此不称本研究为从未接触测试反馈的初始实验。没有读取 test 题目、参考或逐题错误。

## 已核实的 V1 事实

| 项目 | 源码或原始证据 | 核实结果与含义 |
|---|---|---|
| 特征 | `code/mp2/features.py:38`、`:54`、`:65`、`:73` | 1024 维 embedding、1024 维 `abs(e - 同题均值)`、16 个统计，共 2064 维。绝对差和余弦统计已经是非线性特征，不能称旧模型完全没有关系信息。 |
| 学习到的打分函数 | `code/mp2/model.py:27`、`:30` | 折内标准化后单个线性权重向量加 bias，没有多层可学习交互。 |
| 同题公共特征 | `code/mp2/features.py:65` | 有效答案种数、并列最大票数组数对同题候选相同，在线性分数差中抵消。候选相关的绝对差、余弦、票数和文本长度不会因此全部抵消。非线性头可能把公共上下文与候选特征交互，但是否有益待实测。 |
| 多正例目标 | `code/mp2/training.py:61`、`:71`、`:76`、`:77` | `LSE(全部候选) - LSE(所有正确候选)`，没有把其他正确候选作为负例。正例单独平移，避免极端低正例分数的指数下溢。全正例题的数据项梯度为零。 |
| 解码 | `code/solution.py:9` | 直接返回原始候选分数 argmax；没有硬排除不可解析项，也没有按答案组聚合。可解析性仅是软特征。 |
| 标准化与标签 | `code/mp2/training.py:46`、`:85` | 标签由官方 parser 比较候选与参考得到，并检查 ID/题目对齐；均值和标准差只在传入的拟合子集计算。未发现标签或折内标准化公式错误。 |
| 搜索范围 | `code/mp2/training.py:191`、`:203`、`:204`，`results/development/search_summary.json` | 三类线性设定 × 两个 L2，共六配置，train 80/20 初筛；三类胜出配置查看 validation，最终胜者才三折 OOF。实测总耗时 56.0689323 秒。12 次 selector 拟合中 11 次收敛；一次未收敛已在原日志明确保留。 |
| 旧搜索保护 | `code/mp2/training.py:193` | `search_summary.json` 存在则拒绝重跑，V2 不删除结果、不改保护，以新增入口执行。 |
| 置信度事件 | `code/mp2/model.py:36`，旧 `train_oof.npz` | 旧校准器拟合的是原始单候选 argmax 正确事件。V2 改选择规则后不能直接沿用，必须重新构造实际所选答案组事件。 |

冻结资产复核 SHA-256：

- `artifacts/final_bundle/config.json`：`24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703`
- `artifacts/final_bundle/weights.npz`：`a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462`

旧结果（读取原记录，**不是本轮新实验**）：

| 子集 | V1 正确 / N | 说明 |
|---|---:|---|
| 完整 validation | 724 / 798 | 90.7268% |
| hard validation | 282 / 355 | 至少两个不同有效解析答案，79.4366% |
| 单正例 validation | 59 / 117 | 50.4274%，仅开发诊断 |
| 四正例 validation | 379 / 379 | 所有候选正确，无排序难度 |

`results/development/train_statistics.json` 记录 train 共 4755 题，2242 题四正例；2641 题只有一个有效答案组、2114 题 hard；每题均至少一个可解析候选。后两个数字来自输入构造，不需要参考标签才能在推理中计算。单组必正确的结论依赖官方每题至少一个正确候选的课程契约，不能外推到任意真实世界题库。

## 分离改动的可检验假设

1. **Mask**：存在有效候选时排除 `None`，从任务契约可证明不会使原来正确的选择变错；提升多少只能由共同 OOF 和预注册 validation 实测。不能把多个 `None` 视为一个答案组。
2. **答案组解码**：`group_max` 与 masked candidate argmax 完全等价，只是实现一致性检查。`group_lse` 的 `alpha ∈ {0,0.5,1}`、`tau=1` 比较显式组大小奖励，不把相关轨迹的累加说成独立证据。LSE 不保证提高正确率。
3. **目标对齐**：在所有候选有效且同答案标签一致时，group CE 的 `alpha=0,tau=1` 与原多正例 listwise 数学等价。Mask、alpha、tau 或 loss 分母变动才产生新的优化问题。对全题平均和仅 hard 平均要同步审视正则，不把均值缩放误归因为架构收益。
4. **独立非线性**：在相同关系特征、分折和目标下，MLP/ResMLP 与线性比较，检验可学习交互是否改善 hard 和单正例题。
5. **候选上下文**：Deep Sets 与相近规模独立 MLP 比较，检验学习到的题内交互是否有额外作用。不得跨题池化、默认加入位置 embedding 或忽略数组对齐。
6. **融合**：只有实际 train OOF 互补错误支持，才评估预注册的小规模组概率融合；不能在不同尺度 raw logits 上默认相加。

模型选择和 validation 次数由 `SEARCH_PLAN_V2.md` 注册。主表必须使用共同外折；early stopping 来自每个外折训练部分的内部子集。OOF 参与超参数选择后仍有选择偏差，不称为无偏最终估计。

## V2 选择器的明确语义

`code/mp2/selection_v2.py` 只导入 NumPy 和原官方 parser。`answer_groups(bank)` 输出 `[N,4]` 整数组号，按原候选首次出现编号；`None` 为 -1。推理不读取参考、ID、答案缓存或训练数据。

- 有效候选原始索引保留；唯一有效答案组仍返回该组中分数最高的原索引。
- 全不可解析输入确定回退索引 0，并在 `selection_details` 的 `all_unparseable_rows` 记录；它的组概率和 margin 为 0。该回退保证接口有效，不能保证答对课程契约之外的题。
- `candidate_max` / `group_max` 平分选择最小原候选索引。LSE 组平分选择最早出现的组，组内分数平分再选最小原索引。完全平分时这一规则有意优先原顺序，不能声称所有平分也严格置换等变。
- `group_probabilities` 是显式组分布，不自动等于校准后的正确概率。诊断取**实际选择组**的概率；candidate-max 与 group-LSE 不同意时，该概率可能不是最大值，margin 也可能为负。
- NumPy `score_loss_gradient` 提供独立解析梯度，PyTorch `torch_loss` 提供训练 autograd。空组在 logsumexp 前使用安全有限占位，之后屏蔽，不对全 `-inf` 做可微 LSE；全不可解析训练标签明确报错。

## 已执行的行为验证

命令：`.venv\Scripts\python.exe -m unittest discover -s code\tests -p test_selection_v2.py -v`。

初次实际结果：8 个测试，7 个通过，1 个因 PyTorch 尚未安装而跳过，测试耗时 0.471 秒。覆盖官方分数/小数合组、None 和全 None、LSE 合成公式、候选 max/group max（含平分）、全部 24 个候选置换、题序/ID/N、1/2/3/4 正例、全正例零梯度、极端分数、非法输入。54 组 objective/alpha/tau/reduction 设置逐分数做中心有限差分。

安装 PyTorch 2.15.0+cpu 后，实际重新执行结果为 **8/8 通过，0 跳过，1.155 秒**；完整输出保存在 `results/research_v2/selection-unit-tests-torch.log`。PyTorch float64 autograd 与 NumPy 独立解析梯度及损失一致，包含极端分数和没有 hard 题时的连通零梯度。运行限制为一个计算线程。

这些合成正确性检查不构成任何性能提升证据；实际 OOF、官方 validation 和新进程资源结果以 `results/research_v2` 的运行记录为准。
