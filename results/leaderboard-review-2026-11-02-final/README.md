# 冻结后一次性排名核查

核查时间见 `ranking_result.json`；官方最新提交仍为 `65c16a3eef424e30b254ff58804dffe0fbda8793`，榜单生成时间为 2026-11-01 04:44:26 UTC。本次核查发生在 V3 研究归档冻结之后，不再用于研究、选模或调参。

自己的原提交 Issue #294 为 **74.51820128479658%**，状态 `self-reported`。官网按“原样学号＋账号”取最高有效分，得到 **113 条、第 112 名**；同分仅自己，没有其他并列者。

排除教师 `test` 示例后，是 **112 条学生提交记录、第 112 名**。其中同一个账号分别使用带 `u` 和不带 `u` 的相同十位学号，仍被官网保留为两条；合并这一个明确格式重复后，是 **111 人、第 111 名**（110 人严格高于自己）。另一条短学号缺少可公开核实的名册匹配；如不计该条，是 **110 人、第 110 名**。教师账号代录、且教师明确接受的一名正式学生必须保留，不能按账号一概排除。

因此学生去重口径的范围为 **第 110–111 名／110–111 人**；无论采用哪个口径都未进入前 35%。111 人和110人的 `floor(0.35*N)` 均为38。这是用户的目标比例计算，不是课程成绩承诺。正式排名仍须以教师复现、同行评审和裁定为准。

只保存了排名所需的去标识化字段、原始公开文件的 SHA-256/Git blob SHA、来源提交与规则说明，没有保存其他学生姓名、完整学号、代码链接或逐题数据。`minimal_snapshot.json` 可供离线重算，`recompute.py` 使用独立 Python 实现验证与官方 JS 函数结果一致。

来源：

- [官方榜单](https://xudongwu-0.github.io/courses/dase7506/mp2/)
- [固定提交的榜单快照](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/65c16a3eef424e30b254ff58804dffe0fbda8793/courses/dase7506/data/leaderboard.json)
- [固定提交的官方发布记录](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/65c16a3eef424e30b254ff58804dffe0fbda8793/courses/dase7506/data/publication.json)
- [自己的原提交 Issue #294](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294)
