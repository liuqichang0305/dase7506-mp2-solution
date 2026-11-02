# V3 截止后研究归档说明

**已错过正式截止时间。本轮 V3 没有进行正式 test，也没有创建新成绩 Issue 或更新课程成绩。**

保留的原 V1 提交是 https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294，历史成绩 348/467。它不是 V3 新模型的测试成绩。

研究保留模型：deepsets_regularized_s0。实际离线验证为 740/798，难题 297/355。新环境复现验证预测及评分一致，耗时 2.280391 秒；推理资产 963670 字节。

完整训练内三折实验：V3 35 项，历史 V2 19 项；固定融合 4 项均拒绝。未完成的 V3 稳定性实验：v3_attention_ema_s1, v3_attention_ema_s2, v3_tabm_embedding_stats_ema_s1, v3_tabm_embedding_stats_ema_s2。不能用部分折推断完整成绩或宣称完成多种子验证。

Four added-seed jobs were pruned without complete CV. Preserved attempt records include an elapsed wall duration of 186401.74966692924 seconds across 2026-10-30 to 2026-11-01, with long gaps in monitoring; this is not evidence of continuous CPU work or uninterrupted supervision. The underlying cause of the gaps is unestablished. Original nominal attempt and absolute campaign deadlines were not reset; resumption stopped overdue work. The formal closure occurred on 2026-11-02 after both the research budget and course score deadline. The reserved delivery buffer was missed, and no timely V3 submission is claimed.

报告区分优化耗时、进程经过时间和监控间断，不声称连续运行24小时。训练 OOF 和已使用的 validation 只能支持有限研究结论，不能推断榜单名次。缩减输入同时改变容量与初始化；配对区间保留选模偏差；历史测试汇总已知。

归档提交：eee478b4959e555b6ec8c902b6db8b69f31df508。按 FREEZE 与复现记录校验源码和资产，复现已归档验证预测；不得因此重新打开正式测试或成绩更新。英文报告逐页证据和完整表见 closeout_inputs.json 及其 evidence_sha256。正式报告旧流程仍要求 TEST_COMPLETE，本流程没有伪造该记录。

AI 实质参与了代码、实验、检查和写作；内部 AI 审查不等于教师认可。同行评审按教师实际开放流程单独处理，本说明不宣称已完成。
