# 状态与下一步

更新：2026-10-19，Asia/Shanghai。本轮用户明确：不跑服务器训练，先做baseline对齐、FPN adapter对照与完整SOP。

## 已查证与已实现

- GitHub插件核对官方main与锁定commit一致；560个上游文件与官方固定archive逐字节一致，包括LRA基底。`scripts/audit_baseline.py`可重复核验。
- 核实原评测阈值扫描、实际resize、300图GT顺序、word-spotting care差异、识别置信乘入检测score；详情见 `BASELINE_ALIGNMENT.md`。
- 五种module开关、按层消融、等容量/零初始化、梯度与更新健康hook、checkpoint嵌入架构/源配置绑定、key/shape/非有限值/EMA备份审计。
- 原图/场景分组划分train/dev、独立dev GT/matcher、关闭上游默认eval的推理桥、每run独立scratch及数据/权重/配置哈希。
- pre/post recognition score探针、polygon饱和率和GT矩形crop导出、固定识别器输出评分接口。
- Docker参考候选、RAM/CPU/PID/shm上限、无额外swap、watchdog本组清理、超时/无日志/坏样本风暴退出策略。
- G00–G09合格线，30/500/10000步阶段、三seed与最终测试规则、实验复盘模板。
- 本地验收报告：`outputs/local_verification.json`；单一入口 `python scripts/verify_local.py`。最终测试数以报告中的unittest输出为准，不把软件fixture分数放入论文表。

## 未验证或尚待实现

- Docker daemon未运行，参考镜像未构建；实际服务器GPU架构、驱动、CPU RAM、DCNv2/grid sampler kernel、完整模型反向、DDP、训练保存恢复均待服务器门槛。
- 正式图像/官方checkpoint尚未下载，checkpoint对应的论文训练profile待核对。论文目标已摘录，但本地baseline精度未复现，模块有效性未验证。
- 官方Total-Text微调权重可能见过后划分的clean dev源图；如需严格初始训练隔离，所有研究组用同一pretrain初始化并增加train-only clean对照。
- 便携dev序列化已通过官方matcher软件fixture；服务器同预测对照原POSIX序列化的一致性待验。
- GT曲线校正/同模型GT-grid oracle、OpenOCR自动推理、通用增强与曝光校正自动分支、第二数据集/架构、真实heldout适配和配对bootstrap仍待接。SOP已规定这些实验，未冒充已完成代码。

## 下一步

服务器先按 `CONTAINER.md` 实测环境和资源故障演练，再做E000官方复现。之后依次执行独立dev诊断、单卡/四卡30步、五组500步pilot、合格方法10000步三seed，冻结后做最终测试。遇到门槛失败先修该链路；本轮停在本地软件验收。
