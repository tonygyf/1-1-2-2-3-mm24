# 后续待办

2026-10-22 已完成容器初始化、正式资源和真实GPU诊断，**尚无方法有效性实验**。实际状态见 `docs/STATUS.md`，执行细节见 `docs/EXPERIMENTS.md`。

## 代码阶段已完成

- [x] GitHub固定commit、560文件字节核验、LRA基底锁定。
- [x] 普通/容量匹配/饱和/全局/翻转开关及按层消融。
- [x] 原图分组train/dev、独立GT、推理桥、固定阈值与每run scratch。
- [x] checkpoint结构/shape/EMA备份检查和输入哈希。
- [x] Docker资源配置、仅本组watchdog、运行检查入口。
- [x] 分阶段SOP、合格线、复盘与可否证退出条件。

## 服务器第一阶段（前置门槛）

- [x] 实测4卡GPU capability，单独现代CUDA移植；固定镜像/源码/补丁，记录差异。
- [x] CUDA扩展前后向、FPN恒等、单卡/双卡30步、adapter状态保存恢复。
- [ ] 四卡同步反向和完整checkpoint/RNG/DataLoader轨迹一致性。
- [ ] RAM/超时/SIGTERM/孤儿worker/坏数据风暴故障演练。
- [x] 下载官方图像/标注/checkpoint，登记来源、条款状态和哈希。
- [ ] 核对checkpoint论文训练profile；当前不判G04通过。
- [x] E000官方clean实际推理及扫描：det90.11/e2e84.80，profile未核清。
- [ ] 便携dev与原POSIX序列化同预测对照。

## 研究迭代

- [ ] 记录初始化的训练数据暴露；冻结分组、阈值、退化和预算。
- [x] 零样本dev clean+三档glare、固定阈值0.3、score探针。
- [ ] 饱和率/可读性分层与场景/近重复人工审阅。
- [ ] 五组500步pilot；主要三组10000步×3 seeds；条件/层级消融。
- [ ] clean-only、通用增强、曝光校正、识别oracle/GT crop独立识别器。
- [ ] 真实heldout、第二数据集和第二架构的独立协议适配。
- [ ] 最终test、统计/失败图/负结果、成稿证据和复现包。

## 前轮独立事项

- [ ] Codex连接若仍有问题，先读 `outputs/CODEX_CONNECTION.md`；本轮不重复修改用户配置。
