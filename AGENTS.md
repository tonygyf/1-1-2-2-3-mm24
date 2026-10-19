# Project instructions

- 与用户使用中文交流；代码采用清晰英文命名。
- 当前目标：有前景的极端环境 OCR / text spotting，按 CCF-B 工作量推进。用户已授权先搭建、先跑诊断，再迭代；不要反复要求批准目录或常规实现细节。
- 用户训练资源：4 × RTX 6000D，约 83GB/卡。研究质量优先于节省显存。当前本机 CPU torch 不能代表服务器环境。
- 先读 README.md、outputs/PROJECT_BRIEF.md、docs/STATUS.md；实验操作读 docs/EXPERIMENTS.md。
- 项目级 skills 在 .agents/skills。安装 skill 不等于已授权启动其采访、无界循环或发布流程；按用户当次意图使用。新增常规文件不需要 grill-me 审批。
- 研究源代码在 src，固定协议在 configs，正式结果在 outputs。第三方源码和临时文件放 work，数据放 data，权重放 checkpoints。不要重复建立同义目录。
- 修改实验模块时记录配置、数据哈希、初始化权重、训练预算和 seed。指标为检测 Hmean 和无词典端到端 Hmean；零样本与普通微调是必要对照。
- 将“已查证事实 / 研究假设 / 未验证项”分开。软件 smoke 通过不能称为 baseline 复现、模块有效或论文创新成立。
- 不在测试集上选阈值或优化模块。上游逐阈值最优结果只用于其论文协议复现；鲁棒性试验用固定阈值。
- 别把亮度/雨雪/模糊首次用于 OCR 当作创新；已有相关 benchmark。继续围绕局部饱和与检测-识别失配寻找机制证据。
- Git 不存数据、ckpt、预测 pkl 或外部仓库。不要自动执行第三方 skill 安装后的 shell 脚本。
