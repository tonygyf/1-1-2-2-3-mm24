# Docker 初始化与诊断报告

2026-10-22，Asia/Shanghai。按用户确认的范围完成初始化与短程诊断；未启动500/10000步方法实验。

## 已查证事实

- 项目容器 `extremeocr-research-1` 由固定配方重建，镜像 **`sha256:0d8a927f4d52978f41572f37a13dd6485a0761edd072978705f08f061ef99cde`**。它独立于临时开发容器；依赖检查、39项单元/协议测试和fixture smoke全部通过。
- Python3.12.3、NVIDIA torch2.10.0a0/25.12、CUDA13.1、MMCV1.7.2、MMDetection2.15.1；4张RTX6000D capability均实测12.0，约83.1GiB/卡。实验使用空闲的0/1/3，未改其他项目容器或占用GPU2。
- 容器硬限制48GiB RAM、无额外swap、12核、8GiB shm、1024 PID；Compose非root、init与本进程组watchdog启用。新容器watchdog阈值40GiB。早期开发容器阈值自动钳到42GiB。
- 官方固定快照560文件字节一致。现代兼容修改在显式清单与补丁中单独记录；DCNv2/custom sampler均从源码针对sm_120编译，没有把版本号放宽当作运行验证。
- 作者Total-Text下载完整：1255训练/300测试，图像解码、尺寸、ZIP CRC、archive及逐图SHA可核验；重新运行下载器会逐文件比对archive并拒绝被修改的提取文件。
- seed42源划分train1067/dev188；clean+三档glare训练4268图/36052实例，dev752图。PNG统一编码，原始标注与split继承，混合比例1:1:1:1。原始图像、权重、外部源码、预测均未纳入Git。
- 作者原始权重SHA：`520c3285d46fc77cf43bfe8f7f2854254e691e5c28265fd911939f101e8d624e`。它与发布代码有旧键名/冗余层差异；只对该SHA的固定策略保留550个匹配tensor，数值不变，补8个按官方代码生成的TPS常量、略去24个当前代码不用的键。这模拟官方非严格加载；原始权重未改。完整转换记录见权重旁`.source.json`。

## 实测结果及边界

300张官方clean采用上游原阈值扫描，检测Hmean **90.1068%**、无词典端到端Hmean **84.8005%**。由新镜像重跑后两项与开发容器完全一致。两项分别取上游扫描结果，不与固定阈值表混用。

**checkpoint缺少训练元数据，标准/扩展训练profile仍未核清，G04不判通过。** 不能用分数相近反推训练数据，也未以官方test调整研究阈值或模块。

独立dev188张，始终固定阈值0.3，来自同一官方初始化的零样本诊断：

| 条件 | 检测Hmean % | 无词典e2e Hmean % |
| --- | ---: | ---: |
| clean | 86.07 | 85.19 |
| glare s1 | 85.93 | 84.69 |
| glare s2 | 81.88 | 78.19 |
| glare s3 | 62.54 | 59.30 |

这些dev零样本run在开发容器中的已审计移植栈完成，run中保留pip freeze与扩展哈希；正式镜像另通过完整官方推理一致性、CUDA和训练门槛。官方微调权重可能已见过dev源图，因此这张表不是从未见过训练来源的泛化结果。合成可读性与近重复/场景分组仍需人工审阅。

CUDA门槛：零初始化FPN最大差 **0.0**；自定义sampler对原生多batch、重复batch索引及越界采样参考，前后向最大差 **1.19e-07**；DCNv2前后向有限。

真实30步诊断：单卡baseline、双卡adapter、15→30步恢复均通过；恢复包含optimizer、iteration和EMA。正式镜像中普通微调/adapter又以同一混合数据、初始化、batch（4/卡×2）、seed42和30步预算分别完成；adapter梯度有限并更新。训练权重在新镜像中严格重新加载并完成188张dev推理。**这些只验链路，不是方法有效性实验。** 跨world-size续训未承诺RNG/DataLoader逐步轨迹一致。

## Skills 与迭代

- ponytail从[官方仓库](https://github.com/DietrichGebert/ponytail)安装到`.agents/skills/ponytail`，固定commit、许可证及来源；未执行第三方hooks。新增技能下一轮可自动发现，本轮已手工读取应用。
- grill-me为grilling别名；读取项目事实、进行一次技能要求的只读子调查，由用户明确选择“先完成初始化和诊断”，既定研究边界沿用AGENTS/SOP。
- autoresearch以classic方式运行：基线0+6个有界阶段，指标为可机械核验的准备门槛，保留失败并修复后验证。日志遵循项目目录约定在`work/runs/initialization-20261022/autoresearch/classic-261022/`；本轮没有无界循环或发布。

## 修复与失败保留

| 实际失败 | 处理与证据 |
| --- | --- |
| NGC匿名UID令torch optimizer缓存查询失败 | 创建对应容器用户；软件测试通过 |
| OneDrive/Drive流中断 | 断点续传、字节数/SHA、ZIP CRC和提取文件逐字节摘要复核 |
| 旧CUDA API/头、CUDA stream整数参数、DDP local-rank写法 | 独立兼容补丁；CUDA、DDP真实训练通过 |
| 作者checkpoint/发布代码键不同 | 固定SHA的明确转换，所有匹配tensor不变；未知权重拒绝 |
| OpenCV缺TPS、数据错误风暴 | 安装contrib；watchdog实际终止错误组，修复后完成训练 |
| 合法三角形忽略GT被错误拒绝 | 最少三顶点校验；真实matcher回归测试通过，未改标签 |
| MMCV保存非持久buffer、EMA恢复顺序 | 只接受与运行时常量逐值相等的额外buffer；原EMA hook注册备份后恢复 |
| 官方archive文档中的悬空symlink使构建失败 | 安全提取并保留symlink；回归测试与从头镜像构建通过 |

失败run和构建日志保留在`work/runs`与`work/setup`。NCCL正常退出仍给出未显式destroy进程组的警告；本组进程由watchdog/init管理，但完整G05故障注入门槛仍未全部验收。

## 证据索引

- [`initialization_status.json`](initialization_status.json)：镜像、配置/数据/初始化权重哈希、硬件、预算、逐run状态与剩余事项。
- [`baseline_diagnostic_scores.csv`](baseline_diagnostic_scores.csv)：真实评测ledger，扫描/固定阈值协议分开标记。
- `work/setup/docker-build-v2.log`：成功的从头构建；`verify-image.log`、`pip-check-image.log`、`preflight-all-gpus.json`：新镜像验收。
- `work/runs/initialization-20261022/lifecycle-image.json`：新镜像CUDA/FPN门槛。
- `work/runs/E000-rebuilt-image`：300图官方推理；`E001-dev-*`：四组零样本dev与score探针。
- `work/runs/G05-mixed-baseline-image`、`G05-mixed-adapter-image`：同预算30步；`G03-reload-adapter-image`：训练后重新加载推理。
- `work/runs/G03-adapter-resume-probe-v2`：15→30步状态恢复。每个run有代码快照、输入哈希、pip freeze、扩展哈希、console和watchdog轨迹。

## 研究假设与未验证项

局部饱和引起检测-识别失配仍是研究假设。候选过滤统计及两种care集合不同的Hmean，不能直接证明失配机制或adapter收益。

仍需checkpoint训练profile证据、四卡同步（GPU2当前被其他项目使用）、完整故障注入、场景/近重复与可读性审核、严格无dev来源暴露的初始化。500/10000步及三seed、GT-crop识别、真实heldout和第二数据集/架构尚未执行。

启动命令见[README](../README.md)；操作SOP见[EXPERIMENTS](../docs/EXPERIMENTS.md)。
