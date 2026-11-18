# 状态与下一步

更新：2026-10-22，Asia/Shanghai。用户本轮已确认：完成容器、正式资源、baseline 推理和短程诊断；不进入完整方法训练。原2026-10-19“无服务器结果”状态已被本轮实测替代。

## 已查证事实与实测

- 新建本项目 Docker 环境，未修改其他项目容器，所有安装/下载/编译/实验在容器内。实测4×RTX6000D，sm_120，单卡约83.1GiB；GPU2有其他任务，实验用0/1/3。
- 原 LRANet++ 固定快照560文件字节审计保留；现代移植位于独立目录。Dockerfile.modern 固定 NGC digest、依赖源码 archive、补丁 SHA，Compose 提供资源约束。最终镜像验收详见初始化报告。
- 作者 Total-Text：1255 train / 300 test，ZIP CRC、解码、尺寸与哈希检查通过。seed42、按源图/字节重复分组为 train1067/dev188；近重复/场景人工审阅仍待完成。
- 官方 checkpoint 已下载并固定 SHA。它与发布代码有旧键/冗余层差异；仅对这一份原始 SHA 的明确策略保留全部550匹配 tensor，补8个TPS常量、略去24个不使用的键，模拟上游非严格加载；未知权重仍严格拒绝。原始权重保留。
- CUDA 生命周期通过：FPN恒等最大误差0，DCNv2前后向有限，自定义sampler与原生多batch/越界采样参考的前后向最大误差1.19e-7。
- 300张官方clean：上游原阈值扫描实测 detection Hmean90.1068%、无词典e2e84.8005%。checkpoint无训练元数据，标准/扩展profile未核清，G04不判通过。
- dev188张、固定阈值0.3：clean86.0739/85.1941，glare-s1 85.9343/84.6890，s2 81.8825/78.1925，s3 62.5363/59.3017（前者det，后者e2e，单位%）。包含pre/post recognition confidence候选探针。未构造test腐蚀集，未用test选阈值/模块。
- 真实单卡baseline30步、双卡adapter DDP30步、15→30步optimizer/EMA恢复通过。adapter24个tensor有非零梯度并发生更新；这只是链路健康证据。
- 新增/修复：Docker资源与环境入口、断点续传与校验、严格权重转换、合法三角形忽略GT、现代CUDA stream参数、DDP local-rank传参、传统MMCV非持久buffer恢复。39项软件测试和fixture smoke通过。
- 坏样本错误风暴实际触发watchdog终止；全部故障注入未做完。失败run和复盘保留，不重写为成功。

## 研究假设

局部饱和可能造成检测与识别采样/置信度失配；饱和条件adapter能否超过普通微调、等容量残差和位置对照仍未知。当前退化掉点与候选过滤统计不足以证明机制，也不能用两种care集合不同的Hmean相减归因。

## 未验证项

- G04论文profile/精度对齐；原权重实际训练配置/数据仍待作者证据。
- 四卡同步30步、完整资源故障注入、精确RNG/DataLoader续训轨迹；当前只验证单卡/双卡与状态恢复。
- 训练/开发场景近重复审阅、人工可读性分层、严格无训练来源暴露的pretrain初始化。
- 所有500/10000步有效性实验、三seed、GT-crop独立识别、真实heldout、第二数据集/架构和统计显著性。

## 下一步

先解决profile证据与四卡/故障门槛，保持固定阈值和独立dev。随后才按SOP做普通微调、容量匹配、饱和/全局/翻转等预算pilot；不达预登记目标则记录负结果并停止大训练。最新证据索引、镜像ID和运行路径见 `outputs/INITIALIZATION_REPORT.md`，逐run检查用 `scripts/inspect_run.py`。
