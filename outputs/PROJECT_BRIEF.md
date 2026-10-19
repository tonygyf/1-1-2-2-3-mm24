# ExtremeOCR：最小研究项目简报

2026-10-19；目标为 CCF-B；算力为用户提供的 4 × RTX 6000D、约 83GB/卡。

**建议主线：局部强光/高光饱和下的端到端 text spotting。** 从现成 checkpoint 测掉点，再比较普通微调和针对性方法。雨雪、模糊保留为诊断对照；先不同时展开多个课题。

## 已核实的 baseline

| 工作 | 证据 | 本项目用途 |
| --- | --- | --- |
| LRANet++，2025 arXiv；官方仓库现标注 TPAMI | [论文](https://arxiv.org/abs/2511.05818)、[官方源码/权重入口](https://github.com/ychensu/LRANet-PP) | 主 baseline；源码固定快照已下载，尚未运行正式推理 |
| SemiETS，CVPR 2025 | [会议论文](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_SemiETS_Integrating_Spatial_and_Content_Consistencies_for_Semi-Supervised_End-to-end_Text_CVPR_2025_paper.html)、[官方源码](https://github.com/DrLuo/SemiETS) | 后续真实强光无标注图适配入口；目前只核查源码与说明 |
| SwinTextSpotter v2，IJCV 2025 | [论文](https://arxiv.org/abs/2401.07641)、[官方源码](https://github.com/mxin262/SwinTextSpotterv2) | 可作为第二架构候选，仓库已经存在 |
| SVTRv2，ICCV 2025 / OpenOCR | [作者工具库](https://github.com/Topdu/OpenOCR) | 若需单独定位识别失败，可作为 GT crop 识别诊断模型 |
| TextDS，2026 | [论文](https://arxiv.org/abs/2606.28077)、[作者仓库](https://github.com/ZChenDang/TextDS) | 相关工作；仓库仍将训练代码、权重、数据生成列为待发布，暂不列入可运行 baseline |

LRANet++ 有完整实现和权重入口，适合落地；本轮没有足够证据将其称为“行业公认统一基线”。SemiETS 的价值在于提供了近年顶会认可的半监督问题框架，OpenOCR 提供较成熟的 OCR 工程参照。

## 对已有 idea 的关键纠正

1. **雪、模糊、亮度退化并非 OCR 空白。** [On Exploring and Improving Robustness of Scene Text Detection Models](https://arxiv.org/abs/2110.05700)及[代码](https://github.com/wushilian/robust-scene-text-detection-benchmark)已经构造 IC15-C、CTW-C，涵盖多类常见腐蚀。仅换退化类别不足以构成贡献。
2. **真实 glare/reflection 也有前作。** [Robust Scene Text Detection via Learnable Scene Transformations](https://proceedings.mlr.press/v189/cao23a.html)的 Robust-Text 包含真实复杂场景，论文列出雪、雾、眩光、反射等；主要是检测任务。
3. TextDS 覆盖合成雨、雾、欠曝、过曝、低分辨率的检测。它明确提到高光饱和损伤笔画，因此“过曝损坏笔画”本身也不是新发现。其代码可用状态须以仓库内容为准。
4. “检测掉得少、识别掉得多”在本项目中仍是待检验假设。需要 detection/e2e 指标和 GT crop 识别诊断，不能凭两种 Hmean 的数值差直接归因。

## 为什么仍推荐局部高光

这是研究判断：局部高光给出了较具体的失效机制与可控变量，比泛称“恶劣天气鲁棒 OCR”更容易快速形成实验。可研究的问题是：**在仍然可读的文字上，局部饱和如何影响多尺度特征和检测后识别采样，显式饱和条件能否带来超过普通数据增强的收益？**

第一版已实现 68,224 参数的饱和条件 FPN 残差 adapter，任意通道/全通道近饱和统计作为条件，零初始化保持初始模型行为。它是供实验的实现，不是已经成立的论文创新。先证明它优于等预算普通微调和容量匹配残差，再考虑结构一致性或 clean-teacher 监督。

若真实图收益来自识别采样失配，后续集中改对齐/识别端；若主要漏检，集中改检测端。不要一次堆三个模块后才寻找解释。

## 导师工作、基金与学位论文线索

- 许老师的 [Seeing Text in the Dark，MM2024](https://arxiv.org/abs/2404.08965)研究低光文本检测，包含空间约束与特征处理；需要明确本项目研究局部高光与端到端识别的不同问题。
- 作者合作者的[主页](https://wei-min.wang/)还列出 MM2025 的 *Bright to Dark: Stage-wise Bilevel Knowledge Transfer for Seeing Text in the Dark*。这篇应纳入下一轮精读；本轮仅核对到元数据，未读全文。
- [许程沛 UTS 博士论文（2022）](https://opus.lib.uts.edu.au/handle/10453/162796)：*Arbitrary-Shape Scene Text Detection and Its Application in Educational Resource Navigation*。第 6.2 节、印刷页 131 的检索片段指出手工阈值与后处理影响泛化。可从这里继承“稳定定位/泛化”问题意识；论文较早，不代表如今仍未解决。全文下载本轮中断，不能称为已经完整精读。
- 近期官方项目样本：[北京大学多媒体实验室项目列表](https://mipl.pku.edu.cn/project/)列有 2025–2029 年国家自然科学基金重点项目“面向复杂动态场景的多模态对象细粒度持续感知与知识演化理论与方法”；与退化感知的联系属于本项目的选题推断，未见申请书。
- 更垂直的官方样本：[殷绪成项目主页](https://faculty.ustb.edu.cn/XuchengYin/zh_CN/kyxm/81417/content/6155.htm)列有多语言场景文本检测识别项目（2021–2024）及大规模网络图像文本识别项目（2022–2026）。这是已公开项目题目和执行期，不能据此断言 2025/2026 新获批，也不能推演申请书创新点。

本轮是为快速立项进行的定向检索，尚未完成近两年全部相关顶会的系统综述。

## 最小 MVP 已交付

- 清晰目录、Git 忽略规则、属性、数据/权重占位说明和固定上游版本。
- 项目级 grill-me、grilling、autoresearch skills，附来源及许可证。
- clean + 六类退化 × 三档的生成工具；PNG 公平编码、种子、哈希、原始标注继承与 manifest。
- 混合训练集引用合并；LRANet 运行配置生成、固定置信阈值插件与真实日志指标采集。
- adapter 前向、反向、初始恒等及数据协议检查；预览为程序测试图，尚无正式 OCR 精度。

开始运行：`python scripts/smoke.py`；正式资源准备与运行命令见 `docs/EXPERIMENTS.md`。

## 投稿定位

先按 ICME 类 CCF-B 会议的集中贡献组织实验；[CCF 多媒体目录](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/)列有 ICME、ICASSP；[CCF AI 目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)将 Pattern Recognition 列为 B 类。TMM/PR 等期刊需要更完整证据，不能承诺比会议更快。当前不虚构截稿日期或录用周期。

下一步按 [TODOLIST.md](TODOLIST.md) 执行。本轮停在 MVP 与待办，不启动长时间训练或无界 autoresearch。
