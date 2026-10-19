# LRANet++ 对齐审计

2026-10-19；状态：**官方源码已对齐；精度和服务器环境未复现**。

## 来源与事实

- [官方固定版本](https://github.com/ychensu/LRANet-PP/tree/ca1c9e3f60df646b66f399b711ee6f83a3254a14)，commit `ca1c9e3f60df646b66f399b711ee6f83a3254a14`。GitHub插件核验时默认main也是该commit，之后的上游变化不自动合并。
- 固定archive SHA256为 `4ed54c12b8475bf02ed9ee5da31cdc53bc58e96796e40f8e5ce3d02f0c9bfdd0`。独立Git blob核验及archive逐文件SHA256核验均为560个文件、缺失0、变化0。离线契约在 `configs/baseline_source.json`；执行 `python scripts/audit_baseline.py` 重查。
- `orthanchors/syntext_14.npz` 确实包含在官方快照中，不能因Git忽略规则将它误判缺失。各head的U_t是普通CUDA tensor，并不可靠地保存在state_dict里，运行必须使用这个基底。
- [官方环境说明](https://github.com/ychensu/LRANet-PP/blob/ca1c9e3f60df646b66f399b711ee6f83a3254a14/README.md#environment)：Python3.7、PyTorch1.8.1、CUDA11.1、mmcv-full1.3.9、mmdet2.15.1。官方自带Dockerfile是更旧的MMOCR模板，不能代替这份环境。
- [论文](https://arxiv.org/html/2511.05818v1) Table X/XIV 的标准外部数据版 Total-Text det/e2e为88.1%/82.2%；扩展训练版为90.8%/84.6%（Table XII/XIV）。这是文献目标，官方发布checkpoint与profile的对应仍待验证。

## 真实评测协议

原配置的外层MultiScaleFlipAug传入 `(1800,1000)`，覆盖内层Resize的缺省scale。模型训练画布1280×1280。原evaluator对14个置信阈值扫描，并分别取det/e2e最优；两者可能对应不同阈值。研究鲁棒性评测只取冻结阈值，不在各退化测试上扫描。

Total-Text matcher用顺序编号 `i` 对应300个GT文本，不是COCO image id。改变JSON顺序、取子集、训练dev直接套用官方GT都会错配。本项目启动时检查300图有序身份与原标注；官方JSON本身仍需来自可信发布包，不能用任意同数量JSON充当reference。

上游先保留score>0.1，再把score舍入三位，按阈值过滤；IoU严格大于0.5。无词典不意味着没有word-spotting过滤：短词、数字、忽略词等使det/e2e care集不同。识别head还将识别置信度乘到检测分数，因此官方det指标不是纯定位探针。

## 本项目的最小改动面

| 层 | 保留 | 本项目变化 |
| --- | --- | --- |
| 第三方源码 | 全部560文件原字节 | 不编辑；构建产物不属于源码清单 |
| E000模型/匹配 | 官方配置与sweep函数 | 绝对资源路径、独立scratch、严格checkpoint键/形状验证 |
| 研究评测 | 原匹配、词过滤 | test将唯一阈值循环限制为单值；dev使用独立生成GT与便携序列化 |
| 研究训练 | 官方损失/优化器/EMA/数据pipeline | 显式预算、训练路径、禁用test-val、全量checkpoint时省略冗余ImageNet初始化、健康hook |
| 实验模块 | 既有FPN四输出的调用关系 | 可选零初始化残差；none不换模型类型 |

完整checkpoint加载拒绝未解释的missing/unexpected/shape错配。官方初始化可缺新增adapter参数；经过映射与shape确认的EMA备份可忽略，普通参数值保持上游test的加载口径。实验权重没有架构sidecar时不能静默当官方权重加载。

## 未验证项

当前只有CPU torch2.2.2，Docker daemon未运行。参考镜像未构建，GPU架构与驱动未读取，DCNv2/grid sampler/DDP/EMA全生命周期未在服务器运行，官方checkpoint/正式数据未下载。现代GPU如需移植，新增独立镜像、补丁及同图数值对比，不改官方快照后继续声称原环境复现。
