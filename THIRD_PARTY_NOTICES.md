# Code provenance and license review

核对日期：2026-09-19。本目录是整理后的源码快照，来源与许可依据列于下文。上游版权与许可必须随对应文件保留；数据和论文的权限另见 [DATASET.md](docs/DATASET.md)。

## 代码基线

以 [zx-huang0911/CrackZoo](https://github.com/zx-huang0911/CrackZoo) 为主要基线。通过 Git blob SHA-1 核对，从本地归档读取了与远程一致的 329 个源文件/配置文件，避免复制约 30 GB 的完整归档。以下是 Git **tree SHA**，不是 commit SHA：

| 仓库 | 分支 | 树 SHA |
| --- | --- | --- |
| CrackZoo | main | acddbf06f4bdd0ced5b8cbd5832ed29effbf5b1b |
| MCCA | main | 00515b25977570d542dc476c92d124c830fb30b7 |
| CrackResU-Net | main | ab16521f9b23be230ddba2217abe5e3a177c1275 |
| YOLOv4-DAE | master | 046a5fb703b88e2c840b26c6311c47de1d77be85 |

MCCA 与主仓库对应源码一致。CrackResU-Net 主仓库增加了 warm-start 支持，故保留主仓库版本。YOLOv4 独立仓库有训练/验证和损失脚本差异；本次保留主仓库的成套实现，没有混拼不同版本。新增整理和修复见 [CHANGELOG.md](CHANGELOG.md)。

## 许可范围

| 部分 | 来源与现有声明 | 发布处理 |
| --- | --- | --- |
| csnet 基础框架 | [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)，MIT，Gongfan Fang | 保留 [原许可](csnet/LICENSE)；新增结构和工程修改单独标明 |
| yolov7-WMF 基础框架 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)，GPL v3 | 保留 [原许可](yolov7-WMF/LICENSE.md)，不能整体改标为 MIT |
| YOLOv4_DAE 基础框架 | [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) | 项目作者已确认采用此仓库。上游根目录未发现 LICENSE 文件；上游作者在 [issue #2](https://github.com/WongKinYiu/PyTorch_YOLOv4/issues/2#issuecomment-660574873) 指向 ultralytics/yolov3 的许可。按维护者声明及历史 YOLOv3／YOLOv5 GPL v3 文本保留历史条款；41 个文件与固定上游版本完全一致，见 [模块许可](YOLOv4_DAE/LICENSE.md) 与比对记录 |
| MCCA、CrackResU-Net | Zixin Huang（黄子欣）根据论文自行实现；ResNet 主干调用 torchvision | 作者已确认实现来源并授权 GPL-3.0-or-later；网络思想引用原论文，依赖保留自身许可 |
| 数据变换等共享工具 | 部分与 DeepLab 工程接口和实现相近 | 保留 DeepLab 的 MIT 文本，进一步来源说明由作者补充 |
| 新增评估、演示、检查和文档 | 本次工程整理 | 作者原创代码采用 GPL-3.0-or-later，具体范围见 [LICENSE.md](LICENSE.md) |

`LICENSES/` 保留已识别许可的副本，不表示这些文本自动覆盖所有文件。第三方代码、数据、权重和论文不由原创代码授权覆盖。原创修改与继承代码分别遵守对应条款。

## 修改与署名

2026-09-19 的发布整理包括路径配置、数据加载、命令调度、掩膜评估、合成演示与测试。各模型算法思想来自论文与其作者，模型基础实现版权归相应贡献者；CrackZoo 的工程维护署名见 [AUTHORS.md](AUTHORS.md)。

## 当前 YOLOv4 许可参考

按作者“暂以最新版本为准”的要求，已保存 YOLOv4 上游所指向的 ultralytics/yolov3 当前 AGPL-3.0 文本。固定的 commit、原始链接和文件哈希见 [参考记录](docs/yolov4_license_reference.json)，文本见 [AGPL 参考副本](LICENSES/YOLOv3-current-AGPL-3.0-reference.txt)。这份副本记录当前声明，不用于重新授权历史 YOLOv4 代码。历史 GPL 文本与 41 个文件的精确比对已另外记录于同一 JSON，适用说明见 `YOLOv4_DAE/LICENSE.md`。
