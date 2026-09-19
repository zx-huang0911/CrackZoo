# 参考文献与实现说明

以下五项由毕业论文参考文献整理。完整 BibTeX 见 [references.bib](references.bib)。

| 模型 | 论文 | 本项目实现说明 |
| --- | --- | --- |
| CSNet | [CNN-based network with multi-scale context feature and attention mechanism for automatic pavement crack segmentation](https://doi.org/10.1016/j.autcon.2024.105482) | 基于 DeepLabV3+ 工程框架复现编码器、上下文与解码注意力 |
| MCCA | [Multi-scale context feature and cross-attention network-enabled system and software-based for pavement crack detection](https://doi.org/10.1016/j.engappai.2023.107328) | 包含模块和深监督消融，训练配置以本实现为准 |
| CrackResU-Net | [Encoder–decoder with pyramid region attention for pixel-level pavement crack recognition](https://doi.org/10.1111/mice.13128) | ResNet34 编码器与 SA/PRAM 等模块，支持结构变体 |
| YOLOv7-WMF | [Pavement crack instance segmentation using YOLOv7-WMF with connected feature fusion](https://doi.org/10.1016/j.autcon.2024.105331) | 原论文研究实例分割，本实现使用语义分割输出进行掩膜评估 |
| YOLOv4-DAE | [Modeling automatic pavement crack object detection and pixel-level segmentation](https://doi.org/10.1016/j.autcon.2023.104840) | 冻结基础检测器后训练去噪器，属于两阶段流程 |

引用论文不等于获得代码或数据许可。基础代码来源另见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
