# 代码许可与适用范围

Copyright (C) 2026 Zixin Huang（黄子欣）

作者自行编写的 MCCA、CrackResU-Net 实现，以及 CrackZoo 的原创命令调度、评估、演示和检查工具，采用 **GPL-3.0-or-later**：可按 GNU General Public License 第 3 版或任何后续版本的条款使用、修改和再分发。完整条款见 [LICENSE](LICENSE)。软件不提供任何明示或默示担保。

此授权仅覆盖作者有权授权的原创代码。文件内已有的第三方版权和许可声明继续有效，具体范围如下：

| 内容 | 许可或处理方式 |
| --- | --- |
| MCCA、CrackResU-Net 的作者原创实现 | GPL-3.0-or-later；调用的 torchvision 等依赖保留其自身许可 |
| `crackzoo/`、`scripts/`、`tests/` 的原创工具 | GPL-3.0-or-later；保留可能适用的第三方声明 |
| `csnet/` 的 DeepLabV3+ 基础代码及继承的共享工具 | 原 MIT 许可，见 [csnet/LICENSE](csnet/LICENSE) |
| `yolov7-WMF/` 的基础代码 | 原 GPL v3 许可，见 [yolov7-WMF/LICENSE.md](yolov7-WMF/LICENSE.md) |
| `YOLOv4_DAE/` 的继承代码 | 保留历史 GPL v3 条款，见 [模块许可](YOLOv4_DAE/LICENSE.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md) |
| 数据、权重、论文全文与插图 | 不包含在上述代码授权中；分别遵守来源条款 |

作者已同意在项目中公开展示论文图 5.1、5.6。此展示确认不授予底层数据集、标注或第三方素材的批量再分发权，也不自动将图片按 GPL 授权。来源见 [图表说明](docs/FIGURES.md)。

YOLOv4 当前上游所引用的 YOLOv3 仓库使用 AGPL-3.0。其文本仅作为当前条款参考保存在 `LICENSES/`；本项目继承版本的历史 GPL 依据另见模块许可与固定版本比对记录。整个仓库不能仅凭根目录 LICENSE 概括为所有文件统一采用 GPL-3.0-or-later。
