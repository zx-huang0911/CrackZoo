# 发布准备

发布目标：[zx-huang0911/CrackZoo](https://github.com/zx-huang0911/CrackZoo)。2026-09-19 已发布源码、论文图表和测试记录。首次源码提交：`9769f14faff8cfcd3defc39b7622c41e9b0f93e4`。

## 已完成

- 作者确认原创代码采用 GPL-3.0-or-later，论文图 5.1、5.6 可展示。
- 保留 DeepLab、YOLOv7 许可；YOLOv4 依据维护者声明、历史 GPL 文本和 41 个文件的上游比对补齐说明，见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
- 五模型 CPU / CUDA 短程训练、MCCA 和 YOLOv7 原生训练入口及 9 项核心测试，见 [测试记录](VALIDATION.md)。
- README 使用论文原图；私有数据、权重、环境和缓存不进入发布文件。

## 发布方式

新仓库以许可文本初始化，再提交完整源码，不继承旧私有仓库历史。远程 CI 见 [GitHub Actions](https://github.com/zx-huang0911/CrackZoo/actions)；本地短程训练通过不代表论文精度复现。

许可依据和作者确认见 [release_review.json](../configs/release_review.json)。论文全文和演示网页可后续补充。
