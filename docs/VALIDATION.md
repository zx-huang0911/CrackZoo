# 验证记录

日期：2026-09-19。已完成 CPU 与 CUDA 检查。GPU 环境为 RTX 4070 Laptop GPU、Python 3.12.12、PyTorch 2.9.1+cu126；完整版本和原始结果见 [GPU 测试记录](validation/gpu/README.md)。早期 CPU 检查使用单独的环境，版本见 [environment.json](validation/environment.json)。

## 已执行

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 五模型 CPU / CUDA 短程训练 | 每模型 3 次参数更新，损失、梯度、验证与检查点重载通过 | [CPU](validation/cpu_train/)、[GPU](validation/gpu/README.md) |
| MCCA / YOLOv7 原生 CUDA 训练 | 数据读取、短程训练、验证与检查点保存通过 | [JSON](validation/gpu/native_training.json) |
| 核心回归测试 | 手算指标、阈值、编码、尺寸、边缘网格、容差、缺失预测、划分重复和命令路径检查通过 | [测试日志](validation/core_tests.txt) |
| 合成演示 | 生成独立的 train/val/test，共 12 张人工图像；逐图评估和数据审计通过 | [CSV](validation/synthetic_metrics.csv)、[参数与哈希](validation/synthetic_metrics.json) |
| CSNet | 2×3×64×64 随机输入，前向与反向通过 | [JSON](validation/csnet.json) |
| MCCA | 2×3×64×64 随机输入，前向与反向通过 | [JSON](validation/mcca.json) |
| CrackResU-Net | 2×3×64×64 随机输入，前向与反向通过 | [JSON](validation/crackresunet.json) |
| YOLOv7-WMF | 2×3×64×64 随机输入，语义输出与反向通过 | [JSON](validation/yolov7_wmf.json) |
| YOLOv4-DAE | 完整链路随机输入推理、检测头反向及 DAE 独立反向通过 | [JSON](validation/yolov4_full.json) |
| CrackResU-Net 新评估入口 | 随机 state dict 保存、严格加载、4 张合成图推理通过；临时权重已删除 | [JSON](validation/crackresunet_checkpoint.json) |
| CSNet / MCCA 显式数据目录 | 各加载 4 张 train、4 张 val；数据变换输出 3×64×64 | 本地加载检查 |
| 源码语法 | 早期源码 AST 解析与新增入口编译通过 | [早期 AST](validation/syntax.json)、[当前入口检查](validation/gpu/structure.json) |

YOLOv4 记录中的参数数目只统计 DAE；完整链路输入尺寸另列在 JSON 中。时间仅为这次 CPU 检查耗时，不作吞吐率或模型效率比较。合成演示指标来自人为扰动的标签，不是网络预测。

## 尚未验证

尚未重做真实数据上的完整训练、论文精度、全部结构消融、跨域实验、混合精度、多 GPU 或部署 FPS 测试。发布包不附带私有检查点。

根目录的核心 CI 已在 GitHub Actions 通过 Python 3.10、3.11、3.12 检查。历史原生脚本的所有训练、导出与部署路径未作全面兼容性验证。短程训练用于检查代码运行，论文图表仍对应原始实验。

## 已修复的问题

缺失预测原先会被跳过，现改为失败；阈值现在统一传给各指标；前景容差使用固定方形邻域，避免因 OpenCV/scipy 的可用性改变定义。CrackResU-Net 原来以随机训练测试充当 eval 命令，现改为实际 checkpoint 推理。结构检查不再依赖私有权重。

许可状态和旧仓库历史风险不属于计算验证结论，详见 [发布状态](RELEASE_STATUS.md)。
