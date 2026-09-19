# CSNet

MobileNetV2 风格编码器与 CSNet 解码器；含 DenseASPP、低层融合和注意力。

核心结构：[network/modeling.py](network/modeling.py)。基于 [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)；保留 MIT 原版权。

安装与命令见 [统一运行说明](../docs/REPRODUCING.md)，已执行的验证见 [验证记录](../docs/VALIDATION.md)，论文出处见 [参考文献](../docs/REFERENCES.md)。

本目录不附带权重和数据。CPU 随机输入检查不代表训练完成或精度复现。原脚本中部分参数仍沿用历史实验设置，正式运行须自行记录训练协议和检查点。
