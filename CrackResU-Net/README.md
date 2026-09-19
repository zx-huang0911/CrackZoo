# CrackResU-Net

ResNet34 编码器，BAM、浅层 SA、深层 PRAM、辅助监督；保留六种结构变体。

由 Zixin Huang（黄子欣）根据论文自行实现，ResNet34 主干使用 torchvision。

核心结构：[crackresunet/model.py](crackresunet/model.py)。保留主仓库的 warm-start 扩展；新增 evaluate.py 严格加载相容的 state dict。

安装与命令见 [统一运行说明](../docs/REPRODUCING.md)，已执行的验证见 [验证记录](../docs/VALIDATION.md)，论文出处见 [参考文献](../docs/REFERENCES.md)。

本目录不附带权重和数据。CPU 随机输入检查不代表训练完成或精度复现。原脚本中部分参数仍沿用历史实验设置，正式运行须自行记录训练协议和检查点。
