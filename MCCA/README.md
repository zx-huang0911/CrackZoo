# MCCA

ResNet18，M-SCFM、C-AM、深监督；支持 full_mcca、mcca_no_mscfm、mcca_no_cam、baseline_deepsup、baseline_plain。

核心结构：[models/mcca/mcca.py](models/mcca/mcca.py)。由 Zixin Huang（黄子欣）根据论文自行实现，ResNet18 主干使用 torchvision。

安装与命令见 [统一运行说明](../docs/REPRODUCING.md)，已执行的验证见 [验证记录](../docs/VALIDATION.md)，论文出处见 [参考文献](../docs/REFERENCES.md)。

本目录不附带权重和数据。CPU 随机输入检查不代表训练完成或精度复现。原脚本中部分参数仍沿用历史实验设置，正式运行须自行记录训练协议和检查点。
