# YOLOv4-DAE

基础检测器输出粗掩膜，随后以去噪自编码器修正。训练 DAE 前必须先生成冻结检测器的离线缓存。

核心结构：[models/crack_pipeline_v010.py](models/crack_pipeline_v010.py)。基于 [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)；具体上游许可版本尚需完成核对。

安装与命令见 [统一运行说明](../docs/REPRODUCING.md)，已执行的验证见 [验证记录](../docs/VALIDATION.md)，论文出处见 [参考文献](../docs/REFERENCES.md)。

本目录不附带权重和数据。CPU 随机输入检查不代表训练完成或精度复现。原脚本中部分参数仍沿用历史实验设置，正式运行须自行记录训练协议和检查点。
