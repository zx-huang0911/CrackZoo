# YOLOv7-WMF

YOLOv7 特征路径与 WMF 特征融合，最终使用语义分割输出；不是原论文实例分割结果的直接复刻。

核心结构：[models/yolo.py](models/yolo.py)。基于 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)；保留 GPL-3.0 原许可。

安装与命令见 [统一运行说明](../docs/REPRODUCING.md)，已执行的验证见 [验证记录](../docs/VALIDATION.md)，论文出处见 [参考文献](../docs/REFERENCES.md)。

本目录不附带权重和数据。CPU 随机输入检查不代表训练完成或精度复现。原脚本中部分参数仍沿用历史实验设置，正式运行须自行记录训练协议和检查点。
