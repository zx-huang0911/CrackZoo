# GPU 测试记录

2026-09-19，在 NVIDIA GeForce RTX 4070 Laptop GPU（8 GB）上完成五模型 CUDA 短程训练检查，以及 MCCA、YOLOv7-WMF 的原生训练入口检查。

## 环境

Python 3.12.12，PyTorch 2.9.1+cu126，torchvision 0.24.1+cu126，CUDA 运行时 12.6，NVIDIA 驱动 580.159.03。使用仓库内独立虚拟环境，不继承系统 site-packages。`pip check` 通过。

完整版本记录：[environment.json](environment.json)；安装版本清单：[requirements-lock.txt](requirements-lock.txt)。独立环境测试发现 YOLOv7 导入依赖 `requests` 缺失，已补入根目录 `requirements-models.txt` 并安装后复测通过。

## 五模型训练检查

统一使用 `scripts/smoke_train.py --device cuda --steps 3 --size 64`，batch size 为 2，FP32，从随机权重开始，不下载真实数据或预训练权重。训练与验证样例分别生成，随机种子为 17 和 29。

| 模型与原始记录 | 结果 | 参数更新步数 | 本次检查峰值显存（MiB） |
| --- | --- | --- | --- |
| [CSNet](csnet.json) | 通过 | 3 | 116.6 |
| [MCCA](mcca.json) | 通过 | 3 | 376.2 |
| [CrackResU-Net](crackresunet.json) | 通过 | 3 | 757.5 |
| [YOLOv7-WMF](yolov7_wmf.json) | 通过 | 3 | 1496.7 |
| [YOLOv4-DAE](yolov4_dae.json) | 通过 | 3 | 213.8 |

通过项包括有限损失与梯度、参数实际更新、独立生成样例上的验证、模型与优化器状态保存及重载、重载前后预测一致。检查点在核验后自动删除，哈希仍保留于 JSON。显存值包含检查点重载等整个检查过程，仅描述本次小尺寸测试，不是正式训练的显存需求或部署基准。

YOLOv4-DAE 的检测器和 DAE 分别计算损失，二者均确认参数更新，并完成组合推理。检测框与阈值标签为人工构造，DAE 输入为人工扰动掩膜；未执行真实数据缓存生成，也没有通过 NMS／栅格化反向传播。

## 原生训练入口

| 入口 | 本次设置 | 结果 |
| --- | --- | --- |
| MCCA `train.py` | CUDA，64×64，batch 2；训练 2 次迭代，验证 4 个样例 | 通过，生成 latest / best 检查点 |
| YOLOv7-WMF `train_seg.py` | CUDA，64×64，batch 2；1 epoch，2 个训练 batch、1 个验证 batch | 通过，生成 last / best 检查点 |

两者读取 `demo.py` 生成的 PNG 文件数据集。训练、验证、保存均执行完成；四个原生检查点通过 ZIP 完整性检查，序列化张量记录包含 CUDA 设备标记。MCCA latest 检查点另确认迭代数为 2。[native_training.json](native_training.json) 保留实际指标、文件大小与 SHA-256；这些合成数据指标不用于比较模型优劣。原生入口没有在此记录中宣称断点续训或完整实验复现。

可复制的命令见 [复现与训练](../../REPRODUCING.md)。完整日志和原生权重保留在 `.local/logs/` 与 `.local/native-train/`，不随仓库发布。五模型测试结果位于 `.local/smoke-train/`。

## 其他检查

- GPU 环境下的 9 项核心回归测试通过：[core_tests.txt](core_tests.txt)。
- 结构与入口语法检查通过：[structure.json](structure.json)。
- 原 `model_smoke.py` 的 CUDA 分支补测通过：[CSNet](structural_csnet.json)、[YOLOv4 完整推理链路](structural_yolov4_dae.json)。
- 历史 CPU 训练结果保持原样，见 [cpu_train/](../cpu_train/)。

本次未重做论文正式训练、私有检查点精度评估、混合精度训练、多 GPU 训练或部署 FPS 测试。论文图表来自原始实验，与本次训练检查分开记录。
