# 复现与训练

本页给出从环境安装、短程训练检查到自有数据训练的步骤。所有命令默认在仓库根目录的 Bash 终端执行。短程检查使用生成的线状掩膜，不需要下载数据集或预训练权重。

> 已在 RTX 4070 Laptop GPU（8 GB）上通过五类模型的 CUDA 短程训练检查，以及 MCCA、YOLOv7-WMF 的原生训练入口检查。具体环境和结果见 [GPU 测试记录](validation/gpu/README.md)。

## 配置环境

先设置本项目的临时目录，再创建独立环境。`local_env.sh` 将 pip、PyTorch、Matplotlib、CUDA、Python 字节码等缓存定向到 `.local/`，不修改已有 Conda 环境。

```bash
source scripts/local_env.sh
python -m venv .local/venv-gpu
source .local/venv-gpu/bin/activate
python -m pip install --no-cache-dir torch==2.9.1+cu126 torchvision==0.24.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple
python -m pip install --no-cache-dir -r requirements-models.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple
python -m pip check
```

这里使用 CUDA 12.6 版 PyTorch；`nvidia-smi` 中的 CUDA 数字表示驱动支持范围，不需要与 wheel 的运行时版本完全相同。版本配对见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v291)。镜像不可用时，可将普通 Python 包的镜像替换为 `https://pypi.org/simple`；PyTorch 的 CUDA wheel 保留官方源。

大体积 CUDA 依赖也可用 `uv` 并行下载。缓存位于 `.local/cache/uv`：

```bash
python -m pip install --no-cache-dir uv --index-url https://mirrors.aliyun.com/pypi/simple
uv pip install --python .local/venv-gpu/bin/python \
  --index-strategy unsafe-best-match \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.9.1+cu126 torchvision==0.24.1+cu126 -r requirements-models.txt
```

没有 NVIDIA GPU 时，另建 `.local/venv-cpu`，使用同版本 CPU wheel，并在后文命令中指定 `--device cpu`：

```bash
python -m venv .local/venv-cpu
source .local/venv-cpu/bin/activate
python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements-models.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple
```

模型子目录中的 requirements 保留了原工程的历史依赖，不要与根目录这组依赖混装。仅运行掩膜指标与合成演示时，安装根目录 `requirements.txt` 即可，无需 PyTorch。

## GPU 训练检查

先确认当前环境能识别 GPU；指定 CUDA 后，检查脚本不会自动回退到 CPU。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

依次检查五类模型，每个模型在独立进程中运行，避免上游同名 `models`、`utils` 包之间的导入冲突：

```bash
(
for model in csnet mcca crackresunet yolov7_wmf yolov4_dae; do
  python scripts/smoke_train.py --model "$model" --device cuda \
    --steps 3 --size 64 --output ".local/smoke-train/$model" || exit 1
done
)
```

检查包含三次参数更新、合成验证集上的前向计算，以及模型与优化器状态的保存和重载；同时检查损失、梯度和重载前后的预测。结果写入各目录的 `result.json`。输出目录须不存在，重复运行请换目录名；检查点完成核验后自动删除，加入 `--keep-checkpoint` 可保留。

| 模型 | 损失与训练路径 |
| --- | --- |
| CSNet | 全模型，Generalized Dice |
| MCCA | 全模型，加权 BCE、Dice 和侧分支监督 |
| CrackResU-Net | 全模型，交叉熵、Dice 和辅助监督 |
| YOLOv7-WMF | 语义分割配置，BCE + Dice |
| YOLOv4-DAE | 检测器定位／置信度／阈值损失与 DAE 的 BCE + Dice 分别计算，同时检查组合推理 |

YOLOv4-DAE 检查使用人工构造的框与阈值标签、人工扰动的掩膜，不经过真实检测缓存生成；NMS 和掩膜栅格化之间没有反向传播。本入口是小规模训练工具，正式实验仍使用下面的原生训练脚本。

## 原生训练入口检查

下面通过文件数据集运行 MCCA 和 YOLOv7-WMF 的实际训练入口，覆盖数据读取、训练循环、验证与检查点输出。两个入口已在 CPU 和 CUDA 上实测通过。首次运行时生成数据集，三个划分各有四张独立生成的图像。已有本页生成的数据和 YAML 时可直接复用，重复生成需另选输出目录：

```bash
python scripts/demo.py --output .local/native-train/demo
```

MCCA 运行两个训练迭代并验证：

```bash
python scripts/run_experiment.py --model mcca --mode train \
  --dataset-root .local/native-train/demo/dataset \
  --output .local/native-train/mcca --total-itrs 2 \
  --batch-size 2 --val-batch-size 2 --execute -- \
  --input_size 64 --num_workers 0 --val_interval 2 --print_interval 1 \
  --no-pretrained_backbone --no-enable_paper_metrics \
  --no-save_vis --no-save_train_vis --progress_mode plain
```

YOLOv7-WMF 运行一个 epoch，限制为两个训练 batch 和一个验证 batch：

```bash
python scripts/prepare_yolo_config.py --model yolov7_wmf \
  --dataset-root .local/native-train/demo/dataset \
  --output .local/native-train/yolov7.yaml
python scripts/run_experiment.py --model yolov7_wmf --mode train \
  --config .local/native-train/yolov7.yaml --output .local/native-train/yolov7 \
  --epochs 1 --batch-size 2 --execute -- --img 64 --workers 0 \
  --device 0 --max-train-steps 2 --max-val-batches 1 --no-augment
```

这些命令产生的 loss 和指标只用于检查训练流程。三步更新或一个短 epoch 不代表收敛，也不能与论文的精度及 FPS 比较。原生训练输出保留在 `.local/native-train/`，检查点可能占用数百 MB。

## 本地测试记录

实际环境版本、训练结果和检查点哈希见 [GPU 测试记录](validation/gpu/README.md)。本次 CPU 短程训练记录位于 [cpu_train/](validation/cpu_train/)，早期 CPU 前向／反向记录仍保留在 `docs/validation/`。论文中的图表来自原始实验，不是本次合成样例测试的结果。

指标工具与核心回归检查可单独运行：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --compile-third-party --output .local/reports/structure.json
```

## 数据与训练

先按 [数据说明](DATASET.md) 提供经过授权、独立划分的数据。执行：

```bash
python scripts/audit_dataset.py --root .local/data/private --output .local/reports/data_audit.json
python scripts/run_experiment.py --model mcca --mode train --dataset-root .local/data/private
```

默认打印将要执行的命令，加入 `--execute` 开始运行。三个分割模型支持 train/val/test 目录，统一入口执行前会检查跨划分相同图像。原生脚本仍保留旧平铺目录和 tiny-overfit 行为；它们可做调试，不应用于独立测试结论。正式对照必须记录各模型实际采用的划分、种子和超参数，见 [评估协议](EXPERIMENT_PROTOCOL.md)。

CSNet 默认不下载 ImageNet 权重，需要时显式传 `-- --pretrained_backbone`。CrackResU-Net 原训练入口默认使用 torchvision 的 ImageNet 编码器权重；第一次训练可能下载。其 `--use_tiny_random` 会改用随机数据，不能用它替代真实数据训练。

`--output` 对 MCCA 和 YOLOv7 生效；CSNet、CrackResU-Net 原生训练脚本仍在各自工作目录输出。YOLOv4 的轮数、数据、输出和缓存由 YAML 控制，不能假设统一命令的所有选项都会覆盖它。命令清单只是调度记录，不是公平比较的完整实验协议。

## YOLOv7-WMF

```bash
python scripts/prepare_yolo_config.py --model yolov7_wmf --dataset-root .local/data/private --output .local/reports/yolov7_train.yaml
python scripts/run_experiment.py --model yolov7_wmf --mode train --config .local/reports/yolov7_train.yaml --epochs 50
```

配置文件中写入绝对路径，避免切换工作目录后找不到数据。加入 `--execute` 开始训练。`val_seg.py` 评估配置中的 val 路径；最终测试需另外创建指向 test 的评估配置，不要拿该配置训练或选择检查点。

## YOLOv4-DAE

YOLOv4-DAE 分两阶段训练：先训练基础检测器，再冻结检测器、生成粗掩膜缓存并训练 DAE。准备好阈值通道检测标签与检测器检查点后，生成配置：

```bash
python scripts/prepare_yolo_config.py --model yolov4_dae --dataset-root .local/data/private --frozen-detector .local/checkpoints/YOLOv4_DAE/detector.pt --output .local/reports/dae/config.yaml
```

生成的配置中需要分别保存训练、验证和测试缓存。在 `YOLOv4_DAE/` 工作目录运行：

```bash
python scripts/build_dae_cache_v050.py --config /absolute/path/to/CrackZoo/.local/reports/dae/config.yaml
python train_crack_dae_v050.py --config /absolute/path/to/CrackZoo/.local/reports/dae/config.yaml
python scripts/run_v050_dae_integration_eval.py --config /absolute/path/to/CrackZoo/.local/reports/dae/config.yaml
```

这些是使用自有数据和检查点的运行入口；本次没有实测完整缓存、训练、真实数据评估流程。该模块的上游许可版本仍待确认，见第三方说明。

## 检查点评估

CSNet/MCCA 的 `--mode eval` 采用显式 test 目录，可通过 `--checkpoint` 指定匹配的权重。CrackResU-Net 的新入口：

```bash
python scripts/run_experiment.py --model crackresunet --mode eval --dataset-root .local/data/private --checkpoint .local/checkpoints/crackresunet.pth --output .local/reports
```

加 `--execute` 后在 CPU 上生成原始标签尺寸的 PNG，使用严格 state dict 加载，不进行部分参数匹配。其他变体可通过 `-- --variant no_aux` 指定。再用根目录 `scripts/evaluate_masks.py` 计算统一指标。

遗留训练脚本可能使用 `torch.load(..., weights_only=False)`，只加载自己训练或可信来源的权重。公开包不附带这些文件。

## 存储与清理

本页的安装、GPU 检查与原生短程训练均将新增环境、缓存、样例、日志和检查点保存在仓库内。`.local/` 被 Git 忽略；公开记录仅保留 `docs/validation/` 中经过整理的版本信息与测试结果。

结束实验后先退出环境，再删除 `.local/` 即可释放本页流程产生的临时存储。删除前保留需要的训练权重或日志；也可以只删除 `.local/venv-gpu`、`.local/cache` 和 `.local/native-train`。下一次打开终端时重新 `source scripts/local_env.sh`。

真实数据的正式训练请检查各模型原生输出设置。CSNet 和 CrackResU-Net 的历史入口可能在各自源码目录下生成输出，这些仍位于 CrackZoo 内，但不包含在 `.local/` 清理范围中。
