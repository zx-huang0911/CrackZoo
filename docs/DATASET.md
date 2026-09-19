# 数据说明

## 发布内容

仓库不提供完整真实数据集、独立 mask、检测标注、数据缓存、私有下载链接或训练权重。首页新增的论文图 5.1、5.6 含少量真实样例照片与预测，已获作者确认可在本项目中公开展示；其余三张图仅含汇总统计。来源与发布限制见 [FIGURES.md](FIGURES.md)。`scripts/demo.py` 从固定随机种子生成新的折线和背景，用于验证配对、指标与导出，不来自任何真实裂缝数据集。

代码 LICENSE 不赋予第三方图像、标注、论文插图或数据集的使用和再分发权限。权重是否能公开，也需结合来源数据条款和课题组约定判断；本版本不提供权重下载。

## 已发现的来源线索

下表根据归档 README、论文和配置整理，不是逐文件权属核验结论。

| 来源 | 归档中的线索 | 获取与许可处理 |
| --- | --- | --- |
| Crack500 | 论文同域与跨域实验 | 查阅 [原作者项目](https://github.com/fyangneil/pavement-crack-detection)，按原始条款获取；不从本仓库转发 |
| DeepCrack | 论文同域与跨域实验 | 查阅 [原作者项目](https://github.com/yhlleo/DeepCrack)；代码许可与图像许可分别核对 |
| GAPs / GAPs384 | 论文与混合数据 README | [官方申请条款](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Datasets/Request-Gaps2.pdf)限制为学术用途，并要求接收方同意条款；不能视为无条件可再分发 |
| CFD、AEL、CrackTree200 等 | 聚合来源 README 或早期配置中出现 | 是否进入最终混合集及逐文件来源仍需核对，暂不再分发 |
| 自有／通用样本 | 论文单独列出的一组数据 | 采集主体、内部授权、与公开来源的边界待补充；保持私有 |
| 聚合项目 | 归档 README 对应 [khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation) | 聚合或可下载不等于获得所有组成数据的再分发权；不沿用其网盘镜像 |

检查日期：2026-09-19。公开真实示例图前，应逐图记录原始来源、许可版本、署名及裁剪/标注修改情况；重新截图不改变来源权限。

## 本地组织

使用 `train/images`、`train/masks`、`val/images`、`val/masks`、`test/images`、`test/masks`。推荐 PNG 二值标签，前景 255，背景 0。通用评估也支持 0/1；模型内部编码以对应加载器为准。

运行 `python scripts/audit_dataset.py --root data/private` 检查配对、二值标签、尺寸、跨划分同图和同名。该命令只读数据，将结果存入被 Git 忽略的 `reports/`，不判断许可证，也不能替代按原图或场景分组的数据划分。

建议本地维护不入库的来源表：`sample_id, source_dataset, original_id, source_url, license_version, acquisition_date, modifications, split, source_scene_id`。来源不明的样本先排除出准备公开的材料；不能给整个混合集统一套用 MIT 或 CC 协议。
