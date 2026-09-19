# 首页图表

首页使用毕业论文第五章的三张原始统计图，以及图 5.1 和图 5.6 的并排可视化。图片来自最终电子版归档 `v5_0_elec` 中实际被正文引用的 PDF，以 PNG 格式呈现，未重新绘制或修改数值。它们记录的是毕业设计期间的实验结果。

| 论文图号 | 内容 | 首页文件 |
| --- | --- | --- |
| 图 5.1 | 混合总数据集模型推理可视化 | [pixel-predictions.png](assets/pixel-predictions.png) |
| 图 5.6 | 同一组样例的网格级可视化 | [grid-predictions.png](assets/grid-predictions.png) |
| 图 5.2 | 单数据集同域测试中的像素 IoU 对比 | [same-domain-iou.png](assets/same-domain-iou.png) |
| 图 5.4 | 单数据集同域测试中主模型像素 IoU 与 Grid-F1 的关系 | [pixel-iou-grid-f1.png](assets/pixel-iou-grid-f1.png) |
| 图 5.7 | 模块消融实验的 IoU、F1 差值 | [module-ablation.png](assets/module-ablation.png) |

## 阅读说明

图 5.1 和图 5.6 为同一组四域样例，README 仅通过 HTML 表格并排展示两张原图，未拼接、裁剪或重画。图 5.1 展示像素掩膜，图 5.6 展示局部网格是否存在裂缝。它们来自混合集工程监控设置，不能据四个样例推断独立测试集上的模型排名。

图 5.2 和图 5.4 使用四个来源的数据分别训练和同域测试。`Proprietary` 对应论文中的“自有／通用集”；`CrackRes` 为 CrackResU-Net 的图例缩写。这两张图展示数据域及评价粒度的差异，不能当作跨域测试结果。YOLOv4-DAE 在论文中采用单独的两阶段配对评估，因此没有出现在这两张图中。

图 5.4 的颜色区分模型，形状区分数据来源。Grid-F1 基于图像中的局部网格，未换算为真实道路面积。

图 5.7 沿用论文的分组和颜色。柱长表示对应模型内的配对差值，颜色表示论文对该项证据的归类，不是统计显著性检验结果。主要配对关系如下：

| 项目 | 差值方向 |
| --- | --- |
| C-AM | MCCA-Full − MCCA w/o C-AM |
| VIP residual | YOLOv7 + Mycontact + VIP − YOLOv7 + Mycontact |
| DenseASPP | CSNet-Full − CSNet w/ ASPP |
| PRAM-A | CrackRes-All-PRAM − CrackRes-Full |
| PRAM-B | CrackRes-Full − CrackRes-All-SA |
| MultiCat | YOLOv7 + Mycontact + VIP + MultiCat − YOLOv7 + Mycontact + VIP |
| CBAM | CSNet-Full − CSNet w/o CBAM |
| M-SCFM | MCCA-Full − MCCA w/o M-SCFM |
| DAE 损失 | DAE-BCE+Dice − DAE-Dice |

其中混合数据集上的消融属于论文的工程监控设置，val/test 存在镜像组织；DAE 损失对比属于其两阶段链路的补充实验。不同柱子不共享完全相同的训练和评价设置，不能据此比较不同模型的绝对优劣。协议说明见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 来源与文件记录

图表整理自 Zixin Huang（黄子欣）的本科毕业论文。统计图不包含原始样本。图 5.1、5.6 包含真实裂缝照片和预测，作者已确认可在本项目中公开展示，确认记录见 [发布审查配置](../configs/release_review.json)。该记录依据作者说明，不是对各数据来源的独立权属鉴定。图表的展示不授予底层数据的再分发权限。

[figure_sources.json](assets/figure_sources.json) 记录归档内路径、原始 PDF 的 SHA-256 和 PNG 的 SHA-256。转换命令为：

```bash
pdftoppm -singlefile -scale-to 1800 -png original-figure.pdf output-name
```
