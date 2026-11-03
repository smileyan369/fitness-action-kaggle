# 云端训练结果（AutoDL V100S）

## 环境

| 项目 | 配置 |
|---|---|
| GPU | 1× Tesla V100S-PCIE-32GB |
| CPU / 内存 | 80 vCPU / 629 GB |
| PyTorch | 2.5.1+cu121（V100 兼容） |
| Ultralytics | 8.4.198 |
| 训练方式 | 3 个任务共享单卡 + 队列；AMP；每任务 8 个 DataLoader workers |

## 数据

| 数据 | 序列数 | 类别 | 说明 |
|---|---:|---|---|
| MM-Fit 2D pose | 268 | 5 | 官方 session 划分：train 154 / val 42 / test 72 |
| UCF101 健身子集 | 464 | 4 | 按 group 划分；无 sit-up |
| Combined | 732 | 5 | train 483 / val 120 / test 129；UCF 补真实场景，MM-Fit 补 situp |

特征：17 个 COCO 关键点 → 71 维（坐标 + 12 角度 + 8 距离），48 帧窗口。

## 单模型结果（窗口级）

| 模型 | 数据 | test acc | test macro-F1 | 参数量 | 训练耗时 |
|---|---|---:|---:|---:|---:|
| BiLSTM h96 seed42 | MM-Fit | **0.9996** | **0.9997** | 0.37M | 47s |
| BiLSTM h128 seed2024 | MM-Fit | 0.9953 | 0.9962 | 0.49M | 50s |
| TCN seed42 | MM-Fit | 0.9910 | 0.9917 | 0.24M | 33s |
| BiLSTM h96 seed3407 | MM-Fit | 0.9944 | 0.9930 | 0.37M | 34s |
| TCN seed42 | Combined | **0.9877** | **0.9893** | 0.24M | 48s |
| BiLSTM h96 seed42 | Combined | 0.9826 | 0.9863 | 0.37M | 55s |
| BiLSTM h96 seed2024 | Combined | 0.9815 | 0.9844 | 0.37M | 63s |
| BiLSTM h128 seed3407 | Combined | 0.9835 | 0.9839 | 0.49M | 80s |
| BiLSTM h128 seed2024 | Combined | 0.9708 | 0.9732 | 0.49M | 70s |

## 集成结果（Combined，top-3）

| 指标 | 数值 |
|---|---:|
| 窗口级 accuracy | 0.9869 |
| 窗口级 macro-F1 | 0.9890 |
| **片段级 accuracy** | **0.9922** |
| **片段级 macro-F1** | **0.9931** |

集成成员：`combined_tcn_s42`、`combined_bilstm_s42`、`combined_bilstm_s2024`。
片段级：同一视频的所有窗口概率取平均，再 argmax。

## 验收

- 9 个全量任务全部 `exit=0`，`metrics.json` 完整，无 NaN。
- 最佳单模型 Combined TCN：test macro-F1 **0.9893**。
- 最佳集成：片段级 macro-F1 **0.9931**。
- 目标为 test macro-F1 ≥ 0.80，**已通过**。
- 结果与模型已打包、下载并推送到 GitHub（commit `e0f5bd5`），随后于 2026-11-03 08:31 CST 关闭云服务器。

## 重要说明

- 以上是“已切好的动作组/片段”上的窗口分类，不是连续视频里的实时滑窗识别。
- MM-Fit 的动作差异大、姿态干净，所以接近满分；Combined/UCF 的指标更有参考价值。
- 片段级指标比窗口级更严格；两者都保留在 `runs/*/metrics.json` 与 `runs/ensemble_combined/metrics.json`。

## 复现命令

```bash
# 单模型
python -m src.train --data data/processed/combined \
  --config configs/tcn_combined.yaml --seed 42 \
  --device cuda --out runs/combined_tcn_s42

# 集成评测
python scripts/ensemble_eval.py --data data/processed/combined \
  --runs runs --top-k 3 --stride 4 --out runs/ensemble_combined
```

## 文件

- `runs/<model>/best.pt`：checkpoint（含 scaler、classes、config）
- `runs/<model>/metrics.json`：完整指标
- `runs/<model>/confusion_matrix.csv`：混淆矩阵
- `runs/ensemble_combined/metrics.json`：集成指标
- `runs/queue.log`、`runs/queue_status.json`：队列记录
