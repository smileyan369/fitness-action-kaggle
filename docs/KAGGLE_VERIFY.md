# 在 Kaggle 上验证模型效果

## 结论

**可以上 Kaggle**，而且很适合：

- Kaggle Notebook 有免费 GPU（T4×2 / P100），不用本地显卡；
- 仓库是公开的，Notebook 可以直接 `git clone`；
- 处理后的特征只有几十 MB，上传成私有 Kaggle Dataset 很方便；
- 加载 `best.pt` 跑 `test` split，就能核对报告里的指标是否真实。

但要注意：**在同一个 test split 上复现指标，只能证明“模型和数据处理没有造假/没跑错”，不能证明“换个人、换个摄像头也准”。** 真正的外部验证要用新视频。

## 方案 A：复现 MM-Fit 指标（最快）

1. 本地文件：`D:\fitness-action\kaggle_mmfit_processed.zip`（约 37 MB）。
2. Kaggle → Datasets → New Dataset → 上传这个 zip，命名为 `fitness-mmfit`。
3. Kaggle → Notebook → 新建，Accelerator 选 GPU T4×2，Internet 打开。
4. 上传/打开仓库里的 `notebooks/kaggle_verify.ipynb`，直接 Run All。
5. 预期：`accuracy ≈ 0.9996`、`macro_f1 ≈ 0.9997`。

Notebook 里的核心命令等价于：

```bash
git clone https://github.com/smileyan369/-.git
cd repo/谢金言/fitness-action
python -m src.evaluate \
  --checkpoint ../cloud_results/runs/mmfit_bilstm_s42/best.pt \
  --data /kaggle/input/fitness-mmfit/mmfit \
  --split test --out /kaggle/working/eval_mmfit
```

## 方案 B：复现 Combined / Ensemble 指标

1. 明早云服务器开机后，把
   `/root/autodl-tmp/fitness-action/data/processed/combined`
   打包下载（约 44 MB）。
2. 上传为 Kaggle Dataset，例如 `fitness-combined`，挂载路径
   `/kaggle/input/fitness-combined/combined`。
3. 运行 notebook 的后两节。
4. 预期：
   - Combined TCN 单模型：accuracy ≈ 0.9877，macro-F1 ≈ 0.9893
   - Top-3 ensemble：片段级 accuracy ≈ 0.9922，macro-F1 ≈ 0.9931

## 方案 C：用自己的视频验证（最有说服力）

1. 手机录 3–5 段：深蹲、俯卧撑、仰卧起坐、开合跳、弓步蹲；每段 10–20 秒，全身入镜。
2. 用 `src.pose.extract` 提取 17 点，或用 `src.predict_video.py` 直接跑视频。
3. 看模型输出的类别、置信度、每个窗口的预测。
4. 如果新视频掉点，是正常的 domain gap；这时再决定要不要补数据/微调。

## 判断标准

| 检查 | 通过标准 |
|---|---|
| MM-Fit test | acc 0.995–1.000，macro-F1 0.995–1.000 |
| Combined 单模型 | macro-F1 0.98–0.99 |
| Ensemble 片段级 | macro-F1 ≈ 0.993 |
| 自己的新视频 | 先看是否“人能看出来”的动作；不要求立刻 0.99 |

## 常见坑

- `classes.json` 的类别顺序必须和训练时一致；checkpoint 里已保存，不要手改。
- `manifest.csv` 的 `split` 必须是 `test`；不要拿 train 的指标当泛化。
- `seq_path` 是相对数据目录的路径；上传时保持 `manifest.csv` 和 `seq/` 的相对结构。
- 不要只用同一个 test split 证明“真实效果”；新受试者/新摄像头才是外部验证。
