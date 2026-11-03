# 云端 GPU 运行手册

目标：在云服务器上复现/继续训练，不重新下载 39 GB RGB。

## 0. 最短路径

如果只想训练已经处理好的数据：

```bash
# 上传 data/processed/ 和代码
cd fitness-action
pip install -r requirements.txt

python -m src.train --data data/processed/kaggle448 \
  --config configs/bilstm_kaggle448.yaml --out runs/kaggle448_bilstm

python -m src.train --data data/processed/mmfit \
  --config configs/bilstm_mmfit.yaml --out runs/mmfit_bilstm
```

`data/processed/*/seq/*.npz` 是 71 维特征，体积远小于原始视频；训练只需要这些文件和 `manifest.csv`。

## 1. 实例建议

| 项目 | 建议 |
|---|---|
| GPU | 1× RTX 4090 / A10 / L4（24 GB）足够；T4 16 GB 也可以 |
| CPU | 8 vCPU 以上，视频解码用 |
| 内存 | 32–64 GB |
| 磁盘 | 100 GB（UCF101 + 中间文件 + 特征） |
| 系统 | Ubuntu 22.04 + CUDA 12.x |

不需要 A100/H100；模型只有约 0.37M 参数，GPU 主要加速 YOLO-pose 提点和多 seed 训练。

## 2. 数据上传

优先上传：

```text
data/processed/kaggle448/
data/processed/mmfit/
data/raw/ucf101/clips/          # 如果还要提 UCF101
data/raw/ucf101/manifest.csv
```

不要上传 `data/raw/mmfit/*.mp4`；MM-Fit 我们用的是 `pose_2d.npy`，不是 RGB。

如果要从头下 UCF101：

```bash
mkdir -p data/raw/ucf101
curl -L --fail --retry 6 -C - \
  -o data/raw/ucf101/ucf101_kaggle.zip \
  https://www.kaggle.com/api/v1/datasets/download/pevogam/ucf101
```

Kaggle 数据集下载可能需要账号/cookie；在云上更稳的方式是用 `kaggle datasets download -d pevogam/ucf101`。

## 3. 提取 UCF101 的 17 点

```bash
python scripts/prepare_ucf101.py \
  --zip data/raw/ucf101/ucf101_kaggle.zip \
  --out data/raw/ucf101

python -m src.pose.extract \
  --manifest data/raw/ucf101/manifest.csv \
  --out data/processed/ucf101 \
  --model yolov8n-pose.pt \
  --imgsz 640 --batch 32 --device 0

python scripts/build_features_from_keypoints.py --processed data/processed/ucf101
```

- 996 段、约 12 万帧；GPU 上通常 20–60 分钟（取决于卡和解码）。
- 显存不够就把 `--batch` 降到 16 或 8。
- 想先冒烟：加 `--limit 10`。
- 每段视频输出 `(T,17,2)` + `(T,17)` 的 `.npz`，体积很小。

## 4. 合并 MM-Fit + UCF101

```bash
python scripts/merge_manifests.py \
  --inputs data/processed/mmfit,data/processed/ucf101 \
  --out data/processed/combined

python -m src.train --data data/processed/combined \
  --config configs/bilstm_mmfit.yaml --out runs/combined_bilstm
```

合并后 5 类：squat / pushup / situp / jumping_jack / lunge。
UCF101 只提供前 4 类，situp 由 MM-Fit 提供。

## 5. 多 seed / 消融

```bash
for seed in 42 2024 3407; do
  python -m src.train --data data/processed/mmfit \
    --config configs/bilstm_mmfit.yaml \
    --out runs/mmfit_seed_${seed} --seed ${seed} --device 0
done
```

`--seed` 会覆盖 config 里的 seed。

## 6. 产物

```text
runs/<name>/
├─ best.pt              # state_dict + config + classes + scaler
├─ history.csv
├─ metrics.json
└─ confusion_matrix.csv
```

把整个 `runs/` 拷回本地即可；`best.pt` 里已经包含类别顺序、特征均值/方差和配置。

## 7. 常见问题

- **CUDA out of memory**：`--batch 16` 或 8；`--imgsz 512`。
- **OpenCV 读视频失败**：云上装 `ffmpeg`/`libgl1`；UCF101 是 AVI，需要解码器。
- **UCF101 类别不全**：`prepare_ucf101.py` 默认只取 4 类；要加 hard negatives 用 `--include-other`。
- **MM-Fit 没有 RGB**：当前用的是官方 `pose_2d.npy`，不是视频；如果要做视频 demo，再单独下 2–3 个 `wXX_rgb.mp4`。
- **Kaggle 下载 403**：配置 `~/.kaggle/kaggle.json`，或用浏览器下载后上传。
