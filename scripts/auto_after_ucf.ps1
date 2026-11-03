$ErrorActionPreference = "Stop"
$root = "D:\fitness-action"
$py = "C:\Users\lenovo\anaconda3\python.exe"
Set-Location $root
$log = Join-Path $root "runs\auto_after_ucf.log"
New-Item -ItemType Directory -Force (Join-Path $root "runs") | Out-Null
"[$(Get-Date -Format o)] waiting for UCF101 keypoints..." | Tee-Object -FilePath $log -Append

$deadline = (Get-Date).AddHours(3)
$manifest = Join-Path $root "data\processed\ucf101\keypoints_manifest.csv"
while (-not (Test-Path $manifest)) {
    if ((Get-Date) -gt $deadline) {
        "[$(Get-Date -Format o)] timeout waiting for $manifest" | Tee-Object -FilePath $log -Append
        exit 1
    }
    Start-Sleep -Seconds 60
}

"[$(Get-Date -Format o)] UCF keypoints ready; build 71-D features..." | Tee-Object -FilePath $log -Append
& $py scripts\build_features_from_keypoints.py --processed data\processed\ucf101 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format o)] merge MM-Fit + UCF101..." | Tee-Object -FilePath $log -Append
& $py scripts\merge_manifests.py --inputs data\processed\mmfit,data\processed\ucf101 --out data\processed\combined 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format o)] train combined..." | Tee-Object -FilePath $log -Append
& $py -m src.train --data data\processed\combined --config configs\bilstm_combined.yaml --out runs\combined_bilstm 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format o)] done." | Tee-Object -FilePath $log -Append
