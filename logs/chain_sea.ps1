# Espera a que termine el entrenamiento de fuego y lanza el de mar.
# Se ejecuta desacoplado para que la cadena sobreviva al cierre de la sesion.
$root = "D:\Workspace\learning_to_see"
$firePid = [int](Get-Content "$root\logs\train_fire.pid").Trim()

"[$(Get-Date -Format 'HH:mm:ss')] esperando a fuego (PID $firePid)..."
while (Get-Process -Id $firePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}
"[$(Get-Date -Format 'HH:mm:ss')] fuego termino."

if (Test-Path "$root\checkpoints\fire\latest_net_G.pth") {
    "  fuego OK: latest_net_G.pth existe"
} else {
    "  [!] fuego NO dejo checkpoint - se lanza mar igualmente"
}

Start-Sleep -Seconds 20  # que la GPU libere memoria

$a = @(
    "train.py",
    "--dataroot", "../data/sea_AB",
    "--name", "sea",
    "--model", "pix2pix",
    "--direction", "AtoB",
    "--dataset_mode", "aligned",
    "--batch_size", "1",
    "--load_size", "256",
    "--crop_size", "256",
    "--num_threads", "0",
    "--n_epochs", "65",
    "--n_epochs_decay", "65",
    "--save_epoch_freq", "10",
    "--save_latest_freq", "5000",
    "--checkpoints_dir", "../checkpoints"
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList $a `
    -WorkingDirectory "$root\pix2pix" `
    -RedirectStandardOutput "$root\logs\train_sea.log" `
    -RedirectStandardError "$root\logs\train_sea.err.log" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "$root\logs\train_sea.pid" -Encoding ascii
"[$(Get-Date -Format 'HH:mm:ss')] mar lanzado, PID $($p.Id)"
