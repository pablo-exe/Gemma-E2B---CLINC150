param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9.-]+$')]
    [string]$PodIp,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$SshPort,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\runpod_ed25519",
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = 'train_int8_t2_k128'
)

$ErrorActionPreference = 'Stop'
$repoDir = Split-Path -Parent $PSScriptRoot
$localPython = Join-Path $repoDir '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "SSH key missing: $KeyPath" }
if (-not (Test-Path -LiteralPath $localPython)) { throw 'Install the local project environment first.' }
$destinationDir = Join-Path $repoDir 'artifacts/teacher_logits'
New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
$remotePath = "root@${PodIp}:/workspace/finetuning/artifacts/teacher_logits/$RunName"
& scp -r -P $SshPort -i $KeyPath $remotePath $destinationDir
if ($LASTEXITCODE -ne 0) { throw 'SCP failed. Re-run to retry; keep the pod until verified.' }
$runDir = Join-Path $destinationDir $RunName
& $localPython (Join-Path $PSScriptRoot 'generate_teacher_logits.py') --verify $runDir
if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed. Keep the pod and retry the download.' }
Write-Output "Downloaded and verified: $runDir"
