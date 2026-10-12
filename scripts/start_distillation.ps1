param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = 'full_gpu_bf16',
    [ValidatePattern('^[A-Za-z0-9_.-]+\.yaml$')]
    [string]$ConfigName = 'distillation_qlora_gpu.yaml',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repoDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoDir '.venv/Scripts/python.exe'
$runDir = Join-Path $repoDir "artifacts/distillation/$RunName"
$metadataPath = Join-Path $runDir 'run_metadata.json'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv is missing.' }
if ((Test-Path -LiteralPath $metadataPath) -and -not $Resume) {
    throw 'Run already exists. Use -Resume or select a new -RunName.'
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $runDir 'latest_checkpoint.json'))) {
    throw 'No complete checkpoint is available for resume.'
}
$launchPath = Join-Path $runDir 'launch.json'
if (Test-Path -LiteralPath $launchPath) {
    $oldLaunch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
    $candidateIds = @($oldLaunch.worker_pid, $oldLaunch.launcher_pid)
    $statusPath = Join-Path $runDir 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $oldStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $candidateIds += $oldStatus.pid
    }
    foreach ($candidateId in ($candidateIds | Where-Object { $_ } | Select-Object -Unique)) {
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $candidateId"
        if ($existing -and $existing.CommandLine -like '*train_distillation.py*') {
            throw "Training process $candidateId is already running."
        }
    }
}
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$stdout = Join-Path $runDir 'train.stdout.log'
$stderr = Join-Path $runDir 'train.stderr.log'
if ($Resume) {
    $suffix = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $runDir "resume-$suffix.stdout.log"
    $stderr = Join-Path $runDir "resume-$suffix.stderr.log"
}
$scriptPath = Join-Path $PSScriptRoot 'train_distillation.py'
$configPath = Join-Path $repoDir "configs/$ConfigName"
if (-not (Test-Path -LiteralPath $configPath)) { throw "Configuration missing: $configPath" }
$arguments = @('-u', "`"$scriptPath`"", '--config', "`"$configPath`"", '--run-dir', "`"$runDir`"")
if ($Resume) { $arguments += '--resume' }
$trainingProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
    -WorkingDirectory $repoDir -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$worker = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($trainingProcess.Id)" |
    Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*train_distillation.py*' } |
    Select-Object -First 1
$workerId = if ($worker) { $worker.ProcessId } else { $trainingProcess.Id }
@{
    launcher_pid = $trainingProcess.Id
    worker_pid = $workerId
    started_at = [DateTime]::UtcNow.ToString('o')
    run_dir = $runDir
    stdout = $stdout
    stderr = $stderr
    resume = [bool]$Resume
} | ConvertTo-Json | Set-Content -LiteralPath $launchPath -Encoding utf8
Write-Output "Training started: worker PID $workerId, launcher PID $($trainingProcess.Id)"
Write-Output "Run directory: $runDir"
Write-Output "Status: $(Join-Path $runDir 'status.json')"
& $pythonPath (Join-Path $PSScriptRoot 'set_training_performance.py') --pid $workerId `
    --report (Join-Path $runDir 'performance.json')
if ($LASTEXITCODE -ne 0) { Write-Warning 'Process power-policy adjustment failed; inspect throughput.' }
