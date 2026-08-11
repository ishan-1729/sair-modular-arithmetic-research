param(
  [Parameter(Mandatory=$true)][string]$LifecycleSuffix,
  [Parameter(Mandatory=$true)][string]$LaunchDirectory,
  [Parameter(Mandatory=$true)][string]$ControlConfig,
  [string]$TransportFactory = 'experiments.cloud_benchmarks.exp702_runpod.runpod_transport:create_transport',
  [Parameter(Mandatory=$true)][string]$AuthorizationReceipt,
  [string]$WslDistribution = 'Ubuntu-24.04',
  [string]$RepositoryLinuxPath = '/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-651-visit-dilation-adapters',
  [string]$VenvLinuxPath = '/home/ishan/.venvs/sair-mod-challenge'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($LifecycleSuffix -notmatch '^[a-z0-9][a-z0-9-]{2,63}$') { throw 'canonical single lifecycle suffix required' }
if ($WslDistribution -ne 'Ubuntu-24.04') { throw 'EXP702 requires the reviewed Ubuntu-24.04 WSL distribution' }
$receipt = Join-Path $LaunchDirectory "live-$LifecycleSuffix.json"
$restoreReceipt = Join-Path $LaunchDirectory "offpod-restores\$LifecycleSuffix\latest.json"
$watchdogReadyReceipt = Join-Path $LaunchDirectory "watchdog\$LifecycleSuffix.ready.json"
$module = 'experiments.cloud_benchmarks.exp702_runpod.orchestrate'

function ConvertTo-WslPath([string]$Path) {
  $resolved = [System.IO.Path]::GetFullPath($Path)
  if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') { throw "absolute Windows path required: $Path" }
  return "/mnt/$($Matches[1].ToLower())/$($Matches[2].Replace('\','/'))"
}
function Quote-Bash([string]$Value) {
  $single = [char]39
  $replacement = [string]$single + [char]34 + [string]$single + [char]34 + [string]$single
  return [string]$single + $Value.Replace([string]$single, $replacement) + [string]$single
}
function New-WslPythonCommand([string[]]$PythonArguments) {
  $quoted = @($PythonArguments | ForEach-Object { Quote-Bash $_ }) -join ' '
  return "cd $(Quote-Bash $RepositoryLinuxPath) && source $(Quote-Bash "$VenvLinuxPath/bin/activate") && python $quoted"
}

$launchLinux = ConvertTo-WslPath $LaunchDirectory
$controlLinux = ConvertTo-WslPath $ControlConfig
$authorizationLinux = ConvertTo-WslPath $AuthorizationReceipt
$lifecycleStartUtc = [datetime]::UtcNow
$cleanupDeadlineUtc = $lifecycleStartUtc.AddSeconds(6600)
$hardDeadlineUtc = $lifecycleStartUtc.AddSeconds(7200)
$deletePythonArgs = @('-m', $module, 'provider-delete',
  '--control-config', $controlLinux, '--launch-dir', $launchLinux,
  '--transport-factory', $TransportFactory, '--authorization-receipt', $authorizationLinux,
  '--single-lifecycle-suffix', $LifecycleSuffix, '--controller-receipt', (ConvertTo-WslPath $receipt))
$deleteCommand = @('wsl.exe', '-d', $WslDistribution, '--', 'bash', '-lc',
  (New-WslPythonCommand $deletePythonArgs))
$deleteCommandBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(
  ($deleteCommand | ConvertTo-Json -Compress)))
$watchdog = Join-Path $PSScriptRoot 'windows_deadline_watchdog.ps1'
$watchdogArgs = @('-NoProfile', '-File', $watchdog, '-LifecycleSuffix', $LifecycleSuffix,
  '-LifecycleStartUtc', $lifecycleStartUtc.ToString('o'),
  '-CleanupDeadlineUtc', $cleanupDeadlineUtc.ToString('o'),
  '-HardDeadlineUtc', $hardDeadlineUtc.ToString('o'), '-ControllerReceipt', $receipt,
  '-OffPodRestoreReceipt', $restoreReceipt, '-WatchdogReadyReceipt', $watchdogReadyReceipt,
  '-DeletionCommandBase64', $deleteCommandBase64)
# Deadline ownership and an atomic ready receipt exist before execute-live.
Start-Process -FilePath 'powershell.exe' -ArgumentList $watchdogArgs -WindowStyle Hidden | Out-Null
for ($attempt = 0; $attempt -lt 30; $attempt++) {
  if (Test-Path -LiteralPath $watchdogReadyReceipt -PathType Leaf) {
    $ready = Get-Content -Raw -LiteralPath $watchdogReadyReceipt | ConvertFrom-Json
    if ($ready.schema_version -eq 'exp702-watchdog-ready-v1' -and $ready.status -eq 'ready' -and
        $ready.suffix -eq $LifecycleSuffix -and $ready.deadline_seconds -eq 7200) { break }
  }
  Start-Sleep -Seconds 1
}
if ($attempt -eq 30) { throw 'watchdog did not acknowledge readiness before create' }

$controllerArgs = @('-m', $module, 'execute-live', '--control-config', $controlLinux,
  '--launch-dir', $launchLinux, '--single-lifecycle-suffix', $LifecycleSuffix,
  '--transport-factory', $TransportFactory, '--authorization-receipt', $authorizationLinux,
  '--receipt', (ConvertTo-WslPath $receipt))
$controllerCommand = New-WslPythonCommand $controllerArgs
& wsl.exe -d $WslDistribution -- bash -lc $controllerCommand
if ($LASTEXITCODE -ne 0) { throw "EXP702 controller failed: $LASTEXITCODE" }
