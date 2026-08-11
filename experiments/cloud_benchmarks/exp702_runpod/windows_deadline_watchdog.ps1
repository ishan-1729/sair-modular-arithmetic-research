param(
  [Parameter(Mandatory=$true)][string]$LifecycleSuffix,
  [Parameter(Mandatory=$true)][datetime]$LifecycleStartUtc,
  [Parameter(Mandatory=$true)][datetime]$CleanupDeadlineUtc,
  [Parameter(Mandatory=$true)][datetime]$HardDeadlineUtc,
  [Parameter(Mandatory=$true)][string]$ControllerReceipt,
  [Parameter(Mandatory=$true)][string]$OffPodRestoreReceipt,
  [Parameter(Mandatory=$true)][string]$WatchdogReadyReceipt,
  [Parameter(Mandatory=$true)][string]$DeletionCommandBase64
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($LifecycleSuffix -notmatch '^[a-z0-9][a-z0-9-]{2,63}$') { throw 'watchdog requires canonical suffix' }
if ($CleanupDeadlineUtc -ne $LifecycleStartUtc.AddSeconds(6600)) { throw 'cleanup deadline must be absolute start plus 6,600 seconds' }
if ($HardDeadlineUtc -ne $LifecycleStartUtc.AddSeconds(7200)) { throw 'hard deadline must be absolute start plus 7,200 seconds' }

$ready = [ordered]@{
  schema_version = 'exp702-watchdog-ready-v1'
  status = 'ready'
  suffix = $LifecycleSuffix
  deadline_seconds = 7200
}
$readyParent = Split-Path -Parent $WatchdogReadyReceipt
if ($readyParent) { [System.IO.Directory]::CreateDirectory($readyParent) | Out-Null }
$readyTemporary = "$WatchdogReadyReceipt.tmp-$PID"
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($readyTemporary, (($ready | ConvertTo-Json -Compress) + "`n"), $utf8WithoutBom)
Move-Item -LiteralPath $readyTemporary -Destination $WatchdogReadyReceipt -Force

$deletionJson = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($DeletionCommandBase64))
$DeletionCommand = $deletionJson | ConvertFrom-Json
if ($DeletionCommand.Count -ne 7 -or $DeletionCommand[0] -ne 'wsl.exe' -or
    $DeletionCommand[1] -ne '-d' -or $DeletionCommand[2] -ne 'Ubuntu-24.04' -or
    $DeletionCommand[3] -ne '--' -or $DeletionCommand[4] -ne 'bash' -or
    $DeletionCommand[5] -ne '-lc' -or $DeletionCommand[6] -notmatch 'source .*/sair-mod-challenge/bin/activate') {
  throw 'watchdog deletion command is not exact Ubuntu-24.04 WSL/venv execution'
}
$deleteProgram = $DeletionCommand[0]
$wslPrefix = @($DeletionCommand | Select-Object -Skip 1 -First 5)
$baseDeleteCommand = $DeletionCommand[6]
$bound = $null

function Quote-Bash([string]$Value) {
  $single = [char]39
  $replacement = [string]$single + [char]34 + [string]$single + [char]34 + [string]$single
  return [string]$single + $Value.Replace([string]$single, $replacement) + [string]$single
}
function ConvertTo-WslPath([string]$Path) {
  $resolved = [System.IO.Path]::GetFullPath($Path)
  if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') { throw "absolute Windows path required: $Path" }
  return "/mnt/$($Matches[1].ToLower())/$($Matches[2].Replace('\','/'))"
}
function Invoke-Deletion([string[]]$Arguments) {
  $suffix = @($Arguments | ForEach-Object { Quote-Bash $_ }) -join ' '
  $command = if ($suffix) { "$baseDeleteCommand $suffix" } else { $baseDeleteCommand }
  & $deleteProgram @wslPrefix $command
}

function Read-BoundReceipt {
  if (-not (Test-Path -LiteralPath $ControllerReceipt -PathType Leaf)) { return $null }
  $candidate = Get-Content -Raw -LiteralPath $ControllerReceipt | ConvertFrom-Json
  if ($candidate.suffix -ne $LifecycleSuffix -or [string]::IsNullOrWhiteSpace($candidate.owner_nonce)) {
    throw 'controller ownership receipt mismatch'
  }
  return $candidate
}

function Test-ControllerIdentity($Receipt) {
  if ($Receipt.controller_process_identity -notmatch '^(\d+):(\d+)$') {
    throw 'controller process identity is absent or malformed'
  }
  $controllerPid = $Matches[1]
  $startTicks = $Matches[2]
  $probe = "test -r /proc/$controllerPid/stat && test `"`$(awk '{print `$22}' /proc/$controllerPid/stat)`" = `"$startTicks`""
  & $deleteProgram @wslPrefix $probe | Out-Null
  return $LASTEXITCODE -eq 0
}

while ([datetime]::UtcNow -lt $CleanupDeadlineUtc) {
  $candidate = Read-BoundReceipt
  if ($null -ne $candidate) {
    if ($null -eq $bound) { $bound = $candidate }
    if ($candidate.owner_nonce -ne $bound.owner_nonce) { throw 'controller ownership nonce changed' }
    if ($candidate.exact_absence_confirmed -eq $true) { exit 0 }
    if (($candidate.pod_id -or $candidate.volume_id) -and -not (Test-ControllerIdentity $candidate)) {
      break
    }
  }
  Start-Sleep -Seconds 15
}

# At cleanup takeover, reconcile response-loss/controller-death creates by the
# controller's suffix-derived exact names and owner nonce before any deletion.
while ([datetime]::UtcNow -le $HardDeadlineUtc) {
  $candidate = Read-BoundReceipt
  if ($null -ne $candidate) {
    if ($null -eq $bound) { $bound = $candidate }
    if ($candidate.owner_nonce -ne $bound.owner_nonce) { throw 'cleanup takeover identity mismatch' }
  }
  if ($null -ne $bound) {
    $identityArgs = @('--owner-nonce', $bound.owner_nonce)
    if ($bound.pod_id) { $identityArgs += @('--pod-id', $bound.pod_id) }
    if ($bound.volume_id) { $identityArgs += @('--volume-id', $bound.volume_id) }
    $reconcileJson = Invoke-Deletion ($identityArgs + @('--reconcile-uncertain-create'))
    if ($LASTEXITCODE -eq 0) {
      $reconciled = ($reconcileJson | Out-String) | ConvertFrom-Json
      if ($reconciled.pod_id) { $bound.pod_id = $reconciled.pod_id }
      if ($reconciled.volume_id) { $bound.volume_id = $reconciled.volume_id }
    }
    $identityArgs = @('--owner-nonce', $bound.owner_nonce)
    if ($bound.pod_id) { $identityArgs += @('--pod-id', $bound.pod_id) }
    if ($bound.volume_id) { $identityArgs += @('--volume-id', $bound.volume_id) }
    # Pod deletion is unconditional.  Volume deletion remains restore gated.
    if ($bound.pod_id) { Invoke-Deletion ($identityArgs + @('--delete-pod', '--require-pod-absence')) | Out-Null }
    if ($bound.volume_id -and (Test-Path -LiteralPath $OffPodRestoreReceipt -PathType Leaf)) {
      $restore = Get-Content -Raw -LiteralPath $OffPodRestoreReceipt | ConvertFrom-Json
      if ($restore.status -eq 'restore-verified' -and $restore.owner_nonce -eq $bound.owner_nonce -and
          $restore.pod_id -eq $bound.pod_id -and $restore.volume_id -eq $bound.volume_id) {
        Invoke-Deletion ($identityArgs + @('--restore-receipt', (ConvertTo-WslPath $OffPodRestoreReceipt),
          '--delete-volume', '--require-volume-absence')) | Out-Null
      }
    }
    Invoke-Deletion ($identityArgs + @('--confirm-exact-absence')) | Out-Null
    if ($LASTEXITCODE -eq 0) { exit 0 }
  }
  Start-Sleep -Seconds 15
}
throw 'exact zero-resource confirmation failed at absolute 14,400-second deadline'
