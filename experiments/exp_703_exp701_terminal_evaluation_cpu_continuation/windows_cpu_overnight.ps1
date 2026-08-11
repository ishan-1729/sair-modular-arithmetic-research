param(
    [string]$Distribution = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"
$repoWindows = "D:\Projects\SAIR_mod_challenge\.scratch\exp-651-visit-dilation-adapters"
$receipt = Join-Path $repoWindows ".scratch\exp703-cpu-overnight\windows-lifecycle.json"
$scriptLinux = "/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-651-visit-dilation-adapters/experiments/exp_703_exp701_terminal_evaluation_cpu_continuation/launch_cpu_overnight.sh"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Exp703Power {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Receipt([hashtable]$Value) {
    $directory = Split-Path -Parent $receipt
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    $json = $Value | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($receipt, $json + [Environment]::NewLine, $utf8NoBom)
}

$started = [DateTimeOffset]::UtcNow
$keepAwakeResult = [Exp703Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
if ($keepAwakeResult -eq 0) {
    throw "SetThreadExecutionState failed before the overnight CPU run"
}

try {
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList @(
        "-d", $Distribution, "--", "bash", $scriptLinux
    ) -WindowStyle Hidden -PassThru
    Write-Receipt @{
        schema_version = "exp703-windows-lifecycle-v1"
        status = "running"
        started_at = $started.ToString("o")
        windows_pid = $process.Id
        distribution = $Distribution
        keep_awake = $true
    }
    $process.WaitForExit()
    $finished = [DateTimeOffset]::UtcNow
    $terminal = Test-Path (Join-Path $repoWindows ".scratch\exp703-cpu-overnight\terminal-summary.json")
    Write-Receipt @{
        schema_version = "exp703-windows-lifecycle-v1"
        status = $(if ($terminal -and $process.ExitCode -eq 0) { "completed" } else { "failed" })
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        elapsed_seconds = ($finished - $started).TotalSeconds
        windows_pid = $process.Id
        exit_code = $process.ExitCode
        terminal_summary_exists = $terminal
        distribution = $Distribution
        keep_awake = $true
    }
    exit $process.ExitCode
}
finally {
    [void][Exp703Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
