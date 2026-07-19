[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "Validate",
        "RegisterProbe",
        "StartProbe",
        "ProbeStatus",
        "RemoveProbe",
        "RegisterFormal",
        "StartFormal",
        "FormalStatus",
        "RemoveFormal"
    )]
    [string]$Operation,

    [switch]$ConfirmFormalLaunch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ExpectedSid = "S-1-5-21-758172506-2016610370-2094743824-1001"
$Distribution = "Ubuntu-24.04"
$WslExecutable = "C:\Windows\System32\wsl.exe"
$TaskPath = "\"
$ProbeTaskName = "SAIR-EXP647-Persistence-Probe"
$FormalTaskName = "SAIR-EXP647-Formal-Retry3"
$WorkingDirectory = "D:\Projects\SAIR_mod_challenge"
$RepositoryWsl = "/mnt/d/Projects/SAIR_mod_challenge"
$LauncherWindows = Join-Path $PSScriptRoot "run-formal-retry3.sh"
$LauncherWsl = "/mnt/d/Projects/SAIR_mod_challenge/experiments/runs/exp-647-operand-swap-consistency/run-formal-retry3.sh"
$ProbeArguments = "-d $Distribution -- bash $LauncherWsl --persistence-probe"
$FormalArguments = "-d $Distribution -- bash $LauncherWsl"
$ScratchWindows = "D:\Projects\SAIR_mod_challenge\.scratch\exp-647-operand-swap-consistency"
$ProbeStartMarker = Join-Path $ScratchWindows "formal-windows-retry3-probe-start.json"
$ProbeHeartbeatMarker = Join-Path $ScratchWindows "formal-windows-retry3-probe-heartbeat.json"
$ProbeCompletionMarker = Join-Path $ScratchWindows "formal-windows-retry3-probe-completion.json"
$FormalStartMarker = Join-Path $ScratchWindows "formal-windows-retry3-start.json"
$FormalAlertMarker = Join-Path $ScratchWindows "formal-windows-retry3-alert.json"
$FormalExitMarker = Join-Path $ScratchWindows "formal-windows-retry3-exit.json"
$FormalLimit = [TimeSpan]::FromSeconds(22200)
$ProbeLimit = [TimeSpan]::FromSeconds(600)

function Get-CurrentLauncherHash {
    if (-not (Test-Path -LiteralPath $LauncherWindows -PathType Leaf)) {
        throw "Retry3 launcher is missing: $LauncherWindows"
    }
    return (Get-FileHash -LiteralPath $LauncherWindows -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TaskIfPresent {
    param([Parameter(Mandatory = $true)][string]$TaskName)

    return (Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue)
}

function Get-TaskOrThrow {
    param([Parameter(Mandatory = $true)][string]$TaskName)

    $task = Get-TaskIfPresent -TaskName $TaskName
    if ($null -eq $task) {
        throw "Scheduled task is not registered: $TaskName"
    }
    return $task
}

function Assert-HostPrerequisites {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $currentSid = $identity.User.Value
    if ($currentSid -ne $ExpectedSid) {
        throw "Unexpected Windows SID: $currentSid"
    }

    if (-not (Test-Path -LiteralPath $WslExecutable -PathType Leaf)) {
        throw "Pinned WSL executable is missing: $WslExecutable"
    }

    $schedule = Get-Service -Name "Schedule"
    $scheduleConfig = Get-CimInstance Win32_Service -Filter "Name='Schedule'"
    if ($schedule.Status -ne "Running" -or $scheduleConfig.StartMode -ne "Auto") {
        throw "Task Scheduler must be Automatic and Running"
    }

    $rawDistributions = & $WslExecutable --list --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to enumerate WSL distributions"
    }
    $distributions = @(
        $rawDistributions |
            ForEach-Object { ($_ -replace "`0", "").Trim() } |
            Where-Object { $_ -ne "" }
    )
    if ($distributions -notcontains $Distribution) {
        throw "Required WSL distribution is unavailable: $Distribution"
    }

    [void](Get-CurrentLauncherHash)
}

function Resolve-IdentitySid {
    param([Parameter(Mandatory = $true)][string]$Identity)

    try {
        return ([Security.Principal.SecurityIdentifier]$Identity).Value
    } catch {
        $account = [Security.Principal.NTAccount]$Identity
        return $account.Translate([Security.Principal.SecurityIdentifier]).Value
    }
}

function Get-CurrentRepositoryCommit {
    $commit = & $WslExecutable -d $Distribution -- git -C $RepositoryWsl rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve the current repository commit through WSL"
    }
    $resolved = (($commit | Select-Object -Last 1) -replace "`0", "").Trim()
    if ($resolved -notmatch "^[0-9a-f]{40}$") {
        throw "WSL returned an invalid repository commit: $resolved"
    }
    return $resolved
}

function New-RetryTaskDefinition {
    param(
        [Parameter(Mandatory = $true)][string]$Arguments,
        [Parameter(Mandatory = $true)][TimeSpan]$ExecutionLimit,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $action = New-ScheduledTaskAction `
        -Execute $WslExecutable `
        -Argument $Arguments `
        -WorkingDirectory $WorkingDirectory
    $principal = New-ScheduledTaskPrincipal `
        -UserId $ExpectedSid `
        -LogonType Interactive `
        -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit $ExecutionLimit `
        -RestartCount 0 `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries
    return New-ScheduledTask `
        -Action $action `
        -Principal $principal `
        -Settings $settings `
        -Description $Description
}

function Assert-TaskDefinition {
    param(
        [Parameter(Mandatory = $true)]$Task,
        [Parameter(Mandatory = $true)][string]$ExpectedArguments,
        [Parameter(Mandatory = $true)][TimeSpan]$ExpectedLimit
    )

    $actions = @($Task.Actions)
    if ($actions.Count -ne 1) {
        throw "Task must contain exactly one action: $($Task.TaskName)"
    }
    if ($actions[0].Execute -ne $WslExecutable) {
        throw "Task executable differs from the pinned wsl.exe path"
    }
    if ($actions[0].Arguments -ne $ExpectedArguments) {
        throw "Task arguments differ from the pinned action"
    }
    if ($actions[0].WorkingDirectory -ne $WorkingDirectory) {
        throw "Task working directory differs from the pinned repository path"
    }

    $resolvedPrincipalSid = Resolve-IdentitySid -Identity $Task.Principal.UserId
    if ($resolvedPrincipalSid -ne $ExpectedSid) {
        throw "Task principal SID differs from the current-user SID"
    }
    if ([string]$Task.Principal.LogonType -ne "Interactive") {
        throw "Task logon type must be Interactive"
    }
    if ([string]$Task.Principal.RunLevel -ne "Limited") {
        throw "Task run level must be Limited"
    }

    $triggerCount = if ($null -eq $Task.Triggers) { 0 } else { @($Task.Triggers).Count }
    if ($triggerCount -ne 0) {
        throw "Demand task must not have a trigger"
    }
    if (($Task.PSObject.Properties.Name -contains "TaskPath") -and
        -not [string]::IsNullOrEmpty([string]$Task.TaskPath) -and
        [string]$Task.TaskPath -ne $TaskPath) {
        throw "Task path differs from the pinned root task path"
    }
    if ([string]$Task.Settings.MultipleInstances -ne "IgnoreNew") {
        throw "Task multiple-instance policy must be IgnoreNew"
    }
    if ([int]$Task.Settings.RestartCount -ne 0) {
        throw "Task restart count must be zero"
    }
    if ([bool]$Task.Settings.DisallowStartIfOnBatteries) {
        throw "Task must allow battery start"
    }
    if ([bool]$Task.Settings.StopIfGoingOnBatteries) {
        throw "Task must continue across a battery transition"
    }
    if (-not [bool]$Task.Settings.Enabled) {
        throw "Task must be enabled"
    }
    if (-not [bool]$Task.Settings.AllowDemandStart) {
        throw "Task must allow an explicit demand start"
    }
    if ([bool]$Task.Settings.Hidden) {
        throw "Task must not be hidden"
    }
    if ([bool]$Task.Settings.RunOnlyIfIdle) {
        throw "Task must not depend on an idle condition"
    }
    if ([bool]$Task.Settings.RunOnlyIfNetworkAvailable) {
        throw "Task must not depend on network availability"
    }

    $expectedIsoLimit = [Xml.XmlConvert]::ToString($ExpectedLimit)
    if ([string]$Task.Settings.ExecutionTimeLimit -ne $expectedIsoLimit) {
        throw "Task execution limit differs from $expectedIsoLimit"
    }
}

function Register-DemandTask {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][string]$Arguments,
        [Parameter(Mandatory = $true)][TimeSpan]$ExecutionLimit,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if ($null -ne (Get-TaskIfPresent -TaskName $TaskName)) {
        throw "Refusing to replace existing scheduled task: $TaskName"
    }
    $definition = New-RetryTaskDefinition `
        -Arguments $Arguments `
        -ExecutionLimit $ExecutionLimit `
        -Description $Description
    Assert-TaskDefinition `
        -Task $definition `
        -ExpectedArguments $Arguments `
        -ExpectedLimit $ExecutionLimit
    $registeredHere = $false
    try {
        Register-ScheduledTask `
            -TaskPath $TaskPath `
            -TaskName $TaskName `
            -InputObject $definition |
            Out-Null
        $registeredHere = $true
        $registered = Get-TaskOrThrow -TaskName $TaskName
        Assert-TaskDefinition `
            -Task $registered `
            -ExpectedArguments $Arguments `
            -ExpectedLimit $ExecutionLimit
        return $registered
    } catch {
        if ($registeredHere) {
            Unregister-ScheduledTask `
                -TaskPath $TaskPath `
                -TaskName $TaskName `
                -Confirm:$false `
                -ErrorAction SilentlyContinue
        }
        throw
    }
}

function Read-JsonMarker {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Assert-CurrentProbeCompletion {
    $start = Read-JsonMarker -Path $ProbeStartMarker
    $completion = Read-JsonMarker -Path $ProbeCompletionMarker
    if ($null -eq $start -or $null -eq $completion) {
        throw "Persistence-probe start/completion evidence is incomplete"
    }
    $launcherHash = Get-CurrentLauncherHash
    $repositoryCommit = Get-CurrentRepositoryCommit
    if ([string]::IsNullOrWhiteSpace([string]$start.probe_nonce)) {
        throw "Persistence-probe nonce is missing"
    }
    if ($start.status -ne "running" -or [int]$start.probe_seconds -ne 180) {
        throw "Persistence-probe start marker is invalid"
    }
    if ($completion.status -ne "completed" -or [int]$completion.exit_code -ne 0) {
        throw "Persistence probe did not complete successfully"
    }
    if ([int]$completion.elapsed_seconds -ne 180) {
        throw "Persistence probe did not complete the declared 180 seconds"
    }
    if ($completion.probe_nonce -ne $start.probe_nonce) {
        throw "Persistence-probe start/completion nonces differ"
    }
    if ($completion.started_at -ne $start.started_at) {
        throw "Persistence-probe start timestamps differ"
    }
    if ($start.launcher_sha256 -ne $launcherHash -or
        $completion.launcher_sha256 -ne $launcherHash) {
        throw "Persistence-probe evidence has a stale launcher hash"
    }
    if ($start.repository_commit -ne $repositoryCommit -or
        $completion.repository_commit -ne $repositoryCommit) {
        throw "Persistence-probe evidence has a stale repository commit"
    }
    $startedAt = [DateTimeOffset]::Parse([string]$start.started_at)
    $completedAt = [DateTimeOffset]::Parse([string]$completion.completed_at)
    $observedSeconds = ($completedAt - $startedAt).TotalSeconds
    if ($observedSeconds -lt 170 -or $observedSeconds -gt 600) {
        throw "Persistence-probe observed duration is outside [170, 600] seconds: $observedSeconds"
    }
}

function Get-TaskStatusRecord {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][string]$ExpectedArguments,
        [Parameter(Mandatory = $true)][TimeSpan]$ExpectedLimit,
        [Parameter(Mandatory = $true)][ValidateSet("Probe", "Formal")][string]$Kind
    )

    $task = Get-TaskOrThrow -TaskName $TaskName
    Assert-TaskDefinition `
        -Task $task `
        -ExpectedArguments $ExpectedArguments `
        -ExpectedLimit $ExpectedLimit
    $info = Get-ScheduledTaskInfo -TaskPath $TaskPath -TaskName $TaskName
    $record = [ordered]@{
        task_name = $TaskName
        task_path = $task.TaskPath
        kind = $Kind.ToLowerInvariant()
        state = [string]$task.State
        last_run_time = $info.LastRunTime.ToString("o")
        last_task_result = [int64]$info.LastTaskResult
        next_run_time = $info.NextRunTime.ToString("o")
        launcher_sha256 = Get-CurrentLauncherHash
        executable = $task.Actions[0].Execute
        arguments = $task.Actions[0].Arguments
        working_directory = $task.Actions[0].WorkingDirectory
        principal_sid = Resolve-IdentitySid -Identity $task.Principal.UserId
        principal_task_scheduler_identity = $task.Principal.UserId
        logon_type = [string]$task.Principal.LogonType
        run_level = [string]$task.Principal.RunLevel
        restart_count = [int]$task.Settings.RestartCount
        multiple_instances = [string]$task.Settings.MultipleInstances
        execution_time_limit = [string]$task.Settings.ExecutionTimeLimit
        enabled = [bool]$task.Settings.Enabled
        allow_demand_start = [bool]$task.Settings.AllowDemandStart
        disallow_start_if_on_batteries = [bool]$task.Settings.DisallowStartIfOnBatteries
        stop_if_going_on_batteries = [bool]$task.Settings.StopIfGoingOnBatteries
        trigger_count = if ($null -eq $task.Triggers) { 0 } else { @($task.Triggers).Count }
    }
    if ($Kind -eq "Probe") {
        $record["probe_start"] = Read-JsonMarker -Path $ProbeStartMarker
        $record["probe_heartbeat"] = Read-JsonMarker -Path $ProbeHeartbeatMarker
        $record["probe_completion"] = Read-JsonMarker -Path $ProbeCompletionMarker
    } else {
        $record["formal_start"] = Read-JsonMarker -Path $FormalStartMarker
        $record["formal_alert"] = Read-JsonMarker -Path $FormalAlertMarker
        $record["formal_exit"] = Read-JsonMarker -Path $FormalExitMarker
    }
    return $record
}

function Remove-DemandTask {
    param([Parameter(Mandatory = $true)][string]$TaskName)

    $task = Get-TaskOrThrow -TaskName $TaskName
    if ([string]$task.State -eq "Running") {
        throw "Refusing to remove a running scheduled task: $TaskName"
    }
    Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
}

Assert-HostPrerequisites

switch ($Operation) {
    "Validate" {
        $probeDefinition = New-RetryTaskDefinition `
            -Arguments $ProbeArguments `
            -ExecutionLimit $ProbeLimit `
            -Description "Harmless persistence probe for SAIR EXP-647 retry3"
        Assert-TaskDefinition `
            -Task $probeDefinition `
            -ExpectedArguments $ProbeArguments `
            -ExpectedLimit $ProbeLimit
        $formalDefinition = New-RetryTaskDefinition `
            -Arguments $FormalArguments `
            -ExecutionLimit $FormalLimit `
            -Description "Full-fidelity SAIR EXP-647 formal challenger retry3"
        Assert-TaskDefinition `
            -Task $formalDefinition `
            -ExpectedArguments $FormalArguments `
            -ExpectedLimit $FormalLimit
        [ordered]@{
            status = "valid"
            in_memory_definitions_validated = $true
            current_sid = $ExpectedSid
            schedule_service = "Automatic/Running"
            distribution = $Distribution
            wsl_executable = $WslExecutable
            launcher_sha256 = Get-CurrentLauncherHash
            probe_task_present = $null -ne (Get-TaskIfPresent -TaskName $ProbeTaskName)
            formal_task_present = $null -ne (Get-TaskIfPresent -TaskName $FormalTaskName)
        } | ConvertTo-Json -Depth 8
    }
    "RegisterProbe" {
        $task = Register-DemandTask `
            -TaskName $ProbeTaskName `
            -Arguments $ProbeArguments `
            -ExecutionLimit $ProbeLimit `
            -Description "Harmless persistence probe for SAIR EXP-647 retry3"
        [ordered]@{status = "registered"; task_name = $task.TaskName} |
            ConvertTo-Json -Depth 4
    }
    "StartProbe" {
        $task = Get-TaskOrThrow -TaskName $ProbeTaskName
        Assert-TaskDefinition -Task $task -ExpectedArguments $ProbeArguments -ExpectedLimit $ProbeLimit
        if ([string]$task.State -ne "Ready") {
            throw "Probe task must be Ready before start; found $($task.State)"
        }
        foreach ($marker in @($ProbeStartMarker, $ProbeHeartbeatMarker, $ProbeCompletionMarker)) {
            if (Test-Path -LiteralPath $marker) {
                Remove-Item -LiteralPath $marker -Force
            }
        }
        Start-ScheduledTask -TaskPath $TaskPath -TaskName $ProbeTaskName
        [ordered]@{status = "start-requested"; task_name = $ProbeTaskName} |
            ConvertTo-Json -Depth 4
    }
    "ProbeStatus" {
        Get-TaskStatusRecord `
            -TaskName $ProbeTaskName `
            -ExpectedArguments $ProbeArguments `
            -ExpectedLimit $ProbeLimit `
            -Kind Probe |
            ConvertTo-Json -Depth 12
    }
    "RemoveProbe" {
        Assert-CurrentProbeCompletion
        $probeTask = Get-TaskOrThrow -TaskName $ProbeTaskName
        $probeInfo = Get-ScheduledTaskInfo -TaskPath $TaskPath -TaskName $ProbeTaskName
        if ([string]$probeTask.State -ne "Ready" -or [int64]$probeInfo.LastTaskResult -ne 0) {
            throw "Refusing to remove probe before Ready/LastTaskResult=0 evidence"
        }
        Remove-DemandTask -TaskName $ProbeTaskName
        [ordered]@{status = "removed"; task_name = $ProbeTaskName} |
            ConvertTo-Json -Depth 4
    }
    "RegisterFormal" {
        Assert-CurrentProbeCompletion
        $task = Register-DemandTask `
            -TaskName $FormalTaskName `
            -Arguments $FormalArguments `
            -ExecutionLimit $FormalLimit `
            -Description "Full-fidelity SAIR EXP-647 formal challenger retry3"
        [ordered]@{status = "registered"; task_name = $task.TaskName} |
            ConvertTo-Json -Depth 4
    }
    "StartFormal" {
        if (-not $ConfirmFormalLaunch.IsPresent) {
            throw "StartFormal requires -ConfirmFormalLaunch"
        }
        Assert-CurrentProbeCompletion
        $task = Get-TaskOrThrow -TaskName $FormalTaskName
        Assert-TaskDefinition -Task $task -ExpectedArguments $FormalArguments -ExpectedLimit $FormalLimit
        if ([string]$task.State -ne "Ready") {
            throw "Formal task must be Ready before start; found $($task.State)"
        }
        if ((Test-Path -LiteralPath $FormalStartMarker) -or
            (Test-Path -LiteralPath $FormalAlertMarker) -or
            (Test-Path -LiteralPath $FormalExitMarker)) {
            throw "Retry3 formal start/alert/exit marker already exists; refusing another start"
        }
        & $WslExecutable -d $Distribution -- bash $LauncherWsl --preflight-only
        if ($LASTEXITCODE -ne 0) {
            throw "WSL retry3 clean-tree preflight failed"
        }
        Start-ScheduledTask -TaskPath $TaskPath -TaskName $FormalTaskName
        [ordered]@{status = "start-requested"; task_name = $FormalTaskName} |
            ConvertTo-Json -Depth 4
    }
    "FormalStatus" {
        Get-TaskStatusRecord `
            -TaskName $FormalTaskName `
            -ExpectedArguments $FormalArguments `
            -ExpectedLimit $FormalLimit `
            -Kind Formal |
            ConvertTo-Json -Depth 12
    }
    "RemoveFormal" {
        Remove-DemandTask -TaskName $FormalTaskName
        [ordered]@{status = "removed"; task_name = $FormalTaskName} |
            ConvertTo-Json -Depth 4
    }
}
