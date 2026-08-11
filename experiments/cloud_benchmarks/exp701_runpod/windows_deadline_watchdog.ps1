[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $ProviderReceipt,

    [Parameter(Mandatory = $true)]
    [string] $CredentialFile,

    [Parameter(Mandatory = $true)]
    [DateTimeOffset] $DeleteAtUtc,

    [Parameter(Mandatory = $true)]
    [DateTimeOffset] $StopAtUtc,

    [Parameter(Mandatory = $true)]
    [string] $LogPath,

    [Parameter(Mandatory = $true)]
    [string] $WatchdogReceipt,

    [ValidateRange(5, 300)]
    [int] $PollSeconds = 30,

    [switch] $ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BaseUrl = "https://rest.runpod.io/v1"
$ExpectedSchema = "cuda-provider-lifecycle-receipt-v1"
$ExpectedLabel = "sair-exp701-full-fidelity"
$ExpectedSelector = "NVIDIA GeForce RTX 4090"
$MaximumProviderSeconds = 36000
$MaximumTotalUsd = [decimal] 9.00
$script:ApiKey = $null
$script:NoSleepEnabled = $false
$script:ObservedPodId = $null
$script:ObservedVolumeId = $null
$script:MutatingRequests = 0

function Write-WatchdogLog {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Event,
        [hashtable] $Fields = @{}
    )
    $record = [ordered]@{
        observed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        event = $Event
    }
    foreach ($entry in $Fields.GetEnumerator()) {
        $record[$entry.Key] = $entry.Value
    }
    $parent = Split-Path -Parent $LogPath
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    ($record | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $LogPath -Encoding UTF8
}

function Write-WatchdogReceipt {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Status,
        [string] $Failure = $null,
        [bool] $PodDeleteConfirmed = $false,
        [bool] $VolumeDeleteConfirmed = $false
    )
    $value = [ordered]@{
        schema_version = "exp701-windows-deadline-watchdog-v1"
        status = $Status
        observed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        provider_receipt = $ProviderReceipt
        delete_at_utc = $DeleteAtUtc.ToUniversalTime().ToString("o")
        stop_at_utc = $StopAtUtc.ToUniversalTime().ToString("o")
        expected_label = $ExpectedLabel
        expected_resource_selector = $ExpectedSelector
        pod_id = $script:ObservedPodId
        network_volume_id = $script:ObservedVolumeId
        mutating_requests_made = $script:MutatingRequests
        pod_delete_confirmed = $PodDeleteConfirmed
        network_volume_delete_confirmed = $VolumeDeleteConfirmed
        failure = $Failure
        credential_material_recorded = $false
    }
    $parent = Split-Path -Parent $WatchdogReceipt
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $temporary = "$WatchdogReceipt.tmp-$PID"
    ($value | ConvertTo-Json -Depth 8) + "`n" | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $WatchdogReceipt -Force
}

function Read-ApiKey {
    if (-not (Test-Path -LiteralPath $CredentialFile -PathType Leaf)) {
        throw "RunPod credential file is absent"
    }
    $matches = @(
        Get-Content -LiteralPath $CredentialFile | Where-Object {
            $_ -match '^\s*export\s+RUNPOD_API_KEY\s*=\s*(.+?)\s*$'
        }
    )
    if ($matches.Count -ne 1) {
        throw "RunPod credential file must contain exactly one exported API key"
    }
    $value = [string] $matches[0]
    $value = ($value -replace '^\s*export\s+RUNPOD_API_KEY\s*=\s*', '').Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match '[\r\n]') {
        throw "RunPod API key is empty or malformed"
    }
    return $value
}

function Read-ProviderReceipt {
    if (-not (Test-Path -LiteralPath $ProviderReceipt -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $ProviderReceipt -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Get-ReceiptValue {
    param(
        [Parameter(Mandatory = $true)]
        [object] $Receipt,
        [Parameter(Mandatory = $true)]
        [string] $Name
    )
    $property = $Receipt.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Test-DeletionEvent {
    param(
        [Parameter(Mandatory = $true)]
        [object] $Receipt,
        [Parameter(Mandatory = $true)]
        [string] $Name
    )
    $events = Get-ReceiptValue -Receipt $Receipt -Name "events"
    if ($null -eq $events) {
        return $false
    }
    return @($events | Where-Object { $_.event -eq $Name }).Count -gt 0
}

function Assert-BoundReceipt {
    param([Parameter(Mandatory = $true)][object] $Receipt)

    if ($Receipt.schema_version -ne $ExpectedSchema -or
        $Receipt.provider -ne "runpod" -or
        $Receipt.label -ne $ExpectedLabel -or
        $Receipt.resource_selector -ne $ExpectedSelector) {
        throw "provider receipt does not match the admitted EXP701 lifecycle"
    }
    if ([int] $Receipt.budget.max_provider_seconds -gt $MaximumProviderSeconds -or
        [decimal] $Receipt.budget.max_total_usd -gt $MaximumTotalUsd) {
        throw "provider receipt exceeds the watchdog authorization"
    }
    foreach ($candidate in @(
        (Get-ReceiptValue -Receipt $Receipt -Name "resource_id"),
        (Get-ReceiptValue -Receipt $Receipt -Name "network_volume_id")
    )) {
        if ($null -ne $candidate -and [string] $candidate -ne "" -and
            [string] $candidate -notmatch '^[a-z0-9]{6,40}$') {
            throw "provider receipt contains an invalid resource identifier"
        }
    }
}

function Invoke-RunPodRequest {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("GET", "DELETE")][string] $Method,
        [Parameter(Mandatory = $true)][string] $Path,
        [switch] $AllowNotFound
    )
    try {
        if ($Method -eq "DELETE") {
            $script:MutatingRequests += 1
        }
        return Invoke-WebRequest -UseBasicParsing -Method $Method -Uri ($BaseUrl + $Path) -Headers @{
            Authorization = "Bearer $script:ApiKey"
            Accept = "application/json"
            "User-Agent" = "sair-exp701-deadline-watchdog/1"
        } -TimeoutSec 30
    }
    catch {
        $status = $null
        if ($null -ne $_.Exception.Response) {
            $status = [int] $_.Exception.Response.StatusCode
        }
        if ($AllowNotFound -and $status -eq 404) {
            return $null
        }
        throw
    }
}

function Test-PodAbsent {
    param([Parameter(Mandatory = $true)][string] $PodId)
    $response = Invoke-RunPodRequest -Method GET -Path ("/pods?id=" + [Uri]::EscapeDataString($PodId)) -AllowNotFound
    if ($null -eq $response -or [string]::IsNullOrWhiteSpace($response.Content)) {
        return $true
    }
    $value = $response.Content | ConvertFrom-Json
    return @($value | Where-Object { $_.id -eq $PodId }).Count -eq 0
}

function Test-VolumeAbsent {
    param([Parameter(Mandatory = $true)][string] $VolumeId)
    $response = Invoke-RunPodRequest -Method GET -Path ("/networkvolumes/" + [Uri]::EscapeDataString($VolumeId)) -AllowNotFound
    return $null -eq $response
}

function Confirm-Deletion {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("pod", "volume")][string] $Kind,
        [Parameter(Mandatory = $true)][string] $Id
    )
    while ([DateTimeOffset]::UtcNow -lt $StopAtUtc) {
        $absent = if ($Kind -eq "pod") { Test-PodAbsent -PodId $Id } else { Test-VolumeAbsent -VolumeId $Id }
        if ($absent) {
            Write-WatchdogLog -Event "$Kind-delete-confirmed" -Fields @{ id = $Id }
            return $true
        }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Enable-NoSleep {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SairExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
    $continuousSystemAway = [uint32]::Parse(
        "80000043", [Globalization.NumberStyles]::HexNumber
    )
    $result = [SairExecutionState]::SetThreadExecutionState($continuousSystemAway)
    if ($result -eq 0) {
        throw "Windows no-sleep request failed"
    }
    $script:NoSleepEnabled = $true
}

function Disable-NoSleep {
    if ($script:NoSleepEnabled) {
        $continuous = [uint32]::Parse(
            "80000000", [Globalization.NumberStyles]::HexNumber
        )
        [void] [SairExecutionState]::SetThreadExecutionState($continuous)
        $script:NoSleepEnabled = $false
    }
}

try {
    if ($StopAtUtc -le $DeleteAtUtc) {
        throw "StopAtUtc must be later than DeleteAtUtc"
    }
    if (($StopAtUtc - $DeleteAtUtc).TotalSeconds -gt 900) {
        throw "watchdog deletion window cannot exceed 900 seconds"
    }
    $script:ApiKey = Read-ApiKey
    if ($ValidateOnly) {
        Write-WatchdogReceipt -Status "validated-no-mutation"
        Write-Output '{"status":"validated-no-mutation"}'
        exit 0
    }

    Enable-NoSleep
    Write-WatchdogLog -Event "watchdog-started" -Fields @{
        delete_at_utc = $DeleteAtUtc.ToUniversalTime().ToString("o")
        stop_at_utc = $StopAtUtc.ToUniversalTime().ToString("o")
    }

    while ([DateTimeOffset]::UtcNow -lt $DeleteAtUtc) {
        $receipt = Read-ProviderReceipt
        if ($null -ne $receipt) {
            Assert-BoundReceipt -Receipt $receipt
            $resourceId = Get-ReceiptValue -Receipt $receipt -Name "resource_id"
            $networkVolumeId = Get-ReceiptValue -Receipt $receipt -Name "network_volume_id"
            if ($null -ne $resourceId) {
                $script:ObservedPodId = [string] $resourceId
            }
            if ($null -ne $networkVolumeId) {
                $script:ObservedVolumeId = [string] $networkVolumeId
            }
            $podClean = Test-DeletionEvent -Receipt $receipt -Name "delete-confirmed"
            $volumeClean = Test-DeletionEvent -Receipt $receipt -Name "network-volume-delete-confirmed"
            if ($podClean -and $volumeClean) {
                Write-WatchdogLog -Event "controller-cleanup-observed"
                Write-WatchdogReceipt -Status "controller-cleaned-before-deadline" -PodDeleteConfirmed $true -VolumeDeleteConfirmed $true
                exit 0
            }
        }
        $remaining = [Math]::Max(1, [Math]::Ceiling(($DeleteAtUtc - [DateTimeOffset]::UtcNow).TotalSeconds))
        Start-Sleep -Seconds ([Math]::Min($PollSeconds, $remaining))
    }

    $receipt = Read-ProviderReceipt
    if ($null -eq $receipt) {
        throw "provider receipt is absent at the independent deletion deadline"
    }
    Assert-BoundReceipt -Receipt $receipt
    $resourceId = Get-ReceiptValue -Receipt $receipt -Name "resource_id"
    $networkVolumeId = Get-ReceiptValue -Receipt $receipt -Name "network_volume_id"
    $script:ObservedPodId = if ($null -eq $resourceId) { $null } else { [string] $resourceId }
    $script:ObservedVolumeId = if ($null -eq $networkVolumeId) { $null } else { [string] $networkVolumeId }
    Write-WatchdogLog -Event "independent-deletion-deadline-reached" -Fields @{
        pod_id = $script:ObservedPodId
        network_volume_id = $script:ObservedVolumeId
    }

    $podDeleted = $null -eq $script:ObservedPodId
    if (-not $podDeleted) {
        if (-not (Test-PodAbsent -PodId $script:ObservedPodId)) {
            [void] (Invoke-RunPodRequest -Method DELETE -Path ("/pods/" + [Uri]::EscapeDataString($script:ObservedPodId)))
        }
        $podDeleted = Confirm-Deletion -Kind pod -Id $script:ObservedPodId
    }

    $volumeDeleted = $null -eq $script:ObservedVolumeId
    if ($podDeleted -and -not $volumeDeleted) {
        if (-not (Test-VolumeAbsent -VolumeId $script:ObservedVolumeId)) {
            [void] (Invoke-RunPodRequest -Method DELETE -Path ("/networkvolumes/" + [Uri]::EscapeDataString($script:ObservedVolumeId)))
        }
        $volumeDeleted = Confirm-Deletion -Kind volume -Id $script:ObservedVolumeId
    }

    if (-not $podDeleted -or -not $volumeDeleted) {
        throw "independent deletion was not confirmed before StopAtUtc"
    }
    Write-WatchdogReceipt -Status "deadline-cleaned" -PodDeleteConfirmed $podDeleted -VolumeDeleteConfirmed $volumeDeleted
}
catch {
    $message = $_.Exception.Message
    try {
        Write-WatchdogLog -Event "watchdog-failed" -Fields @{ failure = $message }
        Write-WatchdogReceipt -Status "failed" -Failure $message
    }
    catch {
        # Preserve the original exception when even the local receipt path fails.
    }
    throw
}
finally {
    Disable-NoSleep
    $script:ApiKey = $null
}
