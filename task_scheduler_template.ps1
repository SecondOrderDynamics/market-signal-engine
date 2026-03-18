param(
    [string]$TaskName = "DailyScanRunner",
    [string]$WorkDir = "C:\Users\jkers\OneDrive\Desktop\CODE",
    [string]$PythonExe = "py",
    [string]$RunTime = "06:30",
    [int]$RetryCount = 3,
    [int]$RetryIntervalMinutes = 15,
    [switch]$RegisterCleanupTask
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Registering task '$TaskName'..."

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "daily_scan_runner.py --quiet" `
    -WorkingDirectory $WorkDir

$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($RunTime, "HH:mm", $null))

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount $RetryCount `
    -RestartInterval (New-TimeSpan -Minutes $RetryIntervalMinutes) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Runs daily_scan_runner.py in quiet mode with retry policy and timestamped logs."

Write-Host "Task '$TaskName' registered."
Write-Host "Working directory: $WorkDir"
Write-Host "Run time: $RunTime"
Write-Host "Retries: $RetryCount every $RetryIntervalMinutes minute(s)"

if ($RegisterCleanupTask) {
    $cleanupName = "$TaskName-LogCleanup"
    Write-Host "Registering optional cleanup task '$cleanupName'..."

    $cleanupCmd = @'
$root = "C:\Users\jkers\OneDrive\Desktop\CODE\logs"
if (Test-Path $root) {
    Get-ChildItem $root -Filter "daily_scan_*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem $root -Filter "daily_summary_*.json" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | Remove-Item -Force -ErrorAction SilentlyContinue
}
'@

    $cleanupAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -Command `$ErrorActionPreference='Stop'; $cleanupCmd"

    $cleanupTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 03:00
    $cleanupSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    Register-ScheduledTask `
        -TaskName $cleanupName `
        -Action $cleanupAction `
        -Trigger $cleanupTrigger `
        -Settings $cleanupSettings `
        -Description "Removes daily runner logs/summary files older than 30 days."

    Write-Host "Cleanup task '$cleanupName' registered."
}
