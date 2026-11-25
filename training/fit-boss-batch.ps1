param(
    [Parameter(Mandatory=$true)][ValidatePattern('^train-[a-f0-9]{32}$')][string]$PreviousJob,
    [string[]]$Batch = @(),
    [string[]]$Development = @(),
    [ValidateSet('legacy', 'maneuver', 'maneuver-evasion', 'range-evasion')][string[]]$Modes = @('maneuver', 'maneuver-evasion'),
    [double]$RangeWeight,
    [switch]$BalanceMaps
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$request = Join-Path $root ".local/training/jobs/$PreviousJob/request.json"
$work = Get-Content -Raw -Encoding utf8 $request | ConvertFrom-Json
function Get-ReplayIds([string[]]$Paths) {
    foreach ($path in $Paths) {
        $entries = @(Get-Content -Raw -Encoding utf8 $path | ConvertFrom-Json)
        foreach ($entry in $entries) {
            if ($null -eq $entry.id) { throw "回放清单缺少 id $path" }
            $entry.id
        }
    }
}
$trainIds = @($work.params.train_ids) + @(Get-ReplayIds $Batch)
$holdoutIds = @($work.params.holdout_ids) + @(Get-ReplayIds $Development)
$inputs = foreach ($id in $trainIds + $holdoutIds) {
    '--input'
    Join-Path $root ".local/online/replays/$id.json.gz"
}
$tank = Join-Path $root '.venv/Scripts/tank.exe'
$oldThreads = $env:OPENBLAS_NUM_THREADS

Push-Location $root
try {
    $env:OPENBLAS_NUM_THREADS = '1'
    foreach ($mode in $Modes) {
        $params = @{
            train_ids = $trainIds
            holdout_ids = $holdoutIds
            epochs = $work.params.epochs
            policy = $mode
        }
        if ($mode -eq 'range-evasion') {
            if ($PSBoundParameters.ContainsKey('RangeWeight')) { $params.range_weight = $RangeWeight }
            elseif ($null -ne $work.params.range_weight) { $params.range_weight = $work.params.range_weight }
            $params.balance_maps = if ($PSBoundParameters.ContainsKey('BalanceMaps')) { [bool]$BalanceMaps } else { [bool]$work.params.balance_maps }
        }
        $params = $params | ConvertTo-Json -Compress
        & $tank jobs start fit_boss_sparring.py --config $work.config_id @inputs --params $params --json
        if ($LASTEXITCODE -ne 0) { throw "训练作业启动失败 $mode" }
    }
}
finally {
    $env:OPENBLAS_NUM_THREADS = $oldThreads
    Pop-Location
}
