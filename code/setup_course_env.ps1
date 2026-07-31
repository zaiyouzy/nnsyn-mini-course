$courseRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvActivate = Join-Path $courseRoot ".venv-nnsyn\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "The course virtual environment was not found at: $venvActivate"
}

. $venvActivate

$env:nnsyn_origin_dataset = Join-Path $courseRoot "dataset\nnsyn_origin\Task1_AB_3cases"
$env:nnUNet_raw = Join-Path $courseRoot "nnsyn_workspace\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $courseRoot "nnsyn_workspace\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $courseRoot "nnsyn_workspace\nnUNet_results"

Write-Host "nnsyn course environment is ready." -ForegroundColor Green
Write-Host "Python:                 $((Get-Command python).Source)"
Write-Host "nnsyn_origin_dataset:   $env:nnsyn_origin_dataset"
Write-Host "nnUNet_raw:             $env:nnUNet_raw"
Write-Host "nnUNet_preprocessed:    $env:nnUNet_preprocessed"
Write-Host "nnUNet_results:         $env:nnUNet_results"