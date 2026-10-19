<#
.SYNOPSIS
    Lanza Jupyter sobre el entorno GPU de Windows nativo.

.DESCRIPTION
    Comprueba que el entorno .venv-gpu existe, que PyTorch ve la RTX 4060 y que
    el equipo no esta con bateria (con bateria el firmware recorta la potencia
    de la GPU y los tiempos se duplican), y abre el notebook.

    Sustituye a ejecutar_notebook_gpu.sh, que servia para el entorno anterior
    basado en WSL2 + TensorFlow.
#>
$ErrorActionPreference = "Stop"

$raiz = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$python = Join-Path $raiz ".venv-gpu\Scripts\python.exe"
$notebook = Join-Path $raiz "PMC\notebooks\oldDataset\LosSimpsons_GPU_4060M.ipynb"

if (-not (Test-Path $python)) {
    throw "No existe $python. Crea el entorno siguiendo requirements-gpu-windows.txt."
}

Write-Host "Comprobando PyTorch y la GPU..." -ForegroundColor Cyan
& $python -c @"
import torch, sys
if not torch.cuda.is_available():
    sys.exit('PyTorch no ve la GPU. Revisa el driver NVIDIA y el wheel CUDA del entorno.')
mayor, menor = torch.cuda.get_device_capability(0)
print(f'  PyTorch {torch.__version__} (CUDA {torch.version.cuda})')
print(f'  {torch.cuda.get_device_name(0)}  CC {mayor}.{menor}')
"@
if ($LASTEXITCODE -ne 0) { throw "La comprobacion de la GPU fallo." }

# Con bateria, el limite de potencia de la GPU baja y el entrenamiento se ralentiza
# a la mitad. Merece la pena avisar antes de medir tiempos.
$bateria = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bateria -and $bateria.BatteryStatus -eq 1) {
    Write-Warning ("El equipo esta con bateria ({0}%). La GPU se limita en potencia y los " -f $bateria.EstimatedChargeRemaining)
    Write-Warning "tiempos por epoca pueden duplicarse. Conectalo a la corriente y pon el"
    Write-Warning "modo de energia en maximo rendimiento antes de medir."
}

$env:KERAS_BACKEND = "torch"
Write-Host "KERAS_BACKEND=torch" -ForegroundColor Cyan
Write-Host "Abriendo $notebook" -ForegroundColor Cyan
& $python -m jupyter notebook $notebook
