$Env:CONDA_EXE = "/groups/CS156b/2024/gobeavers/cs156b/miniconda3/bin/conda"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "/groups/CS156b/2024/gobeavers/cs156b/miniconda3"
$Env:_CONDA_EXE = "/groups/CS156b/2024/gobeavers/cs156b/miniconda3/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs