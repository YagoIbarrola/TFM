#!/bin/bash
# ===========================================================================
# Job 00 — Setup del entorno (conda + dependencias + descarga del modelo)
# Lanzar UNA sola vez:  sbatch slurm/00_setup.sh
# Requiere conexión a internet desde el nodo.
# ===========================================================================
#SBATCH --job-name=safe_sft_setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"

# Aviso si el submission dir difiere del PROJECT_DIR (los logs habrán ido a otro sitio)
if [[ "${SLURM_SUBMIT_DIR:-$PWD}" != "$PROJECT_DIR" ]]; then
    echo "AVISO: SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}"
    echo "       PROJECT_DIR=$PROJECT_DIR"
    echo "       Los logs (--output/--error) están en \$SLURM_SUBMIT_DIR/logs/."
    echo "       Para tenerlos en \$PROJECT_DIR/logs/ relanza con:"
    echo "         cd $PROJECT_DIR && sbatch slurm/$(basename "$0")"
fi
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WORK_DIR"/{data,models,results/baseline,results/exp_a,results/exp_b}

echo "=== Setup safe_sft env ==="
echo "User:        $USER"
echo "Host:        $(hostname)"
echo "JobID:       $SLURM_JOB_ID"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "WORK_DIR:    $WORK_DIR"

# --------------------------------------------------------------------------
# 1. Instalar Miniconda si no existe
# --------------------------------------------------------------------------
if [[ ! -d "$MINICONDA_DIR" ]] && ! command -v conda &> /dev/null; then
    echo "=== Instalando Miniconda en $MINICONDA_DIR ==="
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh
    cd -
fi

# Cargar conda en este shell
if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

# Aceptar Terms of Service de los canales por defecto de Anaconda
# (requerido desde conda 24.x). 'true' por compatibilidad con versiones antiguas.
echo "=== Aceptando ToS de los canales de Anaconda ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# --------------------------------------------------------------------------
# 2. Crear entorno si no existe
# --------------------------------------------------------------------------
if conda env list | grep -qE "^$CONDA_ENV[[:space:]]"; then
    echo "Entorno '$CONDA_ENV' ya existe — actualizando dependencias"
else
    echo "=== Creando entorno conda: $CONDA_ENV ==="
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

conda activate "$CONDA_ENV"

# --------------------------------------------------------------------------
# 3. Instalar dependencias
# --------------------------------------------------------------------------
echo "=== Instalando PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Instalando dependencias del proyecto ==="
pip install -r "$PROJECT_DIR/requirements.txt"

# NOTA: flash-attn requiere nvcc/CUDA toolkit, no disponible en nodos CPU.
# El entrenamiento usará SDPA de PyTorch (suficientemente rápido).
# Si en el futuro quieres flash-attn, instálalo desde un srun en partición gpu:
#   srun --partition=gpu --gres=gpu:nvidia_h100_nvl:1 --pty bash
#   conda activate safe_sft && pip install flash-attn --no-build-isolation

# --------------------------------------------------------------------------
# 4. Login HuggingFace (necesario para Llama 3.2 gated)
# --------------------------------------------------------------------------
if [[ -n "$HF_TOKEN" ]]; then
    echo "=== Login en HuggingFace Hub ==="
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARNING: HF_TOKEN no definido."
    echo "         Ejecuta manualmente en sesión interactiva: hf auth login"
fi

# --------------------------------------------------------------------------
# 5. Descargar pesos del modelo base
# --------------------------------------------------------------------------
MODEL_DIR="$WORK_DIR/models/Llama-3.2-3B-Instruct"
if [[ -d "$MODEL_DIR" ]] && [[ -f "$MODEL_DIR/config.json" ]]; then
    echo "Modelo ya descargado en $MODEL_DIR — saltando"
else
    echo "=== Descargando meta-llama/Llama-3.2-3B-Instruct ==="
    hf download meta-llama/Llama-3.2-3B-Instruct --local-dir "$MODEL_DIR"
fi

# --------------------------------------------------------------------------
# 6. Verificación
# --------------------------------------------------------------------------
echo ""
echo "=== Verificación ==="
python - <<'EOF'
import sys
try:
    import torch, transformers, trl, peft, datasets, lm_eval
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}   CUDA disponible: {torch.cuda.is_available()}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  trl:          {trl.__version__}")
    print(f"  peft:         {peft.__version__}")
    print(f"  datasets:     {datasets.__version__}")
    print(f"  lm_eval:      {lm_eval.__version__}")
except Exception as e:
    print(f"FALLO en imports: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "=== Setup completo ==="
echo "Modelo en:  $MODEL_DIR"
echo "Siguiente:  sbatch slurm/01_prepare_data.sh"
