#!/bin/bash
# ===========================================================================
# Configuración del cluster UM (Slurm)
# ===========================================================================
# Cluster: 155.54.210.99
# Usuario: yago
# Particiones disponibles:
#   - cpu         : 40 vCPU, 128 GB RAM
#   - gpu (H100)  : 40 vCPU, 128 GB RAM, H100 96GB VRAM  → --gres=gpu:nvidia_h100_nvl:1
#   - gpu (L4)    : 20 vCPU, 128 GB RAM, L4   24GB VRAM  → --gres=gpu:nvidia_l4:1
# Scratch por job (EFÍMERO, se borra al terminar): /scratch/slurm/$USER/$SLURM_JOB_ID
# Almacenamiento persistente:                      /slurm/home/$USER/
# ===========================================================================

# Particiones
export PARTITION_GPU="gpu"
export PARTITION_CPU="cpu"

# GRES (tipo de GPU). Sintaxis: gpu:<nombre>:<cantidad>
# Comprobado con: sinfo -o "%N %G"
#   slurm-gpu01 → gpu:nvidia_h100_nvl:1   (H100 96GB)
#   slurm-gpu02 → gpu:nvidia_l4:1         (L4 24GB)
export GRES_GPU="gpu:nvidia_h100_nvl:1"

# Rutas persistentes (sobreviven entre jobs)
# PROJECT_DIR = ruta absoluta a la carpeta safe_sft/ (donde están slurm/, data/, eval/, etc.)
# Si clonas el repo con: git clone <URL> safe_sft  → te queda /slurm/home/yago/safe_sft/safe_sft
export PROJECT_DIR="/slurm/home/yago/safe_sft/safe_sft"
export WORK_DIR="/slurm/home/yago/safe_sft_work"    # modelos, datos, resultados (fuera del repo)

# Token de HuggingFace (para Llama 3.2 gated).
# Mejor opción: ejecutar 'huggingface-cli login' una vez en sesión interactiva
# y dejar este vacío. Si quieres exportarlo, hazlo en tu ~/.bashrc.
export HF_TOKEN="${HF_TOKEN:-}"

# Conda
export CONDA_ENV="safe_sft"
export MINICONDA_DIR="$HOME/miniconda3"

# W&B (opcional)
export WANDB_PROJECT="safe_sft_tfm"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# ---------------------------------------------------------------------------
# Helper: activa el entorno conda. Instala miniconda si no existe.
# ---------------------------------------------------------------------------
activate_conda() {
    if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "ERROR: conda no encontrado. Ejecuta primero: sbatch slurm/00_setup.sh" >&2
        exit 1
    fi
    conda activate "$CONDA_ENV"
}
