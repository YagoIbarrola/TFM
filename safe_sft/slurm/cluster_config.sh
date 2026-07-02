#!/bin/bash
# ===========================================================================
# Configuración del cluster UM (Slurm)
# ===========================================================================
# Cluster: 155.54.210.99
# Usuario: yago
# Particiones disponibles (actualizadas):
#   - cpu      : 40 vCPU, 128 GB RAM
#   - gpu      : slurm-gpu02, L4 24GB VRAM   → --gres=gpu:nvidia_l4:1
#   - gpuMax   : slurm-gpu01, H100 96GB VRAM → --gres=gpu:nvidia_h100_nvl:1
# (antes ambas GPUs estaban bajo la partición 'gpu'; admin las separó)
# Scratch por job (EFÍMERO, se borra al terminar): /scratch/slurm/$USER/$SLURM_JOB_ID
# Almacenamiento persistente:                      /slurm/home/$USER/
# ===========================================================================

# Particiones
# Cuenta slurm-ple+ tiene acceso a gpuMax (H100) tras la concesión del admin.
# Si se pierde acceso o gpuMax se satura, fallback a gpu/L4 cambiando partition+gres
# en 02_baseline_eval.sh, 03_train.sh, 04_eval_checkpoint.sh.
export PARTITION_GPU="gpuMax"         # H100 96GB
export PARTITION_GPU_FALLBACK="gpu"   # L4 24GB
export PARTITION_CPU="cpu"

# Recurso L40S (48 GB) — 6 nodos (slurm-gpu03..08), usados en el factorial.
# Comparten la partición 'gpu' con el L4 (gpu02); el GRES nvidia_l40s es lo que
# discrimina: con --gres=gpu:nvidia_l40s:1 el job sólo cae en un L40S, no en el L4.
export PARTITION_L40="gpu"                       # misma partición que el L4
export GRES_L40="gpu:nvidia_l40s:1"             # confirmado (L40S, 48GB)

# GRES (tipo de GPU). Sintaxis: gpu:<nombre>:<cantidad>
# Comprobado con: sinfo -o "%N %G"
#   slurm-gpu01      → gpu:nvidia_h100_nvl:1   (H100 96GB)
#   slurm-gpu02      → gpu:nvidia_l4:1         (L4 24GB)
#   slurm-gpu03..08  → gpu:nvidia_l40s:1       (L40S 48GB)
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
