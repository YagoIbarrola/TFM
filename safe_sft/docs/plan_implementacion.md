# Plan de Implementación y Experimentación

**Proyecto:** Degradación de la Seguridad en LLMs bajo Fine-tuning Supervisado  
**Modelo:** Llama 3.2 3B · **Infraestructura:** Slurm (1-2× A100/H100)  
**Fecha:** Mayo 2026

---

## 1. Estructura del Repositorio

```
safe_sft/
├── configs/
│   ├── base.yaml               # hiperparámetros comunes
│   ├── exp_a_pure_sft.yaml     # Experimento A: SFT puro
│   └── exp_b_mixed_sft.yaml    # Experimento B: SFT mixto 15%
├── data/
│   ├── prepare_alpaca.py       # descarga y tokeniza Alpaca
│   ├── prepare_beavertails.py  # filtra is_safe=True de BeaverTails-30k
│   └── mix_datasets.py        # mezcla Alpaca + BeaverTails al ratio indicado
├── train/
│   ├── train.py               # script principal de entrenamiento (TRL SFTTrainer)
│   └── callbacks.py           # SaveAndEvalCallback: guarda checkpoint y lanza eval
├── eval/
│   ├── run_harmbench.py       # lanza lm-evaluation-harness sobre un checkpoint
│   └── aggregate_results.py   # consolida resultados de todos los checkpoints en CSV
├── analysis/
│   ├── plot_curves.py         # genera gráficas de seguridad y task performance
│   └── find_inflection.py     # detecta punto de inflexión en curva A
├── slurm/
│   ├── 00_setup.sh            # entorno conda/pip en el cluster
│   ├── 01_prepare_data.sh     # job de preprocesado de datos
│   ├── 02_train_exp_a.sh      # job de entrenamiento Experimento A
│   ├── 02_train_exp_b.sh      # job de entrenamiento Experimento B
│   ├── 03_eval_checkpoint.sh  # job de evaluación por checkpoint (array job)
│   ├── 04_aggregate.sh        # job de agregación de resultados
│   └── pipeline.sh            # lanza toda la cadena con dependencias
└── results/
    ├── exp_a/                 # checkpoints + JSONs de evaluación
    └── exp_b/
```

---

## 2. Fases de Implementación

### Fase 0 — Setup del entorno (≈ 3–5 días)

**Objetivo:** entorno reproducible en el cluster, datasets descargados y tokenizados, baseline medido.

| Tarea | Herramienta | Entregable |
|---|---|---|
| Crear entorno conda con PyTorch + TRL + lm-evaluation-harness | `00_setup.sh` | `environment.yml` |
| Descargar Llama 3.2 3B desde HuggingFace | `huggingface-cli` | pesos en `/scratch/` |
| Preparar Alpaca (52k instrucciones) | `prepare_alpaca.py` | `data/alpaca_tokenized/` |
| Preparar BeaverTails-30k (filtrar `is_safe=True`) | `prepare_beavertails.py` | `data/beavertails_safe/` |
| Mezcla Alpaca + BeaverTails al 15% | `mix_datasets.py` | `data/mixed_15pct/` |
| Evaluación de seguridad **baseline** (modelo sin fine-tuning) | `run_harmbench.py` | `results/baseline.json` |

**Decisiones de diseño:**
- Se usa **BeaverTails-30k** (no el dataset completo de 330k) para mantener el coste de preprocesado manejable. El 15% de mezcla equivale a ~7700 ejemplos seguros entre las 52k de Alpaca.
- El baseline se evalúa con HarmBench completo (400+ prompts) para establecer el punto de partida de ambos experimentos.

---

### Fase 1 — Experimento A: SFT puro (≈ 1 semana)

**Objetivo:** curva de degradación de seguridad a lo largo de 3 epochs de fine-tuning sobre Alpaca sin ningún dato de seguridad.

**Hiperparámetros (config `exp_a_pure_sft.yaml`):**

```yaml
model: meta-llama/Llama-3.2-3B-Instruct
dataset: data/alpaca_tokenized/
epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4     # batch efectivo = 16
learning_rate: 2.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
peft:
  method: lora
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  lora_dropout: 0.05
save_strategy: steps
save_steps: 500
eval_on_save: true                  # lanza eval HarmBench en cada checkpoint
logging_steps: 10
output_dir: results/exp_a/
```

**Estimación de recursos y tiempo (A100 80GB):**
- ~3250 pasos/epoch × 3 epochs = **~9750 pasos totales**
- Checkpoint cada 500 pasos → **~19 checkpoints intermedios + 3 de epoch = 22 puntos de evaluación**
- Tiempo de entrenamiento: ~6–8 h en 1× A100
- Tiempo de evaluación HarmBench por checkpoint: ~20–40 min en 1× A100
- Evaluación total (array job 22 slots): ~8–15 h en paralelo con 2–4 slots simultáneos

---

### Fase 2 — Experimento B: SFT mixto (≈ 1 semana)

**Objetivo:** reproducir exactamente el experimento A pero con el 15% de BeaverTails (is_safe=True) mezclado en el dataset.

**Cambios respecto a Exp. A:**
- `dataset: data/mixed_15pct/` (Alpaca + ~7700 ejemplos seguros de BeaverTails)
- Ratio de mezcla controlado con semilla fija para reproducibilidad
- Resto de hiperparámetros **idénticos** para garantizar comparabilidad

**Mismo número de checkpoints y mismos puntos de evaluación que Exp. A.**

---

### Fase 3 — Análisis y visualización (≈ 3–5 días)

**Objetivo:** identificar el punto de inflexión y comparar las dos curvas.

| Análisis | Script | Output |
|---|---|---|
| Curva de seguridad epoch-by-epoch (A y B) | `plot_curves.py` | `figs/security_curves.pdf` |
| Curva de task performance (perplexity / instruction-following) | `plot_curves.py` | `figs/task_curves.pdf` |
| Detección del punto de inflexión en curva A | `find_inflection.py` | `results/inflection_report.json` |
| Tabla comparativa A vs B (Attack Success Rate, Harmless Rate) | `aggregate_results.py` | `results/summary_table.csv` |

**Métrica principal:** Attack Success Rate (ASR) en HarmBench, medido en cada checkpoint. Valores más bajos = modelo más seguro.

---

## 3. Jobs de Slurm — Diseño detallado

### 3.1 `00_setup.sh` — Setup del entorno

```bash
#!/bin/bash
#SBATCH --job-name=safe_sft_setup
#SBATCH --partition=<particion>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out

module load anaconda3
conda create -n safe_sft python=3.11 -y
conda activate safe_sft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft accelerate datasets bitsandbytes
pip install lm-eval[harmbench]
pip install wandb
```

### 3.2 `01_prepare_data.sh` — Preprocesado de datos

```bash
#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=<particion>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_data_%j.out

conda activate safe_sft
python data/prepare_alpaca.py --output_dir data/alpaca_tokenized/
python data/prepare_beavertails.py --output_dir data/beavertails_safe/ --split train
python data/mix_datasets.py \
    --base data/alpaca_tokenized/ \
    --safety data/beavertails_safe/ \
    --ratio 0.15 \
    --seed 42 \
    --output_dir data/mixed_15pct/
```

### 3.3 `02_train_exp_a.sh` — Entrenamiento Experimento A

```bash
#!/bin/bash
#SBATCH --job-name=train_exp_a
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_exp_a_%j.out

conda activate safe_sft
python train/train.py --config configs/exp_a_pure_sft.yaml
```

El script `train.py` guarda en `results/exp_a/checkpoints/` un fichero `checkpoint_list.txt`
con las rutas de todos los checkpoints generados. Este fichero es la entrada del array job de evaluación.

### 3.4 `03_eval_checkpoint.sh` — Evaluación HarmBench por checkpoint (array job)

```bash
#!/bin/bash
#SBATCH --job-name=eval_harmbench
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --array=0-21          # ajustar al número de checkpoints
#SBATCH --output=logs/eval_%A_%a.out

conda activate safe_sft
CHECKPOINT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/$EXP/checkpoint_list.txt)
python eval/run_harmbench.py \
    --model_path "$CHECKPOINT" \
    --output_dir "results/$EXP/eval_$(basename $CHECKPOINT)/"
```

Se lanza con `--dependency=afterok:<TRAIN_JOB_ID>` para ejecutarse sólo tras el entrenamiento.

### 3.5 `pipeline.sh` — Lanzamiento completo con dependencias

```bash
#!/bin/bash
# Lanza toda la cadena para un experimento dado
EXP=$1   # "exp_a" o "exp_b"
TRAIN_SCRIPT="slurm/02_train_${EXP}.sh"

DATA_JOB=$(sbatch --parsable slurm/01_prepare_data.sh)
echo "Data job: $DATA_JOB"

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB $TRAIN_SCRIPT)
echo "Train job: $TRAIN_JOB"

N_CHECKPOINTS=$(wc -l < results/$EXP/checkpoint_list.txt 2>/dev/null || echo 22)
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    --array=0-$((N_CHECKPOINTS-1)) \
    --export=EXP=$EXP \
    slurm/03_eval_checkpoint.sh)
echo "Eval array job: $EVAL_JOB"

AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:$EVAL_JOB \
    --export=EXP=$EXP \
    slurm/04_aggregate.sh)
echo "Aggregation job: $AGG_JOB"
```

Los dos experimentos (A y B) se pueden lanzar en paralelo si hay suficientes GPUs disponibles:
```bash
bash slurm/pipeline.sh exp_a &
bash slurm/pipeline.sh exp_b &
```

---

## 4. Pipeline de Evaluación — HarmBench

### Métricas extraídas por checkpoint

| Métrica | Descripción | Dirección |
|---|---|---|
| **ASR (Attack Success Rate)** | % de prompts adversariales con respuesta dañina | ↓ mejor |
| **Harmless Rate** | % de prompts con rechazo seguro | ↑ mejor |
| **Train Loss** | Loss del SFT en datos de entrenamiento | ↓ mejor (por separado) |
| **Eval Perplexity** | Perplejidad en set de validación de Alpaca | — (proxy de task performance) |

### `run_harmbench.py` — lógica interna

```python
# Ejecuta lm_eval con la tarea HarmBench estándar sobre un checkpoint cargado con LoRA
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={base_model},peft={checkpoint_path}",
    tasks=["harmbench"],
    batch_size=16,
    device="cuda",
)
# Guarda el JSON completo + extrae ASR y harmless_rate en un CSV resumen
```

---

## 5. Experimentos Adicionales (Opcionales)

Una vez completados los experimentos A y B, las siguientes extensiones tienen coste incremental bajo:

### Exp. C — Ablación del ratio de mezcla

Entrenar con ratios del 5%, 10%, 20% además del 15% para trazar la curva ratio → degradación.
Implementación: un job array sobre `configs/exp_c_ratio_{r}.yaml` variando sólo el parámetro `ratio`.

### Exp. D — Baseline Safe LoRA

Reimplementar el método de Hsu et al. (2024) como baseline competidor. Permite medir si Safe LoRA supera o iguala la mezcla simple de datos.

### Exp. E — Transferencia a Qwen 2.5 3B

Repetir el experimento A con Qwen 2.5 3B para validar si el punto de inflexión es específico de la arquitectura o generalizable.

---

## 6. Seguimiento y Reproducibilidad

### Logging
- **W&B** para métricas de entrenamiento (loss, lr, gradient norm) en tiempo real.
- CSV acumulativo en `results/exp_{a,b}/security_metrics.csv` con columna `step` como índice.

### Semillas fijas
```python
seed = 42  # en train.py, data loaders y splits de evaluación
```

### Artefactos públicos al finalizar
- Checkpoints relevantes (baseline, inflection point, epoch final) → HuggingFace Hub
- Código completo → GitHub público (Apache 2.0)
- Datos de evaluación → HuggingFace Datasets

---

## 7. Timeline Orientativo

| Semana | Trabajo |
|---|---|
| **S1** | Fase 0: setup cluster, descarga de pesos, preprocesado de datos, evaluación baseline |
| **S2** | Fase 1: lanzamiento Exp. A, depuración del pipeline de entrenamiento + evaluación |
| **S3** | Fase 2: lanzamiento Exp. B en paralelo, corrección de errores de Exp. A si los hay |
| **S4** | Fase 3: análisis, gráficas, detección del punto de inflexión, tabla comparativa |
| **S5+** | (Opcional) Experimentos C/D/E · Redacción de la memoria · Publicación del repositorio |

---

## 8. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Cuota de tiempo en cluster agotada durante eval | Media | Reducir frecuencia de checkpoints a 1 por epoch si la cuota es ajustada |
| HarmBench OOM en GPU 40GB | Baja | Usar `--device_map auto` + cuantización 8-bit para la inferencia de evaluación |
| Degradación no observable en 3 epochs | Baja | Qi et al. (2024) y Fraser et al. (2025) la miden en 1 epoch; si no aparece, revisar LR |
| Varianza alta entre runs | Media | Repetir cada experimento con 3 semillas; `aggregate_results.py` calcula media ± std |
| lm-evaluation-harness desactualizado en HarmBench | Baja | Fijar versión en `requirements.txt`, testear en Fase 0 antes de lanzar entrenamiento |
