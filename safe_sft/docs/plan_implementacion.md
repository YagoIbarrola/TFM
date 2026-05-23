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
│   ├── exp_a_alpaca_pure.yaml  # Exp A: Alpaca puro (general · sin safety)
│   ├── exp_b_alpaca_mixed.yaml # Exp B: Alpaca + 15% safety
│   ├── exp_c_math_pure.yaml    # Exp C: MetaMathQA puro (narrow · sin safety)
│   └── exp_d_math_mixed.yaml   # Exp D: MetaMathQA + 15% safety
├── data/
│   ├── prepare_alpaca.py       # descarga y tokeniza Alpaca
│   ├── prepare_metamath.py     # descarga y submuestrea MetaMathQA a 52k
│   ├── prepare_beavertails.py  # filtra is_safe=True de BeaverTails-30k
│   └── mix_datasets.py        # mezcla {Alpaca|Math} + BeaverTails al ratio indicado
├── train/
│   ├── train.py               # script principal de entrenamiento (TRL SFTTrainer)
│   └── callbacks.py           # SaveAndEvalCallback: guarda checkpoint y lanza eval
├── eval/
│   ├── run_harmbench.py       # eval DirectRequest (curva fina, todos los checkpoints)
│   ├── run_human_jailbreaks.py # eval con HumanJailbreaks de HarmBench (mismo coste)
│   ├── run_pair.py            # eval PAIR sobre checkpoints clave (más caro)
│   ├── run_capability.py      # eval capacidad (Alpaca-eval / GSM8K accuracy)
│   └── aggregate_results.py   # consolida resultados de todos los checkpoints en CSV
├── analysis/
│   ├── plot_curves.py         # genera gráficas de seguridad y task performance
│   ├── plot_robustness.py     # gráficas robustez (DirectRequest vs HJ vs PAIR)
│   └── find_inflection.py     # detecta punto de inflexión en curva A
├── slurm/
│   ├── 00_setup.sh            # entorno conda/pip en el cluster
│   ├── 01_prepare_data.sh     # job de preprocesado de datos
│   ├── 02_train_exp_a.sh      # job entrenamiento Exp A (Alpaca puro)
│   ├── 02_train_exp_b.sh      # job entrenamiento Exp B (Alpaca + safety)
│   ├── 02_train_exp_c.sh      # job entrenamiento Exp C (Math puro)
│   ├── 02_train_exp_d.sh      # job entrenamiento Exp D (Math + safety)
│   ├── 03_eval_checkpoint.sh  # job de evaluación DirectRequest por checkpoint (array)
│   ├── 04_aggregate.sh        # job de agregación de resultados
│   ├── 05_eval_human_jb.sh    # job array de eval con HumanJailbreaks (todos los chkpt)
│   ├── 06_eval_pair.sh        # job de eval con PAIR (solo snapshots clave)
│   └── pipeline.sh            # lanza toda la cadena con dependencias
└── results/
    ├── exp_a/                 # checkpoints + JSONs de evaluación
    └── exp_b/
```

---

## 2. Diseño Experimental — Factorial 2×2

El TFM se estructura como un experimento factorial 2×2 sobre dos ejes:

|  | **Sin safety mix** | **Con 15% safety mix** |
|---|---|---|
| **Tarea general (Alpaca)** | **Exp A** — replica Qi et al. (2024) | **Exp B** — pregunta original del TFM |
| **Tarea narrow (MetaMathQA)** | **Exp C** — testa Chen et al. (2025) | **Exp D** — generalización del mix |

Este diseño permite responder dos preguntas con un único entorno experimental:

1. **¿El dominio importa?** Comparar A vs C (y B vs D) revela si la degradación es función del dominio o del acto de SFT en sí.
2. **¿La mezcla funciona uniformemente?** Comparar A vs B (y C vs D) mide la efectividad del 15% de safety mix. Si funciona en Alpaca pero no en Math, la mitigación es dependiente del dominio.

**Hipótesis competidoras:**
- **H1 (Chen et al. 2025):** Math degrada más que Alpaca por menor solapamiento con datos de safety alignment.
- **H2 (Bianchi et al. 2024):** Dominios narrow degradan más por baja diversidad.
- **H3 (nula):** Degradación equivalente → el problema es el SFT en sí, no el contenido.

## 3. Fases de Implementación

### Fase 0 — Setup del entorno (≈ 3–5 días)

**Objetivo:** entorno reproducible en el cluster, datasets descargados y tokenizados, baseline medido.

| Tarea | Herramienta | Entregable |
|---|---|---|
| Crear entorno conda con PyTorch + TRL + lm-evaluation-harness | `00_setup.sh` | `environment.yml` |
| Descargar Llama 3.2 3B desde HuggingFace | `huggingface-cli` | pesos en `/scratch/` |
| Preparar Alpaca (52k instrucciones) | `prepare_alpaca.py` | `data/alpaca_tokenized/` |
| Preparar MetaMathQA submuestreado a 52k | `prepare_metamath.py` | `data/metamath_52k/` |
| Preparar BeaverTails-30k (filtrar `is_safe=True`) | `prepare_beavertails.py` | `data/beavertails_safe/` |
| Mezcla Alpaca + BeaverTails al 15% | `mix_datasets.py` | `data/alpaca_mixed_15/` |
| Mezcla MetaMath + BeaverTails al 15% | `mix_datasets.py` | `data/math_mixed_15/` |
| Eval seguridad **baseline** (modelo sin fine-tuning) | `run_harmbench.py` | `results/baseline/harmbench.json` |
| Eval capacidad baseline (GSM8K, Alpaca-eval) | `run_capability.py` | `results/baseline/capability.json` |

**Decisiones de diseño:**
- **MetaMathQA** se elige sobre GSM8K (7.5k es muy pequeño) y MATH (12k, demasiado competitivo). Submuestreamos a 52k para igualar Alpaca y garantizar que el efecto observado no se debe al tamaño del dataset.
- El 15% de mezcla equivale a ~9170 ejemplos seguros entre los 52k de tarea, en ambos experimentos B y D (consistencia).
- El baseline se evalúa con HarmBench completo (~300 prompts directos + HJ + PAIR) y con benchmarks de capacidad (GSM8K para math, Alpaca-eval para general).

---

### Fase 1 — Eje Alpaca: Experimentos A y B (≈ 1-2 semanas)

**Objetivo:** trazar las curvas de degradación de seguridad sobre tarea general, con y sin mezcla de safety. Es el backbone del TFM y replica/extiende Qi et al. (2024).

**Hiperparámetros comunes (config `exp_a_alpaca_pure.yaml`, idéntico para B salvo `dataset_path`):**

```yaml
model: meta-llama/Llama-3.2-3B-Instruct
dataset_path: data/alpaca/              # B: data/alpaca_mixed_15/
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
output_dir: results/exp_a/          # B: results/exp_b/
```

**Estimación de recursos y tiempo (H100 96GB):**
- ~3250 pasos/epoch × 3 epochs = **~9750 pasos totales** (algo más en Exp B por el 15% extra)
- Checkpoint cada 500 pasos → **~22 puntos de evaluación por experimento**
- Tiempo de entrenamiento: ~4–6 h por experimento en H100
- Tiempo de evaluación HarmBench DirectRequest por checkpoint: ~10–20 min en H100
- Evaluación total (array job 22 slots): ~5–8 h en paralelo

---

### Fase 2 — Eje Math: Experimentos C y D (≈ 1-2 semanas)

**Objetivo:** repetir el diseño de Fase 1 sobre MetaMathQA en lugar de Alpaca, manteniendo todo lo demás idéntico. Esto cierra el factorial 2×2.

**Cambios respecto a Fase 1:**
- `dataset_path: data/metamath_52k/` (C) o `data/math_mixed_15/` (D)
- **Mismos hiperparámetros, misma semilla, mismo schedule de checkpoints, misma evaluación.**
- La única variable independiente entre Fase 1 y Fase 2 es el **dominio del dataset de entrenamiento**.

**Comparabilidad:** la igualdad de tamaño (52k) y de tokens efectivos por epoch garantiza que cualquier diferencia observada en las curvas A vs C (o B vs D) es atribuible al dominio, no al volumen de SFT.

**Métrica de capacidad adicional para C/D:** accuracy en GSM8K (separado del conjunto de entrenamiento) en cada checkpoint, para verificar que el SFT efectivamente está mejorando la capacidad matemática objetivo.

---

### Fase 2.5 — Evaluación de robustez (≈ 1 semana)

**Objetivo:** complementar las curvas DirectRequest con métricas bajo ataques realistas, separando *alineación nominal* (¿rechaza preguntas dañinas?) de *robustez* (¿resiste jailbreaks?).

**Motivación:** un modelo alineado tiene un ASR DirectRequest muy bajo (≈1-5%), lo que deja poco margen para medir degradación. Bajo ataques fuertes el ASR baseline sube a 20-40%, haciendo visible la trayectoria. Además, es lo que cualquier atacante real haría.

#### Capa 1 — HumanJailbreaks (coste ≈ DirectRequest)

Prompts crafteados por humanos (estilo DAN, role-play, etc.) que distribuye HarmBench. La inferencia es idéntica a DirectRequest: solo se prependen los jailbreaks al behavior y se generan respuestas.

- **Evaluación en todos los checkpoints** (≈22 por experimento), mismo array job que `03_eval_checkpoint.sh` pero con flag `--attack=human_jailbreaks`.
- **Coste extra:** ≈ ×1 del coste DirectRequest (la única diferencia es que cada behavior se prueba con N jailbreaks distintos y se toma el peor caso).
- Implementación en [eval/run_human_jailbreaks.py](safe_sft/eval/run_human_jailbreaks.py).

#### Capa 2 — PAIR snapshots (coste medio)

[PAIR](https://arxiv.org/abs/2310.08419) usa un LLM atacante que itera el prompt hasta conseguir jailbreak. Se evalúa solo en **4 checkpoints por experimento**: baseline, fin epoch 1, fin epoch 2, fin epoch 3.

| Hiperparámetro | Valor |
|---|---|
| LLM atacante | Llama-3.1-8B-Instruct (local en H100) |
| LLM juez | mismo atacante + clasificador HarmBench |
| N iteraciones por behavior | 5 |
| N streams paralelos | 5 |
| Behaviors evaluados | subconjunto de 100 (los más informativos) |

**Coste estimado:** ~8 h GPU por snapshot × 4 snapshots × 2 experimentos = **~64 h GPU adicionales** (manejable en H100 96GB).

Implementación en [eval/run_pair.py](safe_sft/eval/run_pair.py).

#### Estructura de outputs

Cada checkpoint produce ahora hasta 3 JSONs de evaluación:

```
results/exp_a/eval_step1000/
├── harmbench_direct.json        # DirectRequest (ya implementado)
├── harmbench_humanjb.json       # HumanJailbreaks (Fase 2.5 — capa 1)
└── harmbench_pair.json          # PAIR (solo en snapshots clave)
```

#### Criterios de éxito

- **Curva HJ separa visiblemente A y B**: si la mezcla del 15% protege contra DirectRequest pero no contra HumanJailbreaks, el TFM gana matiz.
- **PAIR confirma o contradice las curvas finas**: si PAIR muestra degradación incluso cuando DirectRequest no, eso es un resultado interesante por sí solo.

---

### Fase 3 — Análisis factorial 2×2 (≈ 5–7 días)

**Objetivo:** comparar las cuatro condiciones (A, B, C, D) bajo los tres niveles de ataque y aislar los efectos del **dominio** y del **safety mix**.

| Análisis | Script | Output |
|---|---|---|
| Curvas de seguridad de los 4 experimentos (DirectRequest) | `plot_curves.py` | `figs/curves_direct_2x2.pdf` |
| Curvas de robustez (HumanJailbreaks) | `plot_robustness.py` | `figs/curves_hj_2x2.pdf` |
| Snapshots PAIR superpuestos a las curvas | `plot_robustness.py` | `figs/curves_pair_2x2.pdf` |
| Curva de capacidad por dominio (Alpaca-eval, GSM8K) | `plot_curves.py` | `figs/capability_curves.pdf` |
| Detección del punto de inflexión por experimento y ataque | `find_inflection.py` | `results/inflection_report.json` |
| **Tabla factorial 2×2** (rows: dominio, cols: mix; cells: ΔASR final) | `aggregate_results.py` | `results/factorial_table.csv` |
| **Test estadístico del efecto del dominio** (ANOVA o equivalente) | `aggregate_results.py` | `results/effects_test.json` |

**Métricas principales:**
- **ΔASR_epoch3 = ASR(checkpoint final) − ASR(baseline)** — la magnitud total de degradación.
- **t_infl = step donde dASR/dstep supera umbral** — momento del colapso.
- **Capability gain** = mejora en la métrica de tarea objetivo (GSM8K para C/D, Alpaca-eval para A/B).

**Resultados esperados (predicciones para validar):**
- Si **H1 (Chen)** se confirma: ΔASR(C) > ΔASR(A) y ΔASR(D) > ΔASR(B) → math degrada más en ambas condiciones.
- Si **H2 (Bianchi)** se confirma: efecto del dominio se atenúa cuando hay safety mix (B−A ≠ D−C) → la mezcla compensa el efecto narrow.
- Si **H3 (nula)** se confirma: ambas curvas A/C convergen → la degradación es invariante al dominio.

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

### 3.5 `05_eval_human_jb.sh` — HumanJailbreaks (array job, todos los checkpoints)

Idéntico a `03_eval_checkpoint.sh` pero invocando `run_human_jailbreaks.py`. Comparte tiempo y memoria con DirectRequest (la inferencia es del mismo orden de magnitud).

```bash
#SBATCH --array=0-21
#SBATCH --gres=H_100_NVL
#SBATCH --time=02:00:00
python eval/run_human_jailbreaks.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CHECKPOINT" \
    --output_path "results/$EXP/eval_$(basename $CHECKPOINT)/harmbench_humanjb.json" \
    --jailbreaks_path "$WORK_DIR/data/human_jailbreaks.json"
```

### 3.6 `06_eval_pair.sh` — PAIR (solo snapshots clave)

Job que evalúa un único checkpoint con PAIR. Se lanza manualmente para 4 checkpoints por experimento (baseline, fin epoch 1/2/3).

```bash
#SBATCH --gres=H_100_NVL
#SBATCH --time=10:00:00         # PAIR es lento
#SBATCH --mem=32G

python eval/run_pair.py \
    --target_base "$BASE_MODEL" \
    --target_adapter "$CHECKPOINT" \
    --attacker_model meta-llama/Llama-3.1-8B-Instruct \
    --n_iterations 5 \
    --n_streams 5 \
    --behaviors_subset 100 \
    --output_path "results/$EXP/eval_$(basename $CHECKPOINT)/harmbench_pair.json"
```

### 3.7 `pipeline.sh` — Lanzamiento completo con dependencias

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

# Capa 0: DirectRequest (todos los checkpoints)
EVAL_DR=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    --array=0-$((N_CHECKPOINTS-1)) \
    --export=EXP=$EXP \
    slurm/03_eval_checkpoint.sh)
echo "DirectRequest array job: $EVAL_DR"

# Capa 1: HumanJailbreaks (todos los checkpoints, paralelo a DirectRequest)
EVAL_HJ=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    --array=0-$((N_CHECKPOINTS-1)) \
    --export=EXP=$EXP \
    slurm/05_eval_human_jb.sh)
echo "HumanJailbreaks array job: $EVAL_HJ"

# Capa 2: PAIR (solo snapshots clave — manual, ver §3.6)
# Se lanza manualmente: sbatch --export=EXP=$EXP,CHECKPOINT=... slurm/06_eval_pair.sh

AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:$EVAL_DR:$EVAL_HJ \
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

| Métrica | Descripción | Frecuencia | Dirección |
|---|---|---|---|
| **ASR_direct** | ASR bajo DirectRequest (pregunta directa) | Todos los checkpoints | ↓ mejor |
| **ASR_hj** | ASR bajo HumanJailbreaks (jailbreaks crafteados) | Todos los checkpoints | ↓ mejor |
| **ASR_pair** | ASR bajo ataques PAIR (LLM atacante iterativo) | 4 snapshots por exp | ↓ mejor |
| **Harmless Rate** | % de prompts con rechazo seguro (1 − ASR) | Todos los checkpoints | ↑ mejor |
| **Train Loss** | Loss del SFT en datos de entrenamiento | Continuo | ↓ mejor (por separado) |
| **Eval Perplexity** | Perplejidad en set de validación de Alpaca | Todos los checkpoints | — (proxy de task performance) |

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

Una vez completado el factorial 2×2 principal, las siguientes extensiones tienen coste incremental moderado:

### Ablación del ratio de mezcla (sobre uno o ambos dominios)

Entrenar con ratios del 5%, 10%, 20% además del 15% — al menos sobre el eje donde la mezcla del 15% haya resultado efectiva. Implementación: un job array sobre `configs/exp_{a,c}_ratio_{r}.yaml`.

### Tercer dominio: código

Añadir un eje **Code** (con un dataset tipo Magicoder o CodeAlpaca) para fortalecer el análisis del efecto del dominio. Convierte el 2×2 en un 3×2 (3 dominios × 2 mix).

### Baseline Safe LoRA (Hsu et al. 2024)

Reimplementar como baseline competidor. Permite contrastar la mezcla simple de datos contra una mitigación más sofisticada.

### Transferencia a otra arquitectura

Repetir Exp A o Exp C con Qwen 2.5 3B para validar si los efectos observados son específicos de Llama o generalizables.

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
| **S1** | Fase 0: setup cluster, descarga de pesos, preprocesado de datos (Alpaca, MetaMath, BeaverTails, mezclas), evaluación baseline |
| **S2** | Fase 1: lanzamiento Exp A (Alpaca puro), depuración del pipeline de entrenamiento + evaluación DirectRequest |
| **S3** | Fase 1: lanzamiento Exp B (Alpaca + safety) en paralelo, corrección de errores si los hay |
| **S4** | Fase 2: lanzamiento Exp C (Math puro) — pipeline ya depurado |
| **S5** | Fase 2: lanzamiento Exp D (Math + safety) en paralelo a C |
| **S6** | Fase 2.5: HumanJailbreaks (todos los checkpoints de los 4 experimentos) + PAIR (4 snapshots × 4 exp = 16 evaluaciones) |
| **S7** | Fase 3: análisis factorial 2×2, gráficas, test estadístico, tabla resumen |
| **S8+** | Redacción memoria · publicación repositorio · (opcional) extensiones |

---

## 8. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Cuota de tiempo en cluster agotada durante eval | Media | Reducir frecuencia de checkpoints a 1 por epoch si la cuota es ajustada |
| HarmBench OOM en GPU 40GB | Baja | Usar `--device_map auto` + cuantización 8-bit para la inferencia de evaluación |
| Degradación no observable en 3 epochs | Baja | Qi et al. (2024) y Fraser et al. (2025) la miden en 1 epoch; si no aparece, revisar LR |
| Varianza alta entre runs | Media | Repetir cada experimento con 3 semillas; `aggregate_results.py` calcula media ± std |
| lm-evaluation-harness desactualizado en HarmBench | Baja | Fijar versión en `requirements.txt`, testear en Fase 0 antes de lanzar entrenamiento |
| PAIR se descontrola en tiempo de ejecución (LLM atacante itera demasiado) | Media | Limitar `n_iterations=5` y `n_streams=5`; usar subset de 100 behaviors; timeout duro de 10h en el job |
| HumanJailbreaks tan agresivos que ocultan la curva de degradación (ASR cerca de 1 desde el inicio) | Media | Reportar también ASR por jailbreak individual; si el "peor caso" satura, usar la media |
| PAIR requiere LLM atacante adicional cargado en GPU | Baja | Llama-3.1-8B-Instruct cabe junto a Llama-3.2-3B en H100 96GB sin problemas |
| Confound del formato CoT entre Math y Alpaca (respuestas math son CoT largo) | Media | Reportar también ASR sobre subset de prompts no-CoT; análisis adicional de longitud de respuesta |
| Llama 3.2 3B baseline ya es bueno en math → capability gain bajo en C/D | Media | Si GSM8K no se mueve, usar MATH (más difícil) como métrica complementaria |
| Coste total de los 4 experimentos × 3 ataques > cuota cluster | Alta | Priorizar A→B→C→D en ese orden; si se agota tiempo, reportar parcial (al menos A vs C es publicable) |
