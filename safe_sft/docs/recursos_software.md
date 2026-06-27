# Recursos de software del proyecto

TFM — *Degradación de la seguridad en LLMs bajo SFT: detección temprana y estrategias de mitigación.*
Inventario de todo el software, modelos, datasets y servicios usados. Las versiones
de librerías son las mínimas fijadas en `requirements.txt`.

---

## 1. Modelos

| Modelo | Uso | Fuente |
|---|---|---|
| **Llama-3.2-3B-Instruct** | Modelo principal: SFT, sensor y juez (vía `disable_adapter()`) | `meta-llama/Llama-3.2-3B-Instruct` |
| Llama-3.1-8B-Instruct | Experimentos de escalado | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama-3.1-70B-Instruct | Escalado (QLoRA / 4-bit) | `meta-llama/Llama-3.1-70B-Instruct` |

El mismo modelo base actúa como **juez LLM** congelando el adapter LoRA
(`peft.disable_adapter()`), sin coste de VRAM extra.

---

## 2. Datasets

### Tarea (capacidad)
| Dataset | Uso | Fuente |
|---|---|---|
| **Alpaca** | SFT de instrucciones (dominio general) | `tatsu-lab/alpaca` |
| **MetaMathQA** | SFT de razonamiento matemático | `meta-math/MetaMathQA` |
| **GSM8K** | Evaluación de capacidad matemática | `openai/gsm8k` |

### Safety (alineamiento)
| Dataset | Uso | Fuente |
|---|---|---|
| **BeaverTails** | Prompts dañinos + respuestas; pool de safety, held-out de ASR, prompts self-align | `PKU-Alignment/BeaverTails` |
| **HH-RLHF** | Mezcla estática de safety (Anthropic) | `Anthropic/hh-rlhf` |
| **OR-Bench** | Prompts benignos sobre-rechazados → datos anti-over-refusal | `bench-llm/or-bench` (`or-bench-80k`, `or-bench-hard-1k`, `or-bench-toxic`) |
| **AdvBench** | Fuente de **respaldo (fallback)** de behaviors de HarmBench si falla la descarga del CSV oficial | `walledai/AdvBench` |

### Datasets propios generados
- **self-align** (`beavertails_selfalign_train`): rechazos ideales auto-destilados por el propio modelo (best-of-N con juez por pares).
- **OR-Bench helpful** (`orbench_helpful`): respuestas útiles self-generadas a prompts OR-Bench (anti-over-refusal).
- **canned** (single / pool): negativa fija única o pool de paráfrasis (ablación de diversidad de respuesta).

---

## 3. Benchmarks y evaluación

| Recurso | Mide | Fuente |
|---|---|---|
| **HarmBench** | ASR (attack success rate) — métrica de safety principal | `github.com/centerforaisafety/HarmBench` (behaviors CSV) |
| **XSTest** | Over-refusal (rechazo de prompts benignos) | `paul-rottger/xstest` (`xstest_prompts.csv`) |
| **BeaverTails held-out** | ASR alternativo (descartado como sensor: prompts poco discriminantes) | derivado de BeaverTails |
| **lm-evaluation-harness** | Capacidad general | `lm-eval` (tareas: `ifeval`, `arc_challenge`, `mmlu`, `gsm8k`) |
| **Perplexity** | Coste de capacidad por checkpoint (completion-only sobre val) | propio |

### Jueces de ASR / over-refusal
- **Keyword judge**: detección de rechazo por patrones (rápido, reproducible).
- **LLM judge**: el modelo base evalúa; 3 variantes de prompt (`refuse` / `harm` / `behavior`).
- Principio del proyecto: **el juez del sensor = el juez del test** (una sola fuente de verdad por config).

---

## 4. Librerías y frameworks (Python)

| Librería | Versión mín. | Uso |
|---|---|---|
| **torch** | 2.3.0 | backend de deep learning |
| **transformers** | 4.46.0 | modelos / tokenizers / generación |
| **trl** | 0.12.0 | `SFTTrainer` (entrenamiento SFT) |
| **peft** | 0.13.0 | LoRA / QLoRA |
| **accelerate** | 0.34.0 | orquestación de entrenamiento |
| **datasets** | 2.21.0 | carga y procesamiento de datos |
| **bitsandbytes** | 0.44.0 | cuantización 4-bit (QLoRA 70B) |
| **lm-eval** | 0.4.4 | benchmarks de capacidad |
| **wandb** | 0.18.0 | tracking de experimentos |
| **numpy / pandas** | — / 2.2 | análisis numérico y agregación |
| **matplotlib** | 3.9.0 | figuras |
| **PyYAML** | 6.0.2 | configs de experimentos |
| **huggingface_hub** | 0.25.0 | descarga de modelos/datasets |

Tokenización: la realiza el backend `tokenizers` (fast, tipo *tiktoken*/BPE) que
acompaña a `transformers`; Llama 3.x **no** usa SentencePiece.

Dependencias auxiliares de evaluación (IFEval): **langdetect**, **immutabledict**, **nltk**.

> Nota: `sentencepiece`, `protobuf` y `scipy` aparecen en `requirements.txt` pero
> **no se importan** en el proyecto (sentencepiece/protobuf son vestigios de
> plantillas de Llama-2 — Llama 3.x usa tokenizer BPE *fast*; scipy no se usa).
> Pueden eliminarse de `requirements.txt`.

---

## 5. Configuración de entrenamiento

- **LoRA**: r=16, α=32, dropout=0.05, 7 módulos (`q,k,v,o,gate,up,down_proj`), `task_type=CAUSAL_LM`.
- **SFT**: `SFTTrainer` (TRL), max_length 1024, bf16, lr 2e-4, scheduler coseno, warmup 0.03.
- **Régimen dinámico**: entrenamiento por rondas, sensor configurable (HarmBench / BeaverTails held-out), controladores conmutables: **deadband**, **PID** (log-space + anti-windup), **bandit** (ε-greedy), **fixed**.

---

## 6. Infraestructura

| Recurso | Detalle |
|---|---|
| **Gestor de colas** | Slurm (`sbatch` / `srun`); particiones `cpu`, `gpu` (L40S), `gpuMax` (H100 NVL) |
| **GPUs** | NVIDIA L40S (48 GB) y H100 NVL (96 GB); selección por variable `GPU=l40|h100` |
| **Entorno** | Conda / Miniconda (`activate_conda`) |
| **Pipelines** | `pipeline_factorial.sh` (estáticos), `pipeline_dynamic.sh` (dinámicos); cadena train → eval (array) → aggregate |

---

## 7. Servicios externos

- **Hugging Face Hub** — modelos y datasets (requiere `HF_TOKEN` para mejores límites).
- **Weights & Biases** — seguimiento de métricas de entrenamiento (`WANDB_API_KEY`).

---

## 8. Herramientas de asistencia

- **Claude Code** (Anthropic) — asistencia en desarrollo de scripts, pipelines y análisis.
