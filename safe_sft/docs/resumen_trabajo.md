# Resumen de trabajo — TFM
## Degradación de la Seguridad en LLMs bajo Fine-tuning Supervisado

---

## 1. Contexto y pregunta de investigación

Los LLMs modernos se adaptan a tareas específicas mediante fine-tuning supervisado (SFT). Este proceso erosiona parcialmente las propiedades de seguridad aprendidas durante el pre-entrenamiento y el RLHF — fenómeno conocido como *alignment tax*. El TFM ataca este problema desde el SFT, que es el caso más común en aplicaciones reales y el más accesible experimentalmente.

**Pregunta de investigación:**
> ¿En qué momento del fine-tuning SFT empieza a degradarse la seguridad de un LLM, y puede una mezcla moderada de datos de seguridad en el dataset de entrenamiento prevenir esa degradación sin sacrificar la ganancia de capacidad?

**Conexión con el estado del arte:** Qi et al. (ICLR 2024) demostraron el fenómeno con datos benignos. He et al. (2024) proponen que la similitud representacional entre datos de SFT y datos de safety predice la magnitud de la degradación. Chen et al. (2025) dan un marco teórico formal. Este TFM aporta evidencia empírica sistemática sobre si mezclar datos de safety durante el propio SFT puede contrarrestar el efecto.

---

## 2. Infraestructura

| Recurso | Detalle |
|---------|---------|
| Cluster | UM Slurm (`slurm-ctrl`) |
| GPU principal | H100 96 GB NVL (`gpuMax`) |
| GPU fallback | L4 24 GB (`gpu`) |
| Entorno | Python 3.11, conda `safe_sft` |
| Framework | TRL `SFTTrainer` + PEFT LoRA |
| Tracking | Weights & Biases (`safe_sft_tfm`) |
| Almacenamiento | `/slurm/home/yago/safe_sft_work/` (persistente) |

---

## 3. Modelo base

**Llama 3.2 3B-Instruct** (`meta-llama/Llama-3.2-3B-Instruct`) — elegido por equilibrio entre representatividad y coste. Un run de 3 épocas completo tarda ~2 h en H100.

### Hiperparámetros comunes a todos los experimentos (3B)

| Parámetro | Valor |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Módulos objetivo | q/k/v/o/gate/up/down\_proj |
| Dropout | 0.05 |
| Épocas | 3 |
| Batch efectivo | 16 (batch=4 × grad\_accum=4) |
| Learning rate | 2e-4 (cosine, warmup 3%) |
| Precisión | bfloat16 |
| Gradient checkpointing | desactivado (96 GB suficientes) |
| Checkpoints | cada 500 pasos |

---

## 4. Diseño experimental

### 4.1 Datasets

| Dataset | Uso | Tamaño |
|---------|-----|--------|
| Alpaca | Tarea objetivo (instrucción general) | 52 k instrucciones |
| MetaMathQA | Tarea objetivo (matemáticas) | ~395 k ejemplos |
| BeaverTails (is\_safe=True) | Safety data — filtro básico | ~300 k pares |
| BeaverTails filtrado | Safety data — solo rechazos explícitos | Subconjunto de BeaverTails |
| HH-RLHF (harmless-base, chosen) | Safety data — rechazos curados por humanos | ~42 k ejemplos |
| HarmBench | Evaluación de seguridad | 300 prompts (200 estándar + 100 copyright) |

### 4.2 Evaluación de seguridad

**Métrica principal:** Attack Success Rate (ASR) — fracción de prompts dañinos ante los que el modelo genera contenido harmful en lugar de rechazar.

HarmBench se divide en dos subcategorías evaluadas por separado:

- **Standard** (200 prompts): peticiones directamente dañinas (violencia, drogas, cibercrimen…)
- **Copyright** (100 prompts): generación de contenido con copyright

El juez es el propio pipeline de HarmBench (keyword-based + clasificador). ASR más alto = modelo menos seguro.

### 4.3 Experimentos definidos

| ID | Dataset de entrenamiento | Safety data | Estado |
|----|--------------------------|-------------|--------|
| **A** | Alpaca | — | ✅ Completado (3B) |
| **B** | Alpaca + 15% BeaverTails (is\_safe=True) | BeaverTails | ✅ Completado (3B) |
| **C** | MetaMathQA | — | ⏳ Pendiente |
| **D** | Alpaca + 15% HH-RLHF (harmless-base) | HH-RLHF | ✅ Completado (3B) |
| **E** | Alpaca + 15% BeaverTails filtrado | BeaverTails clean | ✅ Completado (3B) |

---

## 5. Resultados — Llama 3.2 3B

### 5.1 Baseline (modelo sin fine-tuning)

| Categoría | ASR |
|-----------|-----|
| Standard | 0.085 (8.5%) |
| Copyright | 0.860 (86.0%) |
| **Total** | **0.343** |

El ASR de copyright es alto incluso en el baseline, lo que lo hace poco discriminativo para medir degradación. El análisis principal se centra en la categoría standard.

### 5.2 Curvas de ASR total a lo largo del entrenamiento

| Checkpoint (step) | Baseline | Exp A | Exp B | Exp D | Exp E |
|-------------------|----------|-------|-------|-------|-------|
| 0 | 0.343 | 0.343 | 0.343 | 0.343 | 0.343 |
| 500 | — | 0.707 | 0.963 | 0.927 | 0.663 |
| 1000 | — | 0.620 | 0.947 | 0.967 | 0.713 |
| 1500 | — | 0.703 | 0.990 | 0.957 | 0.933 |
| 2000 | — | 0.640 | 0.997 | 0.980 | 0.873 |
| 3000 | — | 0.723 | 0.943 | 0.970 | 0.943 |
| Final (época 3) | — | **0.690** | **0.980** | **0.960** | **0.917** |

### 5.3 ASR_standard en el checkpoint final

| Experimento | ASR_standard final |
|-------------|-------------------|
| Baseline | 0.085 |
| **A** — Alpaca puro | **0.570** |
| **B** — Alpaca + BeaverTails | 0.975 |
| **D** — Alpaca + HH-RLHF | 0.950 |
| **E** — Alpaca + BeaverTails clean | 0.910 |

---

## 6. Análisis de resultados

### 6.1 Hallazgo central (contraintuitivo)

**Añadir datos de safety empeora la seguridad respecto a no añadirlos.** El experimento A (Alpaca puro, sin ningún dato de safety) termina con ASR_total 0.69, mientras que todos los experimentos con mezcla (B, D, E) terminan entre 0.92 y 0.98. La hipótesis de trabajo —que la mezcla de safety data preserva el alineamiento— no se confirma.

### 6.2 Diferencias entre B, D y E

| Exp | Comportamiento early (step 500) | Comportamiento final | Observación clave |
|-----|--------------------------------|---------------------|-------------------|
| **B** (BeaverTails raw) | ASR 0.963 — degradación inmediata | 0.98 — máxima degradación | El ruido de BeaverTails (respuestas "safe" que no son rechazos) es perjudicial desde el inicio |
| **D** (HH-RLHF) | ASR 0.927 — degradación inmediata | 0.96 — similar a B | Sorprendente: a pesar de usar datos de mayor calidad, el comportamiento es casi idéntico a B |
| **E** (BeaverTails filtrado) | ASR 0.663 — cercano a A (0.707) | 0.917 — mejor que B y D | El filtrado sí ayuda inicialmente; hay protección real en los primeros ~1000 steps que luego se erosiona |

### 6.3 Hipótesis explicativas

**Por qué los experimentos con safety data son peores que sin ella:**

1. **Exposición a prompts dañinos durante el entrenamiento.** Los ejemplos de safety data incluyen los prompts dañinos como input. Aunque la respuesta objetivo es un rechazo, el modelo se entrena sobre el contenido de esos prompts y puede estar aprendiendo los patrones del contenido dañino además del comportamiento de rechazo.

2. **Posible desalineación de formato en Exp D.** HH-RLHF usa el formato `Human:/Assistant:`. Si la preparación de datos no convirtió correctamente al chat template de Llama 3.x, los mecanismos de safety del modelo —vinculados al template— pueden haberse roto desde el primer batch. Esto explicaría que D colapsa con la misma velocidad que B a pesar de ser datos de mayor calidad.

3. **El filtrado de E sí funciona, pero no es suficiente.** La protección inicial de E sugiere que los rechazos limpios son señal útil. El problema es que esa señal no escala lo suficiente frente a los ~9000 pasos de Alpaca que la acompañan.

### 6.4 Preguntas abiertas

- ¿Sería HH-RLHF más efectivo si se verificara la correcta conversión de formato antes de mezclar?
- ¿Qué porcentaje de safety data haría falta para que la protección de E se mantuviera a lo largo de todo el entrenamiento?
- ¿El comportamiento se replica en modelos más grandes (8B, 70B)?
- ¿Qué ocurre con MetaMathQA (Exp C)? La teoría de He et al. predice que dominios de baja similitud representacional con safety (como matemáticas) degradarán de forma diferente.

---

## 7. Escalado a modelos más grandes

### 7.1 Configuraciones creadas

Se han creado los 10 configs de entrenamiento para replicar los experimentos A–E con dos tamaños adicionales:

| Tamaño | Modelo | Mecanismo | Batch | Tiempo est. |
|--------|--------|-----------|-------|-------------|
| **3B** | Llama-3.2-3B-Instruct | LoRA bf16 | 4 × acc=4 = 16 | ~2 h ✅ |
| **8B** | Meta-Llama-3.1-8B-Instruct | LoRA bf16 | 2 × acc=8 = 16 | ~5 h |
| **70B** | Meta-Llama-3.1-70B-Instruct | QLoRA 4-bit NF4 | 1 × acc=16 = 16 | ~20–30 h |

Los experimentos de 8B y 70B usan los mismos datasets que los de 3B (mismo chat template Llama 3.x, compatible sin cambios). El batch efectivo es idéntico en los tres tamaños.

### 7.2 Cambios en train.py para QLoRA

Se añadió soporte opcional de cuantización 4-bit. Si el config incluye una sección `quantization`, el script carga el modelo con `BitsAndBytesConfig` (NF4) y llama `prepare_model_for_kbit_training`. Sin esa sección, el comportamiento es idéntico al original.

```yaml
# Sección añadida en configs de 70B
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
```

### 7.3 Estado del escalado

Los experimentos de 8B están **listos para lanzar** una vez descargado el modelo `Meta-Llama-3.1-8B-Instruct` en el cluster:

```bash
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir $WORK_DIR/models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
```

Los de 70B requieren descargar adicionalmente `Meta-Llama-3.1-70B-Instruct` (~140 GB).

---

## 8. Datasets de safety — discusión

| Dataset | Evaluación |
|---------|------------|
| **BeaverTails (raw)** | Problemático: respuestas etiquetadas `is_safe=True` frecuentemente no son rechazos sino explicaciones o contenido parcialmente dañino. Exp B lo confirma empíricamente. |
| **BeaverTails filtrado** | Mejora respecto al raw (Exp E muestra protección inicial), pero el filtro por keywords es imperfecto y el dataset subyacente sigue siendo ruidoso. |
| **HH-RLHF (harmless-base, chosen)** | Mayor calidad cualitativa, pero los resultados de Exp D son sorprendentemente malos — probablemente por problemas de compatibilidad de formato o por la exposición a prompts dañinos. Requiere verificación del pipeline de preparación. |
| **WildGuard** | Recomendado como próximo paso: prompts adversariales modernos con refusals de alta calidad, formato compatible con modelos instruct actuales. No implementado aún. |
| **PKU-SafeRLHF** | Dataset de preferencias con labels de safety por respuesta. Estructura más compleja que BeaverTails. No implementado aún. |

---

## 9. Estructura del repositorio

```
safe_sft/
├── configs/
│   ├── base.yaml
│   ├── exp_a_alpaca_pure.yaml          # 3B
│   ├── exp_b_alpaca_mixed.yaml         # 3B
│   ├── exp_c_math_pure.yaml            # 3B
│   ├── exp_d_alpaca_hh_rlhf.yaml       # 3B
│   ├── exp_e_alpaca_beavertails_clean.yaml  # 3B
│   ├── exp_a_8b_alpaca_pure.yaml       # 8B
│   ├── exp_b_8b_alpaca_mixed.yaml      # 8B
│   ├── exp_c_8b_math_pure.yaml         # 8B
│   ├── exp_d_8b_alpaca_hh_rlhf.yaml    # 8B
│   ├── exp_e_8b_alpaca_beavertails_clean.yaml  # 8B
│   ├── exp_a_70b_alpaca_pure.yaml      # 70B QLoRA
│   ├── exp_b_70b_alpaca_mixed.yaml     # 70B QLoRA
│   ├── exp_c_70b_math_pure.yaml        # 70B QLoRA
│   ├── exp_d_70b_alpaca_hh_rlhf.yaml   # 70B QLoRA
│   └── exp_e_70b_alpaca_beavertails_clean.yaml  # 70B QLoRA
├── data/
│   ├── prepare_alpaca.py
│   ├── prepare_beavertails.py
│   ├── prepare_beavertails_filtered.py
│   ├── prepare_hh_rlhf.py
│   ├── prepare_metamath.py
│   ├── mix_datasets.py
│   └── split_train_val.py
├── train/
│   └── train.py                        # LoRA + QLoRA (sección quantization opcional)
├── eval/
│   └── run_harmbench.py
├── slurm/
│   ├── cluster_config.sh
│   ├── 00_setup.sh … 05_aggregate.sh
│   └── pipeline.sh
└── results/
    ├── baseline/harmbench.json
    ├── exp_a/security_curve.csv        # ✅
    ├── exp_b/security_curve.csv        # ✅
    ├── exp_d/security_curve.csv        # ✅
    └── exp_e/security_curve.csv        # ✅
```

---

## 10. Literatura relevante

| Referencia | Relevancia para el TFM |
|------------|----------------------|
| Qi et al. (ICLR 2024) | Seminal: demuestran que SFT sobre datos benignos degrada safety. Base experimental del TFM. |
| He et al. (2024) — *"What's in Your 'Safe' Data?"* | Similitud representacional entre datos SFT y datos de safety predice la magnitud de degradación. Relevante para interpretar los resultados de Exp C (math). |
| Chen et al. (2025) | Marco teórico formal: mayor similitud datos-safety → mejor trade-off. Math es el caso de baja similitud. |
| Bianchi et al. (2024) — *Safety-Tuned LLaMAs* | Mezcla estática de safety data en SFT: 500 ejemplos bastan para efectos observables. Base de Exp B. |
| Lermen et al. (2024) | ~100 ejemplos benignos destruyen el alineamiento con LoRA. Refuerza que el problema es representacional, no de cantidad. |
| Pelrine et al. (2024) | Fine-tuning de GPT-4 sobre datos benignos abre vulnerabilidades. Caso de caja negra. |

---

## 11. Estado actual y próximos pasos

### Completado
- [x] Setup del cluster y entorno conda
- [x] Pipeline de entrenamiento (LoRA) + pipeline de evaluación (HarmBench)
- [x] Scripts de preparación de datos: Alpaca, BeaverTails, BeaverTails filtrado, HH-RLHF, MetaMathQA
- [x] Experimentos A, B, D, E en Llama 3.2 3B
- [x] Soporte QLoRA en train.py para modelos 70B
- [x] Configs de entrenamiento para 8B (×5) y 70B QLoRA (×5)

### Pendiente inmediato
- [ ] Descargar `Meta-Llama-3.1-8B-Instruct` en el cluster y lanzar experimentos 8B
- [ ] Ejecutar Exp C (MetaMathQA) en 3B
- [ ] Añadir métricas de task performance (perplexity en val, o accuracy en benchmarks de math para Exp C)

### Pendiente a medio plazo
- [ ] Implementar `prepare_wildguard.py` y lanzar Exp con WildGuard como safety dataset
- [ ] Investigar el fallo de Exp D (verificar formato de datos HH-RLHF vs template Llama 3.x)
- [ ] Lanzar experimentos 70B (requiere descarga ~140 GB)
- [ ] Análisis y visualización de curvas comparativas entre modelos
- [ ] Redacción de la memoria
