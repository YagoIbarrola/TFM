# TFM — Experimentos: degradación de seguridad en SFT y mitigación adaptativa

Documento de síntesis para la memoria. Recoge pregunta de investigación,
infraestructura, datasets, métricas, diseño experimental, resultados con cifras,
el régimen de entrenamiento dinámico y sus controladores, los controles
metodológicos y las lecciones aprendidas.

> **Nota de estado:** los experimentos *estáticos* y las *ablaciones canned/self-align*
> están consolidados. Los *dinámicos* están en una segunda iteración tras corregir
> el sensor/target del controlador (ver §8); las cifras dinámicas marcadas como
> "(v1)" usan el sensor antiguo y quedan superadas por la v2.

---

## 1. Pregunta de investigación

> ¿En qué momento del fine-tuning supervisado (SFT) empieza a degradarse la
> seguridad de un LLM (*alignment tax*), y puede una mezcla de datos de seguridad
> —fija o **adaptativa**— prevenir esa degradación sin sacrificar la capacidad en
> la tarea ni provocar *over-refusal*?

Contribuciones diferenciales del trabajo:
1. Curvas de seguridad **por paso** de SFT en un modelo 3B (HarmBench + held-out propio).
2. Comparación sistemática de **datasets de safety** (reales vs sintéticos vs
   auto-generados) en el plano **seguridad ↔ over-refusal**.
3. Un **régimen de entrenamiento dinámico** que ajusta el % de datos de safety en
   función del ASR medido, con tres controladores (banda muerta, PID, bandit).
4. Generación de un dataset de alineamiento **self-distill** (respuestas ideales
   del propio modelo) y evidencia de que **domina** a las alternativas.

---

## 2. Infraestructura

| Recurso | Detalle |
|---|---|
| Modelo base | **Llama-3.2-3B-Instruct** |
| Adaptación | LoRA (r=16, α=32, dropout 0.05, módulos q/k/v/o/gate/up/down) |
| Framework | TRL `SFTTrainer` (estático) + bucle manual `train_dynamic.py` (dinámico) |
| Hiperparámetros | 3 épocas, batch efectivo 16 (4×grad_accum 4), lr 2e-4 cosine, warmup 3%, bf16 |
| Cluster | UM Slurm: **6× L40S (48 GB)** partición `gpu` + **1× H100 (96 GB)** `gpuMax` |
| Tiempo | ~2-2.5 h por run de 3B (≈9 000-10 900 pasos según dataset) |

Un run de 3B tarda ~2 h; los dinámicos añaden una eval de sensor por ronda.

---

## 3. Datasets

### Tarea (capacidad objetivo)
| Dataset | Dominio | Tamaño usado |
|---|---|---|
| Alpaca (`tatsu-lab/alpaca`) | instrucción general | ~52 k (split 95/5) |
| MetaMathQA (`meta-math/MetaMathQA`) | matemáticas | submuestreado a 52 k |

### Safety (mezcla)
| Pool | Construcción | Nota |
|---|---|---|
| BeaverTails-safe (raw) | `is_safe=True` | ruidoso: muchas respuestas no son rechazos |
| BeaverTails-clean | is_safe + patrón de rechazo | **pool pequeño (~830 prompts)** → ratio efectivo bajo |
| HH-RLHF harmless (chosen) | rechazos curados por humanos | formato 2022, resultados mediocres |
| **Canned single** | prompts BeaverTails + **una** negativa fija | ablación "diversidad 0" |
| **Canned pool** | prompts BeaverTails + paráfrasis de ~15 negativas | ablación diversidad baja |
| **Self-align** | prompts dañinos BeaverTails + **respuesta ideal auto-generada** por el propio 3B (best-of-N, juez por pares, system prompt de seguridad; respuesta formateada con system prompt estándar) | ~10 k pares |

> **Caveat de ratio efectivo:** `mix_datasets.py` topa el nº de ejemplos de safety
> al tamaño del pool. Con BeaverTails-clean (~830-1341 filas), un "15 %" nominal
> sobre 52 k de Alpaca da un **ratio efectivo ≈ 2.6 %**. Las cifras deben reportarse
> con el ratio efectivo, no el nominal. Self-align (~10 k) y canned (mismos prompts)
> sí alcanzan ratios reales mayores.

---

## 4. Métricas y evaluación

Toda la evaluación es **por checkpoint** (curva a lo largo del entrenamiento) salvo
las marcadas "solo final".

| Métrica | Qué mide | Verificador | Dirección |
|---|---|---|---|
| **ASR HarmBench** (`asr_standard`) | seguridad (200 prompts standard) | juez HarmBench | ↓ |
| **ASR held-out BeaverTails** (`bt_asr`) | seguridad in-distribution (256 prompts dañinos disjuntos) | keyword **o** juez LLM | ↓ |
| **Over-refusal XSTest** (`xstest_refusal_safe`) | rechazo de peticiones **benignas** | keyword | ↓ |
| **Perplexity** (val) | ajuste a la tarea (completion-only) | — | ↓ (proxy débil) |
| **GSM8K accuracy** | capacidad math real (solo dominio math) | extracción nº + exact-match | ↑ |
| **IFEval / ARC / MMLU** (lm-eval) | seguimiento de instrucciones / capacidad general | programático / MC | ↑ (solo final) |

Baselines del modelo sin entrenar: **HarmBench ASR_std = 0.085**; held-out BeaverTails
con **juez keyword ≈ 0.56** (inflado), con **juez LLM ≈ pendiente** (mucho menor).
La discrepancia 0.085 vs 0.56 evidencia la descalibración del juez keyword (§8).

---

## 5. Diseño experimental

**Factorial principal** {Alpaca, MetaMath} × {0 %, 5 %, 15 %} de safety, con
distintos pools (BeaverTails-clean, canned, self-align), evaluado por checkpoint.

**Ablación "diversidad de respuesta"** (mismos prompts, solo cambia la respuesta):
canned-single → canned-pool → respuesta real → self-align.

**Régimen dinámico**: ratio de safety adaptativo, 3 controladores (§7), sobre
Alpaca y MetaMath, con safety canned y self-align.

**Controles metodológicos**: estático vs dinámico-fijo a igualdad de condiciones;
ruido entre semillas; con-reemplazo vs sin-reemplazo (§9).

---

## 6. Resultados clave (cifras)

### 6.1 Añadir safety "real" (pool pequeño) puede EMPEORAR HarmBench
Primeros experimentos (Alpaca, pool clean, ratio efectivo ~2.6 %), ASR_std final:

| Exp | Safety | ASR_std |
|---|---|---|
| baseline | — | 0.085 |
| Alpaca puro (A) | — | **0.57** |
| Alpaca + BeaverTails raw (B) | 15 %nom | 0.975 |
| Alpaca + HH-RLHF (D) | 15 %nom | 0.95 |
| Alpaca + BeaverTails clean (E) | 15 %nom | 0.91 |

Resultado contraintuitivo: **con datos de safety ajenos/ruidosos el ASR sube** por
encima de Alpaca puro (hipótesis: exposición a prompts dañinos como input +
posible desajuste de formato + ratio efectivo bajo).

### 6.2 Ablación de diversidad de respuesta (Alpaca 15 %)
| Variante | ASR_std ↓ | over-refusal (XSTest) ↓ |
|---|---|---|
| respuesta real (exp_e) | 0.91 | — |
| **canned-single** | **0.16** | **0.65** |
| **canned-pool** | **0.185** | **0.65** |

La respuesta **reducida/rígida preserva la seguridad** (ASR 0.16 vs 0.91) **pero
dispara el over-refusal** (~0.65): el modelo aprende a rechazar como reflejo bruto.
Canned-5 % ≈ canned-15 % (el over-refusal **satura** ya al 5 %).

### 6.3 Self-align domina la frontera de Pareto (Alpaca)
| Punto | over-refusal ↓ | ASR_std ↓ |
|---|---|---|
| **Self-align 5 %** | 0.29 | **0.085** (=baseline) |
| **Self-align 15 %** | 0.43 | **0.03** |
| Canned (5/15 %) | 0.65 | 0.16-0.19 |
| Dinámico canned (v1) | 0.30 | 0.21-0.25 |

- **Self-align estático Pareto-domina a canned** en ambos ejes.
- **Self-align 5 %** es el *sweet spot*: ASR igual al modelo original con
  over-refusal de solo 0.29 y apenas 5 % de datos.
- Self-align 15 % logra **ASR 0.03 < baseline 0.085** (el modelo queda *más* robusto
  que el original).
- Interpretación: las respuestas **in-distribution** (generadas por el propio
  modelo) refuerzan el rechazo sin la divergencia que causa el over-refusal.

### 6.4 MetaMath: el safety NO cuesta capacidad
GSM8K accuracy se mantiene **~0.72-0.77 en TODAS las condiciones** (0/5/15 % y
dinámicos). Mezclar safety **no degrada** la accuracy matemática. El ASR-BT del
math-0 % es alto (~0.56) y baja a ~0.2 con 5-15 % de self-align.

### 6.5 Perplexity (overfitting leve, común a todos)
Alpaca ~3.85-4.3 (mínimo ~3.82 sobre el paso 1500, sube a ~4.16 al final = overfit
de la 3.ª época); MetaMath ~1.22. La subida es **igual en estáticos y dinámicos** a
igualdad de ratio → no la causa el régimen, sino el propio SFT.

---

## 7. Régimen de entrenamiento dinámico

Entrenamiento **por rondas** (`round_steps=500`, ~18 rondas). Cada ronda:
construye una mezcla fresca task/safety al ratio actual → entrena 500 pasos →
mide el ASR en el held-out de BeaverTails (**sensor**) → el **controlador** ajusta
el ratio. Optimizer y scheduler (cosine) únicos sobre todo el run (LR continuo).
Muestreo **sin reemplazo** (cobertura tipo época, §9).

### Controladores (`train/controllers.py`)
| Tipo | Regla | Notas |
|---|---|---|
| **Deadband** | banda muerta; fuera de banda **multiplica** (×up / ×down) | sin memoria; `min_ratio`=0.01 (suelo > 0) |
| **PID** | Δ = Kp·e + Ki·∫e + Kd·Δe, `e=ASR−target` | **actuador log** (ratio *= exp Δ, nunca ≤0) + **anti-windup** |
| **Bandit** | ε-greedy no estacionario sobre brazos = ratios discretos | recompensa asimétrica + coste de ratio |
| **Fixed** | ratio constante | control para comparar con estático |

El **target** puede fijarse a mano o **auto-calibrarse** al ASR del modelo base
(`target_from_baseline`, con multiplicador `target_baseline_mult` ×1.0/1.5/2.0).

---

## 8. El problema del sensor/target (y su corrección)

**Síntoma (v1):** el PID llevaba el safety a 0 y el ASR (HarmBench) subía a ~0.27,
con over-refusal muy bajo. El deadband no lo sufría tanto **por accidente**: su
suelo `min_ratio=0.01` mantenía 1 % de safety, suficiente (self-align es potente)
para sostener el ASR; el PID (suelo 0) llegaba a 0 %.

**Diagnóstico (dos causas):**
1. **Target inflado:** `target_from_baseline` medía el base con el **juez keyword**
   → 0.56 (mientras HarmBench da 0.085). El PID apuntaba a "ASR 0.56" → toleraba
   ASR alto. El juez keyword **infla** el ASR (cuenta como no-rechazo respuestas
   matizadas).
2. **Windup del integral:** el sobreimpulso inicial (10 % safety hundía el sensor)
   acumulaba un integral negativo que dejaba el ratio clavado en 0 aun cuando el
   error se volvía positivo.

**Corrección (v2):**
- **Target a 0.1** (o relativo al base **con juez LLM**), `target_from_baseline:false`
  o multiplicador.
- **PID en espacio logarítmico** (actualización multiplicativa, encaja con la
  no-linealidad ASR↔ratio) + **anti-windup** (integración condicional en saturación).
- **Juez LLM**: el sensor `bt_asr` usa el **3B base congelado** (vía
  `disable_adapter()`, sin cargar un 2.º modelo) con un prompt de clasificación
  REFUSE/COMPLY. Mucho más fiable que keywords.

**Resultado (v2):** los tres controladores ahora **suben el safety** cuando el
ASR supera el target (juez LLM da `bt_asr ≈ 0.15-0.25`). De hecho **saturan en
`max_ratio=0.30`**: ni con 30 % de self-align el `bt_asr` baja del ~0.18 → el
target 0.1 no es alcanzable en este sensor (hallazgo: el controlador "quiere" más
safety del permitido). Variantes `pidB10/B15/B20` (target = base ×1.0/1.5/2.0)
miden cuánto safety hace falta para igualar / relajar 1.5×-2× el ASR del original.

---

## 9. Controles metodológicos

- **Estático vs dinámico-fijo (15 %, igualdad de condiciones):** a mismo ratio,
  misma cobertura (sin reemplazo) y mismos pasos, las curvas (ASR, over-refusal,
  perplexity) **coinciden** salvo un offset de perplexity ~1.4 %. → el bucle manual
  es **fiel** a SFTTrainer; las diferencias entre dinámicos y estáticos se deben al
  **ratio efectivo** que elige el controlador, no al método.
- **Ruido entre semillas:** estático 15 % seed42 (ppl 4.162) vs seed43 (4.149) →
  **~0.3 %**. El offset estático↔dinámico (~1.4 %) es mayor que el ruido de semilla
  → hay una diferencia de implementación **pequeña** (acumulación bf16 / normalización
  de loss en grad-accum), no estructural.
- **Muestreo con vs sin reemplazo (dinámico):** el muestreo **con reemplazo** inflaba
  la perplexity (cobertura desigual): 4.2-4.3 → **3.85-3.87** al pasar a **sin
  reemplazo**. Se adoptó sin reemplazo para comparabilidad con el estático.

---

## 10. Lecciones / pitfalls (útiles para "trabajo futuro" y reproducibilidad)

1. **Juez de seguridad:** un juez por keywords **infla y descalibra** el ASR
   (0.56 vs 0.085 de HarmBench) y puede romper un controlador que lo use de target.
   Un juez LLM (base congelado) es barato (`disable_adapter`) y mucho más fiable.
2. **Ratio nominal ≠ efectivo:** revisar el "Effective ratio" del mezclador; un pool
   pequeño capa el ratio real.
3. **Cobertura del array de evaluación:** `ARRAY_SIZE` < nº de checkpoints **recorta
   la curva en silencio** (los dinámicos guardan ~19 checkpoints; con `--array=0-11`
   solo se evalúan los 12 primeros). Usar `ARRAY_SIZE ≥ 20` para dinámicos.
4. **Cuota de disco/inodes:** los checkpoints (sobre todo estáticos con `optimizer.pt`)
   y un entorno conda grande (~107 k inodes) saturan la cuota. Mitigaciones aplicadas:
   `save_steps` 1000 + `save_total_limit` 12, **auto-limpieza** de adapters tras
   agregar (conservando `evals/`), y separar los JSON de eval (con prompts+respuestas)
   del checkpoint para poder **re-juzgar luego sin re-entrenar**.
5. **bf16 + manual loop:** no es determinista; esperar offsets ~1 % vs Trainer.
6. **Resume + torch<2.6:** `transformers` bloquea `torch.load` del optimizer al
   reanudar (CVE) → entrenar **en fresco** (los dinámicos no resumen de todas formas).

---

## 11. Estado y pendientes

- **Consolidado:** factorial estático (canned, self-align 5/15 %, math), ablaciones
  de diversidad, controles (semilla, estático↔dinámico-fijo), métricas IFEval/ARC/MMLU
  por el final.
- **En curso (v2):** re-ejecución de los 6 dinámicos self-align con sensor LLM +
  target 0.1 + PID log/anti-windup; 3 variantes PID `B10/B15/B20` (target = base
  ×1.0/1.5/2.0); baseline `bt_asr` con ambos jueces (job 09).
- **Trabajo futuro:** datasets SOTA de safety con benignos de contraste
  (WildJailbreak, CoCoNot) para atacar el over-refusal; variante híbrida self-align
  + contraste benigno; track de código (HumanEval) como tercer dominio objetivo.

---

## 12. Estructura del repositorio (orientación)

```
safe_sft/
├── configs/                 # un YAML por experimento (estáticos + dinámicos)
│   └── scaling/             # configs 8B/70B (no usados en la tanda 3B)
├── data/                    # prepare_*.py, mix_datasets.py, gen_selfalign.py
├── train/                   # train.py (SFTTrainer), train_dynamic.py, controllers.py
├── eval/                    # run_harmbench, run_xstest, run_task_eval,
│                            # run_beavertails_asr (juez keyword/LLM), run_lmeval,
│                            # aggregate_results, aggregate_task, plot_curves
├── slurm/                   # 01x prep, 03 train, 04 eval, 05 agg, 06 train_dynamic,
│                            # 07 gen_selfalign, 08 eval_final, 09 baseline_bt_asr,
│                            # pipeline_factorial / pipeline_dynamic
└── results/                 # security_curve.csv, task_curve.csv, dynamic_log.csv,
                             # evals/<ckpt>/*.json (prompts+respuestas), figures/
```

**Figuras clave** (`results/figures/`): `12_tradeoff_final` (ASR vs over-refusal,
la "money plot"), `16/17` rejillas por dominio, `15` controladores dinámicos,
`18` control estático↔dinámico-fijo + ruido de semilla.
