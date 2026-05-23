Trabajo de Fin de Máster — Propuesta extendida (v2)

**Trayectoria de Degradación de Seguridad en LLMs bajo Fine-tuning Supervisado: Efecto del Dominio de la Tarea Objetivo y Eficacia de la Mezcla de Datos de Seguridad**

Máster en Inteligencia Artificial

_Trabajo alineado con la investigación de Yuvaraj Govindarajulu (Bosch AIShield) y con el marco teórico de Chen et al. (2025)_

---

# 1. Contexto y motivación

Los grandes modelos de lenguaje (LLMs) se adaptan habitualmente a tareas específicas mediante fine-tuning supervisado (SFT). Sin embargo, este proceso erosiona las propiedades de seguridad que el modelo tenía tras su entrenamiento original — fenómeno conocido como "alignment tax". Qi et al. (2024) demostraron que incluso fine-tuning sobre datasets benignos como Alpaca degrada la seguridad de modelos previamente alineados como Llama-2 y GPT-3.5 Turbo.

Hasta el momento, la literatura ha investigado *si* y *cuándo* ocurre la degradación, pero ha tratado el SFT como una operación monolítica. **La pregunta de cómo depende la degradación del dominio de la tarea objetivo permanece prácticamente sin explorar de forma empírica**, a pesar de que:

- Chen et al. (2025) probaron *teóricamente* que la magnitud del trade-off seguridad-capacidad depende del solapamiento entre los datos de seguridad y los de capacidad. Su predicción: dominios estrechos y desalineados (matemáticas, código) degradarán más que dominios generalistas (instrucciones).
- Bianchi et al. (2024) observaron, en estudios cross-domain, que dominios estrechos (médico, legal) degradan más seguridad que datos generalistas.
- Ningún paper actual mide *epoch-by-epoch* la degradación entre dominios benignos distintos sobre el mismo modelo open-source con metodología comparable.

**Pregunta de investigación**

_¿En qué medida la trayectoria de degradación de la seguridad durante el SFT depende del dominio de la tarea objetivo, y la eficacia de una mezcla moderada de datos de seguridad como mitigación es uniforme entre dominios o dependiente de ellos?_

---

# 2. Objetivos

## 2.1 Objetivo principal

Cuantificar, mediante un diseño experimental factorial 2×2 (dominio × mezcla de seguridad), la trayectoria de degradación de seguridad durante el fine-tuning supervisado y aislar:

1. El efecto del **dominio** de la tarea objetivo (general vs. narrow) sobre la magnitud y el momento del colapso.
2. El efecto de la **mezcla de datos de seguridad** como mitigación.
3. La interacción entre ambos: ¿la mezcla compensa el efecto del dominio o ambos factores son independientes?

## 2.2 Objetivos específicos

- Establecer la línea base de seguridad del modelo (Llama 3.2 3B Instruct) bajo tres niveles de ataque: DirectRequest, HumanJailbreaks y PAIR.
- Trazar cuatro curvas de degradación epoch-by-epoch correspondientes a las cuatro condiciones del diseño factorial.
- Validar o refutar empíricamente las predicciones teóricas de Chen et al. (2025) sobre el efecto del solapamiento de dominios.
- Comparar la efectividad de la mezcla del 15% de safety data en dominios general y narrow.
- Publicar código, checkpoints, datos de evaluación y curvas en abierto, con licencia Apache 2.0.

---

# 3. Metodología

## 3.1 Diseño experimental factorial 2×2

|  | **Sin safety mix** | **Con 15% safety mix** |
|---|---|---|
| **Tarea general (Alpaca)** | Exp A | Exp B |
| **Tarea narrow (MetaMathQA)** | Exp C | Exp D |

Cada celda produce ~22 checkpoints (uno cada 500 pasos durante 3 epochs) evaluados sobre 3 ataques. Total: ~88 checkpoints × 3 niveles de ataque + 16 snapshots PAIR.

## 3.2 Modelo y datasets

| Recurso | Uso | Tamaño efectivo |
|---|---|---|
| Llama 3.2 3B Instruct | Modelo base de los 4 experimentos | 3B params, alineado |
| Alpaca | Dataset SFT general (Exp A, B) | 52k instrucciones |
| MetaMathQA | Dataset SFT narrow (Exp C, D) submuestreado | 52k problemas con CoT |
| BeaverTails-30k (is_safe=True) | Pool de safety data para mezclas | ~9k ejemplos en cada mezcla 15% |
| HarmBench (DirectRequest + HumanJailbreaks) | Evaluación de seguridad | ~300 + jailbreaks crafteados |
| PAIR + Llama-3.1-8B atacante | Evaluación de robustez en snapshots clave | 100 behaviors × 4 snapshots × 4 exp |
| GSM8K, Alpaca-eval | Métricas de capacidad por dominio | — |

**Decisiones clave:**
- Igualar tamaño de datasets (52k) garantiza que el efecto del dominio no se confunde con el volumen de entrenamiento.
- Mantener todos los hiperparámetros idénticos entre experimentos (LR, batch, schedule, semilla, LoRA r=16).
- La única variable independiente entre celdas adyacentes del 2×2 es bien el dominio (eje vertical) o bien la mezcla (eje horizontal).

## 3.3 Pipeline de evaluación en tres capas

| Capa | Método | Frecuencia | Coste relativo |
|---|---|---|---|
| 1. **Alineación nominal** | HarmBench DirectRequest (pregunta directa) | Todos los checkpoints (~22 por exp) | Bajo |
| 2. **Robustez moderada** | HarmBench HumanJailbreaks (jailbreaks crafteados) | Todos los checkpoints (~22 por exp) | Bajo (mismo orden que la capa 1) |
| 3. **Robustez adversarial** | PAIR (LLM atacante iterativo) | 4 snapshots clave por exp (baseline + 3 epochs) | Alto (~8 h GPU por snapshot) |

Las tres capas son complementarias: la capa 1 mide si el modelo rechaza preguntas dañinas, la 2 mide si resiste a usuarios maliciosos comunes, y la 3 mide si resiste a atacantes optimizadores.

## 3.4 Plan de trabajo

| Fase | Tareas principales | Entregable |
|---|---|---|
| **0. Setup** | Entorno cluster, modelo, 4 datasets, baseline en 3 ataques | Tabla baseline |
| **1. Eje Alpaca** | Exp A y B con evaluación DirectRequest en cada checkpoint | Curvas A y B (capa 1) |
| **2. Eje Math** | Exp C y D con evaluación DirectRequest en cada checkpoint | Curvas C y D (capa 1) |
| **2.5. Robustez** | HumanJailbreaks sobre los 4 ejes + PAIR en snapshots clave | Curvas capas 2 y 3 |
| **3. Análisis 2×2** | Test estadístico del efecto dominio y mezcla, gráficas factoriales | Tabla 2×2 + figuras |
| **4. Memoria** | Redacción TFM, repositorio público, posible preprint | TFM + código abierto |

---

# 4. Hipótesis competidoras

El diseño factorial 2×2 permite distinguir tres hipótesis sobre el origen de la degradación:

| Hipótesis | Predicción operativa | Origen |
|---|---|---|
| **H1 — Solapamiento de dominio** | ΔASR(C) > ΔASR(A) y ΔASR(D) > ΔASR(B). Math (lejos de safety alignment) degrada más que Alpaca (más solapamiento implícito). | Chen et al. (2025), marco teórico |
| **H2 — Diversidad del dataset** | Dominios narrow degradan más independientemente de su contenido, por baja diversidad/overfitting de estilo. Efecto mitigable parcialmente por la mezcla. | Bianchi et al. (2024) |
| **H3 — Invariancia al dominio** | ΔASR(A) ≈ ΔASR(C) y ΔASR(B) ≈ ΔASR(D). La degradación es función del acto de SFT, no de su contenido. | Hipótesis nula |

Cualquiera de las tres es publicable. H1 es la más interesante por convertir el TFM en validación empírica directa de un paper teórico reciente.

---

# 5. Contribución científica

Este TFM produce cuatro contribuciones diferenciadas:

1. **Primer estudio factorial público** del efecto del dominio sobre la trayectoria de degradación de seguridad bajo SFT en un modelo open-source de 3B, con metodología epoch-by-epoch.
2. **Validación o refutación empírica** del marco teórico de Chen et al. (2025) — actualmente la única predicción cuantitativa sobre cómo el solapamiento de datos afecta el trade-off seguridad-capacidad.
3. **Caracterización tri-nivel de la robustez** (DirectRequest + HumanJailbreaks + PAIR) para los 4 experimentos. Distingue degradación de alineación nominal vs. robustez adversarial real.
4. **Código y artefactos abiertos** que sirven como plataforma de investigación reproducible para extender a otros dominios (código, conversación, summarization) o arquitecturas (Qwen, Mistral, Gemma).

**Conexión con Yuvaraj Govindarajulu (Bosch AIShield):** los resultados responden empíricamente a las hipótesis de Govindarajulu (2025) sobre fragilidad del alignment bajo SFT, extendiéndolas con el eje del dominio que no había sido considerado. El repositorio público proporciona un punto de entrada concreto para colaborar con el grupo de AIShield: extender los experimentos a modelos más grandes, incorporar fases RL sobre los checkpoints generados, o evaluar atacantes más sofisticados.

---

# 6. Stack técnico

| Componente | Herramienta |
|---|---|
| Modelo base | Llama 3.2 3B Instruct (HuggingFace) |
| Fine-tuning | TRL (HuggingFace) + LoRA (r=16) |
| Evaluación seguridad | HarmBench (DirectRequest, HumanJailbreaks) + PAIR custom |
| Evaluación capacidad | lm-evaluation-harness (GSM8K, Alpaca-eval) |
| Infraestructura | PyTorch + Accelerate, Slurm cluster (H100 96GB) |
| Tracking | Weights & Biases |
| Repositorio | GitHub público, licencia Apache 2.0 |

---

# 7. Riesgos clave y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Coste total del 2×2 + 3 ataques excede cuota cluster | Priorizar A → B → C → D; reportar parcial es publicable (al menos A vs C es resultado válido) |
| Confound del formato CoT (Math respuestas más largas que Alpaca) | Análisis adicional sobre longitud de respuesta; subset no-CoT |
| Baseline Llama 3.2 3B ya es bueno en GSM8K → capability gain bajo en C/D | Métrica complementaria con MATH (más difícil) |
| Inestabilidad de PAIR entre runs | Múltiples semillas en snapshots clave; reportar media ± std |
| Degradación no observable en 3 epochs en algún experimento | Qi et al. (2024) la miden en 1 epoch; si no aparece, revisar LR o extender a 5 epochs |

---

# 8. Diferencia respecto a la propuesta original

| Aspecto | Propuesta original ([descripcion.md](descripcion.md)) | Propuesta extendida (este documento) |
|---|---|---|
| Diseño | 1×2 (Alpaca puro vs. mixto) | **Factorial 2×2** (dominio × mezcla) |
| Experimentos | 2 (A, B) | **4 (A, B, C, D)** |
| Métricas de seguridad | HarmBench DirectRequest | **3 capas: DR + HJ + PAIR** |
| Pregunta principal | ¿Cuándo se degrada? ¿La mezcla lo evita? | ¿Depende del dominio? ¿La mezcla funciona igual en ambos? |
| Validación de literatura | Replica Qi et al. (2024) | **Replica Qi (2024) + valida Chen et al. (2025)** |
| Coste GPU estimado | ~30 h | **~150-200 h** |
| Duración estimada | 4-5 semanas | **7-8 semanas** |

La propuesta extendida convierte el TFM en una contribución más ambiciosa y publicable, manteniendo la propuesta original como **subset válido** (Exp A y B). Si por restricciones de tiempo o cómputo no fuera posible completar el 2×2, los resultados de Fase 1 (eje Alpaca) son ya publicables como replicación extendida de Qi et al. (2024) con la novedad de las tres capas de evaluación.
