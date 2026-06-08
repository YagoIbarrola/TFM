Qi et al. (2024) — ICLR (el seminal) — Mostraron que Alpaca SFT degrada seguridad. También probaron variantes: SFT con datos explícitamente dañinos (degradación brutal), con "identity-shifting" (role-play) y con datos benignos. Pero NO compararon sistemáticamente dominios benignos entre sí. La pregunta del dominio queda abierta.

He et al. (2024) — "What's in Your 'Safe' Data?" — Identifican que la similitud representacional entre los datos SFT y los datos de safety alignment del pre-training es lo que predice degradación. Datos que activan circuitos similares a los de safety degradan más. Esto predice que matemáticas (representación muy distinta) podría degradar más o menos dependiendo de la dirección del drift.

Chen et al. (2025) — marco teórico (citado en tu descripcion.md) — Prueban formalmente que mayor similitud entre los datos de seguridad y los datos de capacidad mejora el trade-off. Las matemáticas son el ejemplo paradigmático de baja similitud con datos de seguridad, así que esta teoría predice que math SFT degradará más que Alpaca SFT. Tu experimento sería una validación empírica directa de su teoría.

Bianchi et al. (2024) — "Safety-Tuned LLaMAs" — Investigan SFT en dominios específicos (médico, legal). Muestran que dominios estrechos degradan safety. Math no está en su tabla pero estaría en la misma familia.

Lermen et al. (2024) — "LoRA Fine-tuning Efficiently Undoes Safety Training" — Muestran que basta con ~100 ejemplos benignos para destruir alineación. Refuerza la idea de que el problema no es la cantidad, sino la naturaleza del cambio representacional.

Pelrine et al. (2024) — "Exploiting Novel GPT-4 APIs" — Demostraron que fine-tuning de GPT-4 sobre datos completamente benignos (incluyendo dominios técnicos) abre vulnerabilidades de seguridad. Trabajo cercano pero en modelo cerrado.


DoReMi (Xie et al., NeurIPS 2023)	Modelo proxy determina ratios óptimos entre dominios para pretraining	(a) Pretraining no SFT, (b) ratios estáticos una vez determinados, (c) no específico de safety
ODM — Online Data Mixing (Albalak et al., 2023)	Bandidos multibrazo ajustan ratios entre dominios durante el pretraining	(a) Pretraining no SFT, (b) optimiza loss agregada no trade-off objetivo, (c) no safety
DoGE (Fan et al., 2024)	Reweighting de dominios con señal de gradiente generalizada	Similar a DoReMi pero con feedback más rico. Mismo gap respecto a safety SFT.
Bianchi et al. (2024) — Safety-Tuned LLaMAs	Estudia mezcla estática de safety data en SFT (1-1000 ejemplos)	Tu base teórica. Demuestra que 500 ejemplos bastan. Pero solo ratios fijos, no curriculum.
Constitutional AI (Anthropic, 2022)	Iteración de SFT phases donde el modelo critica y revisa sus respuestas	Es un curriculum de fases discretas no de mezcla continua. Y usa critique automático, no eval externo.
GradNorm / PCGrad	Balanceo de gradientes en multi-task learning para que ningún objetivo domine	Es lo más cercano conceptualmente. Pero opera a nivel de gradientes, no de mezcla de datos.
MGDA / Pareto fine-tuning	Optimización multi-objetivo Pareto-aware	Marco matemático aplicable pero no aplicado a safety SFT que yo conozca