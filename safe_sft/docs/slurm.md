> **Página visible a externos sin autenticación**

> **Tipos de nodos disponibles:**
> - Partición CPU (40 vCPU, 128GB)
> - Partición GPU (40 vCPU, 128GB, H100 96GB VRAM): `--gres=H_100_NVL`
> - Partición GPU (20 vCPU, 128GB, L4 24GB VRAM): `--gres=L4`
>
> Recuerda que si no especificas recurso con gres, entrará en la primera gpu libre

# Uso de Slurm

Guía de usuario para el uso del cluster de Slurm desplegado.

Ver también cómo [enviar trabajos al clúster con sbatch](knowhow:slurm:guias_de_uso) y cómo [ejecutar Jupyter en Slurm mediante sbatch](knowhow:slurm:ejemplo_workflow_jupyternb).

## 1. Requisitos previos

- Tener cuenta de usuario activa: `LDAP ANTS` o [LDAP EXTERNOS](https://externos.inf.um.es/fusiondirectory).
- Tener acceso SSH al clúster: `155.54.210.99`.
- Tener permiso de uso en al menos una partición:
  - `CPU`
  - `GPU` (si aplica)

## 2. Acceso al clúster

Conexión típica:

```bash
ssh <USUARIO_LDAP>@<HOST_IP>
```

Te dejará en `/slurm/home/<USUARIO>`.

## 3. Comandos básicos que vas a usar siempre

- Ver estado general del clúster:

```bash
sinfo
```

- Ver cola de trabajos:

```bash
squeue
```

- Ver solo tus trabajos:

```bash
squeue -u <USUARIO>
```

- Ver detalle de un job:

```bash
scontrol show job <JOB_ID>
```

- Cancelar un job:

```bash
scancel <JOB_ID>
```

## 4. Flujo estándar de trabajo

### 4.1 Crear un script de job (`.sbatch`)

Ejemplo CPU:

```bash
cat > job_cpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=cpu_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

echo "Usuario: $USER"
echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Trabajo CPU de prueba"
sleep 20
EOF
```

Ejemplo GPU:

```bash
cat > job_gpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

nvidia-smi || true
echo "Trabajo GPU de prueba"
sleep 20
EOF
```

### 4.2 Enviar el job

```bash
sbatch job_cpu.sbatch
```

Slurm devolverá algo como:

```
Submitted batch job <JOB_ID>
```

### 4.3 Monitorizar

```bash
squeue -u <USUARIO>
```

Estados comunes:

- `PD`: pendiente (pendiente de recursos/políticas).
- `R`: ejecutando.
- `CG`: terminando.
- `CD`: completado.
- `F`: fallado.

## 5. Dónde salen stdout y stderr

En este despliegue, si no defines `--output`/`--error`, Slurm aplica rutas por defecto:

- `stdout` → `/slurm/home/%u/output/%j/stdout.txt`
- `stderr` → `/slurm/home/%u/output/%j/stderr.txt`

Equivalente para un job real:

- `/slurm/home/<USUARIO>/output/<JOB_ID>/stdout.txt`
- `/slurm/home/<USUARIO>/output/<JOB_ID>/stderr.txt`

Consultar:

```bash
cat /slurm/home/<USUARIO>/output/<JOB_ID>/stdout.txt
cat /slurm/home/<USUARIO>/output/<JOB_ID>/stderr.txt
```

## 6. Scratch por job (importantísimo)

Durante la ejecución, el clúster crea automáticamente:

- `/scratch/slurm/<USUARIO>/<JOB_ID>`

Y al terminar el job, se elimina automáticamente.

Recomendación:

- Copia resultados importantes desde scratch a una ruta persistente antes de finalizar.
- Usa rutas persistentes para salidas finales (por ejemplo `/slurm/home/<USUARIO>/...`).

## 7. Job interactivo

Para abrir una sesión interactiva en un nodo:

```bash
srun --partition=<PARTICION_CPU> --ntasks=1 --cpus-per-task=2 --time=00:30:00 --pty bash
```

Para GPU:

```bash
srun --partition=<PARTICION_GPU> --gres=gpu:1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --pty bash
```

## 8. Arrays de jobs

Ejemplo:

```bash
cat > job_array.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=array_demo
#SBATCH --partition=<PARTICION_CPU>
#SBATCH --array=1-10
#SBATCH --time=00:05:00

echo "Task ID: $SLURM_ARRAY_TASK_ID"
sleep 5
EOF
```

Enviar:

```bash
sbatch job_array.sbatch
```

## 9. Dependencias entre jobs

Enviar un primer job:

```bash
jid1=$(sbatch --parsable job_cpu.sbatch)
```

Enviar un segundo job que empiece cuando termine bien el primero:

```bash
sbatch --dependency=afterok:${jid1} job_cpu.sbatch
```

## 10. Historial y contabilidad

Ver historial básico:

```bash
sacct -u <USUARIO> --starttime today
```

Formato ampliado:

```bash
sacct -u <USUARIO> --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## 11. Ver recursos de nodos

```bash
sinfo -N -l
```

Detalle de un nodo:

```bash
scontrol show node <NODO>
```

## 12. Errores comunes y solución rápida

- `Invalid account or account/partition combination specified`
  - Tu usuario no está asociado a esa cuenta/partición.
  - Contacta con administración.

- Job en `PD` mucho tiempo
  - Puede faltar recursos o prioridad.
  - Revisa motivo:

```bash
squeue -j <JOB_ID> -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

- No aparece output esperado
  - Revisa rutas por defecto `/slurm/home/<USUARIO>/output/<JOB_ID>/`
  - Revisa `stderr.txt`

- Error de GPU (`Requested node configuration is not available`)
  - `--gres=gpu:<N>` no cuadra con lo disponible.
  - Verifica con `sinfo -N -l`

## 13. Buenas prácticas de uso

- Solicita solo recursos que realmente necesitas.
- Define `--time` realista.
- Usa scripts reproducibles y versionados.
- Guarda resultados finales en almacenamiento persistente.
- Limpia datos intermedios pesados.

## 14. Plantilla mínima recomendada

```bash
#!/bin/bash
#SBATCH --job-name=<NOMBRE_JOB>
#SBATCH --partition=<PARTICION_CPU_O_GPU>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N_CPUS>
#SBATCH --time=<HH:MM:SS>
#SBATCH --mem=<RAM_MB_O_GB>

set -euo pipefail
echo "Job $SLURM_JOB_ID en $(hostname)"

# Tu carga de trabajo aquí
```

## 15. Checklist rápido antes de enviar

- [ ] Script `.sbatch` tiene `#!/bin/bash`.
- [ ] Partición correcta (`CPU` o `GPU`).
- [ ] Recursos correctos (`cpu`, `mem`, `time`, `gres` si GPU).
- [ ] Ruta de datos de entrada existe.
- [ ] Resultado final se guarda en ruta persistente.
