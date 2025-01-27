#!/bin/bash

# Función para mostrar el uso del script
usage() {
    echo "Usage: $0 -p <partition> --gpus=<num_gpus> --test_app=<app> -r <num_reps> --gpu_backend=<cuda/rocm>"
    echo "  -p <particion>       : Partition to be used for testing"
    echo "  --gpus=<num_gpus>    : Number of GPU's to run the tests"
    echo "  --test_app=<app>     : Application to test"
    echo "  -r <num_reps>        : Number of repetitions to run the test"
    echo "  --gpu_backend=<cuda/rocm>   : gpu backend to be used for the test"
    echo "  --mem=<RAM>          : MAX memory usage by the test"
    exit 1
}

# Inicializar variables
partition="mi210"
gpus=1
test_app="ollama"
repetitions=1
gpu_backend="cuda"
mem=250000

# Parsear los argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) partition="$2"; shift ;;
        --gpus=*) gpus="${1#*=}" ;;
        --test_app*) test_app="${1#*=}" ;;
        -r) repetitions="$2"; shift ;;
        --gpu_backend*) gpu_backend="${1#*=}" ;;
        --mem*) mem="${1#*=}" ;; 
        -h|--help) usage ;;
	    *) echo "Error: Opción desconocida: $1"; usage ;;
    esac
    shift
done

# Validaciones
if [[ -z "$partition" ]]; then
    echo "Error: Faltan parámetros obligatorios."
    usage
fi

# Validar que el número de GPUs sea un entero positivo
if ! [[ "$gpus" =~ ^[0-9]+$ ]]; then
    echo "Error: El parámetro --gpus debe ser un número entero positivo."
    exit 1
fi

# Crear un script temporal para SBATCH
temp_script="../../sbatch_script_$$.job"
cat <<EOF > "$temp_script"
#!/bin/bash
#SBATCH -J ollama_bench_${partition}
#SBATCH -p ${partition}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=${mem}
#SBATCH --gres=gpu:${gpus}
#SBATCH -o logs/ollama_bench_%j.out.err
#SBATCH -e logs/ollama_bench_%j.out.err

# ----------------Modulos----------------------------

ml ollama

# ---------------- Variables de entorno ----------------------------

export OLLAMA_HOST="$OLLAMA_HOST"
export OLLAMA_MODELS="$OLLAMA_MODELS"
export TEST_DATA="$TEST_DATA"
export RESULT_PATH="$RESULT_PATH"
export GPU_MIN_W_USAGE="$GPU_MIN_W_USAGE"
export MAX_VRAM="$MAX_VRAM"

# ---------------- Comandos --------------------
ollama serve &
sleep 3

# Cargar ambiente venv
source ollama_bench/bin/activate

# Ejecutar la inferencia
python inference.py -g ${gpus} -r ${repetitions} --test_app=${test_app} --gpu_backend=${gpu_backend}
EOF

# Enviar el script a la cola de trabajos
chmod 755 "$temp_script"

if [[ job_id == -1 ]]; then
    sbatch "$temp_script"
else
    sbatch --dependency=afterany:"$job_id" "$temp_script"
fi

# Limpiar
sleep 2
rm -f "$temp_script"