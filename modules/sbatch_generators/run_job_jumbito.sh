# Función para mostrar el uso del script
usage() {
    echo "Usage: $0 -p <partition> --gpus=<num_gpus> --test_app=<app> -r <num_reps> --gpu_backend=<cuda/rocm>"
    echo "  -p <particion>       : Partition to be used for testing"
    echo "  --gpus=<num_gpus>    : Number of GPU's to run the tests"
    echo "  --test_app=<app>     : Application to test"
    echo "  -r <num_reps>        : Number of repetitions to run the test"
    echo "  --gpu_backend=<cuda/rocm>   : gpu backend to be used for the test"
    exit 1
}

# Inicializar variables
partition="mi210"
gpus=1
test_app="ollama"
repetitions=1
gpu_backend="cuda"

# Parsear los argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) partition="$2"; shift ;;
        --gpus=*) gpus="${1#*=}" ;;
        --test_app*) test_app="${1#*=}" ;;
        -r) repetitions="$2"; shift ;;
        --gpu_backend*) gpu_backend="${1#*=}" ;;
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

# Validate GPU backend
if [[ "$gpu_backend" != "cuda" && "$gpu_backend" != "rocm" ]]; then
    echo "Error: Invalid GPU backend. Must be 'cuda' or 'rocm'."
    exit 1
fi

# -------------- START THE JOB --------------

export OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_MODELS=$OLLAMA_MODELS
export TEST_DATA=$TEST_DATA
export RESULT_PATH=$RESULT_PATH
export GPU_MIN_W_USAGE=$GPU_MIN_W_USAGE
export MAX_VRAM=$MAX_VRAM
export OLLAMA_KEEP_ALIVE="4m0s"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_SCHED_SPREAD=true

# Activate the virtual environment
source ~/NLHPC_IA_inference_benchmark/nlhpc_benchmark/bin/activate

export CUDA_VISIBLE_DEVICES=""

# Start the job
for ((i=0; i < ${gpus}; i++)); do
    # Add the current GPU to CUDA_VISIBLE_DEVICES
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES="$i"
    else
        export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$i"
    fi

    # Start ollama serve in the background
    ~/ollama/bin/ollama serve &
    OLLAMA_PID=$!

    # Run the inference script
    python inference.py -g $i -r ${repetitions} --test_app=${test_app} --gpu_backend=${gpu_backend}

    # Wait for ollama serve to finish
    kill ${OLLAMA_PID}
    wait ${OLLAMA_PID}
done
