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
    echo "  --job_id=<job_id>    : The job_id of the previus job to set the dependency. Default=-1"
    exit 1
}

# Inicializar variables
partition="mi210"
gpus=1
test_app="ollama"
repetitions=1
gpu_backend="cuda"
mem=250000
job_id=-1

# Parsear los argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) partition="$2"; shift ;;
        --gpus=*) gpus="${1#*=}" ;;
        --test_app*) test_app="${1#*=}" ;;
        -r) repetitions="$2"; shift ;;
        --gpu_backend*) gpu_backend="${1#*=}" ;;
        --mem*) mem="${1#*=}" ;; 
        --job_id*) job_id="${1#*=}" ;;
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
temp_script="sbatch_script_$$.job"
cat <<EOF > "$temp_script"
#!/bin/bash
#SBATCH -J vllm_bench_${partition}
#SBATCH -p ${partition}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=${mem}
#SBATCH --gres=gpu:${gpus}
#SBATCH -o logs/vllm_bench_%j.out.err
#SBATCH -e logs/vllm_bench_%j.out.err
export OLLAMA_KEEP_ALIVE=2m0s
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_SCHED_SPREAD=true

# ----------------Modulos----------------------------

# ---------------- Variables de entorno ----------------------------

export TEST_DATA="$TEST_DATA"
export RESULT_PATH="$RESULT_PATH"
export GPU_MIN_W_USAGE="$GPU_MIN_W_USAGE"
export MAX_VRAM="$MAX_VRAM"

# ---------------- Comandos --------------------

docker run -v /home/ai_inference_db:/home/ai_inference_db -v /home/intern02:/home/intern02 -it --rm --device=/dev/kfd --device=/dev/dri --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/infiniband --ulimit memlock=-1:-1 --shm-size 256g --cap-add=IPC_LOCK rocm/vllm:rocm6.3.2_mi210_ubuntu22.04_py3.12_vllm_0.7.1.dev103_ib /bin/bash -c "

export HF_HOME=/home/ai_inference_db/models/
export HF_DATASETS_CACHE=/home/ai_inference_db/data/

export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=112
export TORCH_BLAS_PREFER_HIPBLASLT=1
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
export TORCHINDUCTOR_FREEZING=1
export TORCHINDUCTOR_CPP_WRAPPER=1
export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export PYTORCH_MIOPEN_FORCE_NO_CACHE=0
export TORCH_BLADE_ROCM=1
export TORCH_HIP_FORCE_PERSISTENT_CACHE=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_JIT_LOG_LEVEL=0
export TORCH_COMPILE_MAX_AUTOTUNE=1

# Cargar ambiente venv
source nlhpc_benchmark/bin/activate

# Ejecutar la inferencia
python inference.py -g ${gpus} -r ${repetitions} --test_app=${test_app} --gpu_backend=${gpu_backend}
"
EOF

# Enviar el script a la cola de trabajos
chmod 755 "$temp_script"

sbatch "$temp_script"

# Limpiar
sleep 2
rm -f "$temp_script"