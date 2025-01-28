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

