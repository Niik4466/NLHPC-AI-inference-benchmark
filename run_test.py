import json, os, argparse, subprocess, sys, re

# ------------ FLAGS SETTINGS -----------------

parser = argparse.ArgumentParser(description="#----------- NLHPC AI BENCHMARK -----------#")

parser.add_argument(
    "--gpus", "-g",
    type=int,
    default=1,
    help="Number of GPU's to run tests. Default = 1",
    dest="num_gpus"
)

parser.add_argument(
    "--rep", "-r",
    type=int,
    default=1,
    help="Number of test repetitions. Default = 1",
    dest="rep"
)

parser.add_argument(
    "--partition", "-p",
    type=str,
    default="mi210",
    help="Partition to be used for testing",
    dest="partition"
)

parser.add_argument(
    "--test_app",
    type=str,
    default="ollama",
    help="Application to test. Default=ollama. Options:ollama-vLLM",
    dest="test_app"
)

parser.add_argument(
    "--gpu_backend",
    type=str,
    default="cuda",
    help="gpu backend to be used for the test. Default=cuda. Options:cuda-rocm",
    dest="gpu_backend"
)

parser.add_argument(
    "--cluster",
    type=str,
    default="NLHPC",
    help="cluster where the tests will be run",
    dest="cluster"
)

args = parser.parse_args()

# ---------------- LOAD THE TEST DATA ----------------

# Get the environment variables
test_data_json = os.getenv('TEST_DATA') or 'data.json'
result_path = os.getenv('RESULT_PATH') or '.'
gpu_min_w_usage = os.getenv('GPU_MIN_W_USAGE') or '47'

# Read the Json file
with open(test_data_json, 'r') as test_data_json_file:
    test_data = json.load(test_data_json_file)

# Obtain the list of models and prompts
models_name_list = test_data.get('models', [])
prompts_list = test_data.get('prompts', [])

# ---------------- INFO ----------------------

print("############ NLHPC AI BENCHMARK ############\n\n")
print("------------ TEST INFO ----------------")
print("Models:\t", models_name_list)
print("Prompts:\t", prompts_list)
print("Num of Gpus:\t", args.num_gpus)
print("Gpu Backend:\t", args.gpu_backend)
print("Gpu min w usage:\t", gpu_min_w_usage)
print("App to test:\t", args.test_app)
print("Repetitions:\t", args.rep)
print("Results dir:\t", result_path)
print("Cluster:\t", args.cluster)

if args.test_app == "ollama":
    ollama_models = os.getenv('OLLAMA_MODELS') or "~/.ollama/models"
    ollama_host = os.getenv('OLLAMA_HOST') or "127.0.0.1:11434"
    print("------------ OLLAMA CONFIG ----------------")
    print("Ollama host:\t", ollama_host)
    print("Ollama models:\t", ollama_models)
if args.test_app == "vLLM":
    print("------------ VLLM CONFIG ----------------")

print("--------------------------------------")

# -------------------- RUN THE TESTS --------------------

# aux function to find the job_id in the launched job

def get_job_id(result):
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return match.group(1)
    else:
        print(result.stderr)
        sys.exit()

if (args.cluster == "NLHPC"):
    mem_usage_nlhpc = os.getenv('NLHPC_MEM_USAGE') or "250000"
    job_id = -1
    for g in range(1,args.num_gpus+1):
        result = subprocess.run(
            [
                "modules/sbatch_generators/sbatch_generator_nlhpc.sh",
                "-p", args.partition,
                f"--gpus={g}",
                f"--test_app={args.test_app}",
                "-r", str(args.rep),
                f"--gpu_backend={args.gpu_backend}",
                f"--mem={mem_usage_nlhpc}",
                f"--job_id={job_id}"
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit()
        job_id = get_job_id(result=result)

elif (args.cluster == "patagon"):
    for g in range(1,args.num_gpus+1):
        result = subprocess.run(
            [
                "modules/sbatch_generators/sbatch_generator_patagon.sh",
                "-p", args.partition,
                f"--gpus={args.num_gpus}",
                f"--test_app={args.test_app}",
                "-r", str(args.rep),
                f"--gpu_backend={args.gpu_backend}"
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit()
        job_id = get_job_id(result=result)
