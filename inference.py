import json, argparse, csv, os, sys
from functools import partial

# --------------- PARSER CONFIG --------------

parser = argparse.ArgumentParser(description="Inference program for NLHPC AI BENCHMARK")

parser.add_argument(
    "--gpus", "-g",
    type=int,
    help="Number of GPUS to use in the test",
    required=True,
    dest="num_gpus"
)

parser.add_argument(
    "-r",
    type=int,
    help="Number of  times to repeat the inference",
    required=False,
    default=1,
    dest="rep"
)

parser.add_argument(
    "--test_app",
    type=str,
    help="Application to test",
    required=False,
    default="ollama",
    dest="test_app"
)

parser.add_argument(
    "--gpu_backend",
    type=str,
    help="gpu backend to be used in the test.",
    required=False,
    default="cuda",
    dest="gpu_backend"
)


args = parser.parse_args()

# --------------- LOAD NECESSARY MODULES --------------

if (args.gpu_backend == "rocm"):
    from modules.gpu_monitor.gpu_monitor_rocm import GpuMonitor
else:
    from modules.gpu_monitor.gpu_monitor_cuda import GpuMonitor

if (args.test_app == "vLLM"):
    pass
else:
    from modules.ollama.ollama_api import *

# --------------- LOAD ENVIRONMENT VARIABLES ----------

ollama_host = os.getenv('OLLAMA_HOST') or "127.0.0.1:11434"
test_data_json = os.getenv('TEST_DATA') or 'data.json'
result_path = os.getenv('RESULT_PATH') or "."

try: 
    max_vram = int(os.getenv('MAX_VRAM') or "64")
except:
    max_vram = 64

# -------------- LOAD TEST DATA ---------------

# Read the Json file
with open(test_data_json, 'r') as test_data_json_file:
    test_data = json.load(test_data_json_file)

# Obtain the list of models and prompts
models_name_list = test_data.get('models', [])
prompts_list = test_data.get('prompts', [])

# ------------ DOWNLOAD DE MODELS WITH OLLAMA API -----------

for model in models_name_list:
    download_model_ollama(model, ollama_host)

# ------------ GET MODELS INFORMATION WITH OLLAMA API -----------
if args.test_app == "ollama":
    model_parameters_and_quantization = list(map(partial(obtain_model_data_ollama, port=ollama_host), models_name_list))
elif args.test_app == "vLLM":
    pass

models_weight = [calculate_theorical_weight(parameters, quantization) for parameters, quantization in model_parameters_and_quantization]

# ------------ OPEN THE CSV FILE AND START THE INFERENCE ----------


if (args.test_app == "ollama"):

    output_file = os.path.join(result_path, "ollama_benchmark_results.csv")
    file_exists = os.path.isfile(output_file)

    gpu_monitor = GpuMonitor(0.1)

    with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Model"
                            , "Params"
                            , "Quantization"
                            , "Tokens/s"
                            , "Eval_duration"
                            , "Eval_count"
                            , "Theorical_size"
                            , "Num_Gpus"
                            , "Prompt"
                            , "Response"
                            , "GPU_0_Power_avg"
                            , "GPU_0_Power_max"
                            , "GPU_0_VRAM_usage_avg"
                            , "GPU_0_VRAM_usage_max"
                            , "GPU_1_Power_avg"
                            , "GPU_1_Power_max"
                            , "GPU_1_VRAM_usage_avg"
                            , "GPU_1_VRAM_usage_max"
                            , "GPU_2_Power_avg"
                            , "GPU_2_Power_max"
                            , "GPU_2_VRAM_usage_avg"
                            , "GPU_2_VRAM_usage_max"
                            , "GPU_3_Power_avg"
                            , "GPU_3_Power_max"
                            , "GPU_3_VRAM_usage_avg"
                            , "GPU_3_VRAM_usage_max"
                            , "GPU_4_Power_avg"
                            , "GPU_4_Power_max"
                            , "GPU_4_VRAM_usage_avg"
                            , "GPU_4_VRAM_usage_max"
                            , "GPU_5_Power_avg"
                            , "GPU_5_Power_max"
                            , "GPU_5_VRAM_usage_avg"
                            , "GPU_5_VRAM_usage_max"])

        for r in range(args.rep):
            for model, model_data, weight in zip(models_name_list, model_parameters_and_quantization, models_weight):
                if weight > max_vram:
                    continue
                for prompt in prompts_list:
                    gpu_monitor.start()
                    prompt_eval_duration, prompt_eval_count, response = query_ollama(prompt.strip(), model.strip(), port=ollama_host)
                    gpu_monitor.stop()
                    # If the values are errors, we exit the program
                    if (isinstance(prompt_eval_duration, str)):
                        print(prompt_eval_count)
                        sys.exit()
                    # Calculate tokens/s
                    tokens_per_second = prompt_eval_count / (prompt_eval_duration / 1e9)
                    gpu_stats = gpu_monitor.get_stats()
                    sorted_gpu_stats = {key: gpu_stats[key] for key in sorted(gpu_stats.keys())}
                    # Write in the CSV file
                    writer.writerow([model, 
                                    model_data[0], #Params
                                    model_data[1], #Quantization
                                    tokens_per_second, 
                                    prompt_eval_duration,
                                    prompt_eval_count,
                                    weight,
                                    args.gpus,
                                    prompt,
                                    response] 
                                    + list(sorted_gpu_stats.values()))
elif args.test_app == "vLLM":
    pass