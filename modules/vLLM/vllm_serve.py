import os, csv, json, argparse, time, sys
from models_data_utils import *
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ------ Parser config -------

parser = argparse.ArgumentParser(description="vLLM launch service program for NLHPC AI BENCHMARK")

parser.add_argument(
    "--gpus", "-g",
    type=int,
    help="Number of GPU's to use in the test",
    required=True,
    dest="num_gpus"
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the model to launch",
    required=True,
    dest="model_name"
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
model = args.model_name

if args.gpu_backend == 'rocm':
    from modules.gpu_monitor.gpu_monitor_rocm import GpuMonitor
else:
    from modules.gpu_monitor.gpu_monitor_cuda import GpuMonitor

# ------ Load necessary data ------

result_path = os.getenv('RESULT_PATH') or "."
test_data_json = os.getenv('TEST_DATA') or 'data.json'
vllm_bench_args_path = os.getenv('VLLM_BENCH_ARGS') or 'vllm_config.json'

try: 
    max_vram = int(os.getenv('MAX_VRAM') or "64")
except:
    max_vram = 64

with open(vllm_bench_args_path, 'r') as vllm_bench_args_file:
    vllm_bench_args = json.load(vllm_bench_args_file)

with open(test_data_json, 'r') as test_data_json_file:
    test_data = json.load(test_data_json_file)

prompts = test_data.get('prompts', [])

output_file = os.path.join(result_path, "vllm_inference_benchmark_results.csv")
file_exists = os.path.isfile(output_file)

# ----- Open the results file and save the results --------


try:
    config = vllm_bench_args[model]
except:
    raise RuntimeError(f"{model} configuration not found")

try:
    dtype_config = vllm_bench_args[model]["dtype"]
except ValueError:
    dtype_config = None

# --------- Load Model -----------

print(f"Loading {model} config...")
model_data = get_model_info(model_name=model, dtype=dtype_config)
print("Done!")

if model_data[2] > max_vram*args.num_gpus:
    print(f"The model {model} weight exceds avaible VRAM")
    raise RuntimeError(f"model config\nParams: {model_data[0]}\tQuantization: {model_data[1]}\t Weight: {model_data[2]}GB")

try:
    llm = LLM(
        model=model,
        max_num_seqs=config.get("max_num_seqs", 512),
        max_seq_len_to_capture=config.get("max_seq_len_to_capture", 16384),
        served_model_name=config.get("served_model_name", model),
        enable_chunked_prefill=config.get("enable_chunked_prefill", False),
        num_scheduler_steps=config.get("num_scheduler_steps", 15),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.97),
        enforce_eager=config.get("enforce-eager", False),
        dtype=config.get("dtype", "float16"),
        tensor_parallel_size=2
    )

except torch.OutOfMemoryError as oom_error:
    print(f"Out of memory error: {oom_error}")
    raise RuntimeError("Error loading the model, insufficient VRAM")

except RuntimeError:
    raise RuntimeError(f"Error executing the service")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=config.get("output_len", 128)
)

gpu_monitor = GpuMonitor(0.1)
gpu_monitor.start()
init = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed_time = time.time() - init
gpu_monitor.stop()

gpu_stats = gpu_monitor.get_stats()
sorted_gpu_stats = {key: gpu_stats[key] for key in sorted(gpu_stats.keys())}

total_tokens_per_sec = 0

for output in outputs:
    response_tokens = len(output.outputs[0].token_ids)
    total_tokens_per_sec += response_tokens / elapsed_time if elapsed_time > 0 else 0

# -------- Save the result -------

with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Model"
                        , "Params"
                        , "Quantization"
                        , "Tokens/s"
                        , "Theorical_size"
                        , "Num_Gpus"
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
    #Save the data
    writer.writerow([model,
                model_data[0],
                model_data[1],
                total_tokens_per_sec,
                model_data[2],
                args.num_gpus
                ]
                + list(sorted_gpu_stats.values()))
