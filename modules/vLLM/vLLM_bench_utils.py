import sys, io, os, time, subprocess, re, torch, signal
from vllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

def run_vllm_bench(model_name:str, config, num_gpus=1):
    """
    
    """

    # Prepare the arguments for the bench
    try:
        cant_gpu_config_index = config.index("--tensor-parallel-size")
        config[cant_gpu_config_index+1] = str(num_gpus)
    except ValueError:
        config.append("--tensor-parallel-size")
        config.append(str(num_gpus))

    print("Running bench...")
    sys.stdout.flush()
    # Execute the benchmark
    result = subprocess.run(
        ["python", 
        "/home/intern02/vLLM/vllm/benchmarks/benchmark_throughput.py"]
        + config,
        capture_output=True,
        text=True,
        timeout=1000
    )
    
    if (result.returncode != 0):
        print("Benchmark Error\n\n", result.stdout, result.stderr, "\nBenchmark Execution failed")
        sys.stdout.flush()
        return -1, -1
        # raise RuntimeError("Benchmark execution failed")
    
    # Obtain result line:
    output = result.stdout
    match_tokens_per_second = re.search(r"([\d\.]+) output tokens/s", output)
    match_requests_per_second = re.search(r"([\d\.]+) requests/s", output)

    print("Done!")

    if match_tokens_per_second:
        return float(match_tokens_per_second.group(1)), float(match_requests_per_second.group(1))
    else:
        raise RuntimeError("Tokens_per_second output not found in", result.stdout)

def run_inference_vllm(model, prompts:list[str], num_gpus=1, config={}):
    """
    
    """

    try:
        llm = LLM(
            model=model,
            max_num_seqs=config.get("max_num_seqs", 512),
            max_seq_len_to_capture=config.get("max_seq_len_to_capture", 16384),
            served_model_name=config.get("served_model_name", model),
            enable_chunked_prefill=config.get("enable_chunked_prefill", False),
            num_scheduler_steps=config.get("num_scheduler_steps", 15),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            enforce_eager=config.get("enforce-eager", False),
            dtype=config.get("dtype", "float16"),
            tensor_parallel_size=num_gpus
        )

    except torch.OutOfMemoryError as oom_error:
        print(f"Out of memory error: {oom_error}")
        print("Error loading the model, insufficient VRAM")
        torch.cuda.empty_cache()
        return -1

    except RuntimeError:
        return -1

    if (llm == None):
        return -1

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=config.get("output_len", 128)
    )
    
    init = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - init

    total_tokens_per_sec = 0

    for output in outputs:
        response_tokens = len(output.outputs[0].token_ids)

        total_tokens_per_sec += response_tokens / elapsed_time if elapsed_time > 0 else 0


    return total_tokens_per_sec
