import sys, io, os, time
from vllm.entrypoints.api_server_benchmark import main as vllm_bench
from vllm import LLM, SamplingParams

def run_vllm_bench(model_name:str, config, num_gpus=1):
    """
    
    """
    # Prepare the arguments for the bench
    try:
        cant_gpu_config_index = config.index("--model-name")
        config[cant_gpu_config_index] = num_gpus
    except:
        config.append("--model-name")
        config.append(model_name)

    sys.stdout = io.StringIO()
    # Execute the benchmark
    vllm_bench(config)
    # Get the stdout of the bench
    vllm_bench_output = sys.stdout.getvalue()

    sys.stdout = sys.__stdout__
    # Search the tokens/s
    tokens_per_second = None
    for line in vllm_bench_output.split("\n"):
        if "tokens/s" in line.lower():
            tokens_per_second = float(line.split()[-1])
    
    # return the value
    return tokens_per_second

def run_inference_vllm(model, prompt:str, num_gpus=1):
    # Create the sampling
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )

    start_time = time.time()
    # Start the inference
    output = model.generate([prompt], sampling_params)
    # Get time metrics
    elapsed_time = time.time() - start_time
    # Calculate tokens in the answer
    num_tokens = len(output.outputs[0].tokens_ids)

    # Calculate the tokens/s
    tokens_per_second = num_tokens / elapsed_time

    return tokens_per_second, num_tokens, elapsed_time, output.outputs[0].text