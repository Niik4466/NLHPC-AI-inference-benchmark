# NLHPC AI INFERENCE BENCHMARK

## Overview

"NLHPC AI Inference Benchmark" is a benchmark created to test the use of ollama and vLLM in supercomputing environments that uses SLURM task manager and lmod software modules.
This project focuses on customization and monitoring of GPU resources utilization for both rocm and cuda backends.

## Project structure

- **run_test.py**: Main program to execute. It takes the execution parameters, informs the configuration to execute and launches at least <num_gpu> instances of sbatch jobs, the grace is that it will save the job_id to generate a task dependency, making that only one job uses the resources at the same time.

- **sbatch_generator.sh**: Bash script that generates sbatch scripts automatically to run, these scripts run the ollama service (for vLLM to be defined) and launch inference.py program.

- **inference.py**: This program take the arguments of the run_tests program and performs several queries to the ollama API, for vLLM to be defined all this while taking gpu metrics. It also saves a csv with the following data:
    - **Model**
    - **Params**
    - **Quantization**
    - **Tokens/s**
    - **Eval_duration**
    - **Eval_count**
    - **Theorical_size**
    - **Num_Gpus**
    - **Prompt**
    - **Response**
    - **GPU_{x}_Power_avg**
    - **GPU_{x}_Power_max**
    - **GPU_{x}_VRAM_usage_avg**
    - **GPU_{x}_VRAM_usage_max**

- **data.json**: Stores the execution parameters to perform the inference, such as the models to be executed and the prompts to be consulted
    - For a Ollama execution, it is necessary to put the model code, such as `llama2` or `llama2:70b-chat-fp16`
    - For a vLLM execution, to be defined

- **gpu_monitor_{backend}**: it is a class created to monitor the use of the gpu. It supports rocm and cuda backend

- **ollama_api.py**: contains all the querys that inference.py uses for ollama. Such as `api/generate`, `api/pull` and `api/show`

After the execution is finished. All the data will be in the `$RESULT_PATH`

## Instalation

#### Virtual environment

The program requires a virtual environment (venv). To install it, you need python3.9.19 or similar

1. Install the virtual environment
    ```bash
    python -m venv nlhpc_benchmark
    ```
2. Activate the virtual environment
    ```bash
    source nlhpc_benchmark/bin/activate
    ```
3. Install the python libraries
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

#### Ollama

It depends on the machine you are running the benchmark. Here is the ollama repository tutorial: https://github.com/ollama/ollama/blob/main/README.md

#### vLLM

To be defined

## Usage

To run the benchmark, you must follow these steps:

1. Activate the virtual environment (not necessary if already activated)
    ```bash
    source nlhpc_benchmark/bin/activate
    ```
2. Execute the program run_test.py
    ```bash
    python main.py <params> #can be python3 too
    ```

### Params

The `run_test.py` program includes several parameters to customize the execution (you can run `python run_test.py --help` to see this also).

- **`--gpus`, `-g`**:  
  Specifies the number of GPUs to use for the tests. This parameter determines how many GPUs will be utilized for the benchmark.  
  **Default**: `1`.  
  **Example**: `--gpus=4` or `-g 4`.

- **`--rep`, `-r`**:  
  Defines the number of repetitions for each test. This can be helpful for achieving consistent results by averaging multiple runs.  
  **Default**: `1`.  
  **Example**: `--rep=5` or `-r 5`.

- **`--partition`, `-p`**:  
  Specifies the cluster partition to use for testing.
  **Default**: `"mi210"`.  
  **Example**: `--partition=v100` or `-p v100`.

- **`--test_app`**:  
  Defines the application to test. The default is `ollama`, but additional supported options includes `vLLM`.  
  **Default**: `"ollama"`.  
  **Example**: `--test_app=vLLM`.

- **`--gpu_backend`**:  
  Specifies the GPU backend for the tests. The default backend is `cuda`, but alternative options such as `rocm` are supported.
  **Default**: `"cuda"`.  
  **Example**: `--gpu_backend=rocm`.

- **`--cluster`**:  
  Indicates the cluster where the tests will be executed. 
  **Default**: `"NLHPC"`.  
  **Example**: `--cluster=HPC-Cluster`.


### Environment variables

The program uses different environment variables that configures some aspects of the benchmark:

- **`TEST_DATA`**: 
Path to the JSON file containing the prompts and models to be used during testing. This file must include the input texts or configurations for the benchmark.
    **Default**: `data.json` 
    **Example**: `/path/to/data.json`.

- **`GPU_MIN_W_USAGE`**:
The minimum watt use for the gpu. This is used by `GpuMonitor` to start taking metrics.
    **Default**: `0`
    **Example**: `47`

- **`MAX_VRAM`**:
The maximum amount of VRAM (in GB) that the gpu can handle. This is used by the `inference.py` program to avoid the use of heavy models (that also avoid the use of cpu in the inference)
    **Default**: `64` 
    **Example**: `120`

- **`NLHPC_MEM_USAGE`**:
Set the maximum amount of RAM (in MB) that the job can utilice. Exclusive to NLHPC cluster at the moment
    **Default**: `250000`
    **Example**: `100000`

-  **`RESULT_PATH`**: 
Path to the directory where the results generated by the benchmark will be stored. The program will save the data in this location for further analysis.
    **Default**: `.` 
    **Example**: `/path/to/results`.

- **`OLLAMA_MODELS`**:
Tells the ollama service from which directory to fetch and download the models to be used in the `models.JSON` file
    **Default**: `~/.ollama/models`
    **Example**: `~/ollama_models`

- **`OLLAMA_HOST`**:
Tells the ollama service wich ip and port adress use for recieve the queries.
    **Default**: `127.0.0.1:11434`
    **Example**: `0.0.0.0:4466`
