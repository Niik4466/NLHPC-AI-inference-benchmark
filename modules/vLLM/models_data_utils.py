from transformers import AutoModelForCausalLM
import torch

def get_model_info(model_name: str, dtype=None):
    """
    Calculates the approximate size of a GB model based on its parameters and accuracy.
    """
    # Cargar el modelo
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Obtener el número total de parámetros
    num_params = sum(p.numel() for p in model.parameters()) 

    # Obtener la precisión (dtype)
    if dtype == None:
        dtype = next(model.parameters()).dtype 

    dtype_size = {
        "float32": 32, 
        "float16": 16,
        "bfloat16": 16, 
        "int8": 8,
        "int4": 4,
        "torch.float32": 32,
        "torch.float16": 16,
        "torch.bfloat16": 16,
        "torch.int8": 8,
        "torch.int4": 4
    }.get(str(dtype), 32)  # Si no está en la lista, asumir float32

    # Calcular el tamaño en GB
    model_size_gb = (num_params * dtype_size) / (8*10**9)  # Convertir de bytes a GB
    num_params = num_params / 1e9
    num_params = f"{num_params:.3f}B"
    return num_params, dtype, model_size_gb 
