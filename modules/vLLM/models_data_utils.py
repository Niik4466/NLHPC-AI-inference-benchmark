import torch
from transformers import AutoModelForCausalLM

def get_params_quantization(model_name:str):
    """
    
    """
    # Load de model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    num_params = f"{num_params}b"

    # Get the type of quantization
    if 'quantization' in model.config.to_dict():
        quantization = model.config.to_dict()['quantization']
    else:
        quantization = "None"

    # Return the values
    return num_params, quantization