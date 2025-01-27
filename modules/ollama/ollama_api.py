import requests, re

def query_ollama(prompt, model="llama2", port="127.0.0.1:11434"):
    """
    Sends a direct query to the Ollama API and returns the token(s) along with the prompt.

    Args:
        prompt (str): The query you wish to send to the model.
        model (str): The model to use (default, 'llama2'). 
        port (str): The port where the Ollama service runs (default, 127.0.0.1:11434).
    Returns:
        tuple: prompt eval duration, prompt eval count (or an error message if something goes wrong)
    """
    # Definimos la url a consultar y los headers de la consulta
    url = f"http://{port}/api/generate"
    headers = {
        "Content-Type": "application/json",
    }

    # Definimos la consulta
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        # Realizamos la consulta a la API. Importante considerar que se nos devuelve un stream de Jsons no uno unico
        response = requests.post(url, json=payload, headers=headers)
        # Terminamos de monitorear la gpu una vez esta lista la respuesta

        response.raise_for_status()  # Lanza una excepción para errores HTTP

        # Cargamos los datos del JSON de respuesta
        data = response.json()

        # Extraer la duración de la evaluación y la cantidad de evaluaciones
        prompt_eval_duration = data.get("eval_duration", 0)  # En Nanosegundos
        prompt_eval_count = data.get("eval_count", 0) 
        response = data["message"]["content"]
        return prompt_eval_duration, prompt_eval_count, response

    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
        print("Respuesta completa:", response.text)
        return f"Error: {e}", f"Error: {e}", f"Error:{e}"


def download_model_ollama(model="llama2", port="127.0.0.1:11434"):
    """
    Send a request to Ollama to download the model if it does not exist.

    Args:
        model (str): The model to use (default, 'llama2'). 
        port (str): The port where the Ollama service runs (default, 127.0.0.1:11434).
    
    Return:
        bool: True if the download was successful, false if not
    """
    
    # Definimos la url a consultar y los headers de la consulta
    url = f"http://{port}/api/pull"
    headers = {
        "Content-Type": "application/json",
    }
    
    # Definimos la consulta
    payload = {
        "model": model,
        "stream": False
    }
    
    # Realizamos la consulta al endpoint especificado
    response = requests.post(url, json=payload, headers=headers)

    # Obtenemos el resultado de la consulta
    data = response.json()

    # Retornamos True si fue posible hacer pull al modelo, false si no 
    result = data.get("status")
    return (result == "success")


def obtain_model_data_ollama(model="llama2", port="127.0.0.1:11434"):
    """
    Obtains the number of billions of parameters and the quantization of the specified model.

    Args:
        model (str): the model to use (by default: 'llama2')
        port (str): The port where the Ollama service runs (by default, 127.0.0.1:11434).
    Return:
        tuple: number of parameters, quantization
    """

    # Definimos la url y los headers de la consulta
    url = f"http://{port}/api/show"
    headers = {
        "Content-Type": "application/json",
    }
    
    # Definimos la consulta
    payload = {
        "model": model,
        "stream": "false"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        parameter_size = data["details"]["parameter_size"]
        quantization = data["details"]["quantization_level"]
        return parameter_size, quantization
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener los datos del modelo {model}")
        return f"Error {e}", f"Error {e}"

def calculate_theorical_weight(parameters, quantization):
    """
    Calculates the theoretical weight of a model based on the number of parameters and quantization.    

    Args:
        parameters (str): Number of parameters as string (e.g. "13B", "7B"). 
        quantization (str): Quantization as string (e.g. "Q4_0", "F16").
    Returns:
        float: Theorical size in GB.
    """
    # Convertir los parámetros a número
    scale = {"B": 1e9, "M": 1e6}  # Escalas para billones (B) y millones (M)
    unit = parameters[-1].upper()  # Último carácter para determinar escala
    if unit not in scale:
        raise ValueError(f"Unidad desconocida en parámetros: {unit}")

    num_parameters = float(parameters[:-1]) * scale[unit]  # Convertir a número total de parámetros

    # Extraer el número de bits del string de cuantización
    match = re.match(r"[A-Za-z]*(\d+)", quantization)  # Captura el primer número
    if not match:
        raise ValueError(f"Formato de cuantización inválido: {quantization}")
    
    bits = int(match.group(1))  # Obtiene el primer número del string
    parameter_size = bits / 8  # Convertir bits a bytes

    # Calcular peso total en bytes
    size_in_bytes = num_parameters * parameter_size
    size_in_gb = size_in_bytes / (1e9)  # Convertir bytes a GB

    return size_in_gb