"""
Funciones de utilidad para el proceso de fine-tuning de un modelo matemático
"""

import os
import torch
from transformers import AutoTokenizer, AutoConfig
import bitsandbytes as bnb
import json
import random

def create_directories():
    """Crear las carpetas necesarias si no existen"""
    from config import MODEL_PATH, CHECKPOINT_PATH
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def load_tokenizer(model_name):
    """Cargar el tokenizer del modelo base"""
    try:
        # Para GGUF de LM Studio, usamos el tokenizador de Mistral directamente
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
        print("Usando tokenizador de Mistral (compatible con Mathstral)")
    except Exception as e:
        print(f"Error al cargar tokenizador, usando fallback: {str(e)}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Configurar el tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configurar la plantilla de chat si existe
    from config import CHAT_TEMPLATE
    if CHAT_TEMPLATE and hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = CHAT_TEMPLATE
        
    return tokenizer

def print_trainable_parameters(model):
    """
    Imprime el número de parámetros entrenables vs total
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"Parámetros entrenables: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}% del total)")
    print(f"Total de parámetros: {all_param:,d}")

def format_chat_to_instruction_response(messages):
    """Convertir un formato de chat a pares de instrucción-respuesta"""
    # Asumimos que los mensajes vienen en pares usuario-asistente
    if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        return {
            "instruction": messages[0]["content"],
            "response": messages[1]["content"]
        }
    return None

def format_conversation(messages):
    """Formatea una conversación para que el modelo la entienda"""
    # Formatear manualmente sin depender del tokenizador
    formatted_chat = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted_chat += f"[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            formatted_chat += f"{msg['content']}</s>"
    
    return formatted_chat

def load_json_dataset(file_path):
    """Cargar un dataset en formato JSON de chat matemático"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Advertencia: El archivo {file_path} no existe.")
        # Crear una estructura mínima para las pruebas
        return [{"messages": [{"role": "user", "content": "¿Qué es una función?"}, 
                             {"role": "assistant", "content": "Una función matemática..."}]}]
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return []

def save_model_adapter(model, output_dir):
    """Guardar el adaptador LoRA del modelo"""
    model.save_pretrained(output_dir)
    print(f"Adaptador LoRA guardado en: {output_dir}")

def find_all_linear_layers(model):
    """Encontrar todas las capas lineales en el modelo para cuantización"""
    lora_target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora_target_modules.append(name.split('.')[-1])
    
    return list(set(lora_target_modules))

def load_model_from_gguf(model_path, model_file=None):
    """
    Función para cargar un modelo en formato GGUF (para modelos de LM Studio)
    """
    # Simplemente verificamos que el archivo existe
    if model_file:
        full_path = os.path.join(model_path, model_file)
    else:
        full_path = model_path
        
    print(f"Verificando ruta del modelo GGUF: {full_path}")
    
    if os.path.exists(full_path):
        print(f"✅ El archivo del modelo existe en: {full_path}")
        return True
    else:
        print(f"❌ El archivo del modelo no existe en: {full_path}")
        return False

def validate_math_response(response, problem):
    """
    Función simple para validar respuestas matemáticas (puede expandirse)
    """
    # Detectar si hay números en la respuesta cuando debería haberlos
    if any(c.isdigit() for c in problem) and not any(c.isdigit() for c in response):
        print("⚠️ Advertencia: La respuesta no contiene números pero la pregunta sí.")
        return False
    
    # Detectar inconsistencias básicas en fórmulas
    if "=" in response:
        left_side = response.split("=")[0].strip()
        if not left_side or left_side.isspace():
            print("⚠️ Advertencia: Fórmula matemática mal formada.")
            return False
    
    return True