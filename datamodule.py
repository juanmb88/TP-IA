"""
Módulo para cargar y preparar los datos de entrenamiento para el tutor matemático
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from utils import format_chat_to_instruction_response, format_conversation, load_json_dataset, load_tokenizer
from config import MAX_LENGTH, DATASET_PATH

def prepare_dataset():
    """Preparar y cargar el dataset de conceptos matemáticos"""
    try:
        # Cargar desde un archivo JSON
        data = load_json_dataset(DATASET_PATH)
        
        # Convertir los datos al formato necesario
        formatted_data = []
        
        for item in data:
            if "messages" in item:
                # Extraer la instrucción (pregunta) y respuesta
                instruction_response = format_chat_to_instruction_response(item["messages"])
                if instruction_response:
                    formatted_data.append(instruction_response)
        
        # Verificar que tenemos datos
        if not formatted_data:
            raise ValueError("No se pudieron extraer pares de instrucción-respuesta válidos del dataset")
        
        # Convertir a Dataset de HuggingFace
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        print(f"Dataset de matemáticas cargado con {len(dataset)} ejemplos")
        return dataset
    
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return None

def prepare_conversations_dataset():
    """Preparar el dataset manteniendo el formato de conversación"""
    try:
        # Cargar desde un archivo JSON
        data = load_json_dataset(DATASET_PATH)
        
        # Mantener el formato de conversación completo
        formatted_data = []
        
        for item in data:
            if "messages" in item:
                formatted_data.append({"messages": item["messages"]})
        
        # Verificar que tenemos datos
        if not formatted_data:
            raise ValueError("No se pudieron extraer conversaciones válidas del dataset")
        
        # Convertir a Dataset de HuggingFace
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        print(f"Dataset de conversaciones matemáticas cargado con {len(dataset)} ejemplos")
        return dataset
    
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return None

def preprocess_function(examples, tokenizer):
    """Preprocesar y tokenizar los ejemplos de instrucción-respuesta"""
    # Formatear los ejemplos según el formato de instrucción
    formatted_texts = []
    
    for instruction, response in zip(examples['instruction'], examples['response']):
        # Crear una conversación en formato de lista de mensajes
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        # Formatear la conversación
        formatted_text = format_conversation(messages)
        formatted_texts.append(formatted_text)
    
    # Tokenizar los textos
    tokenized = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Prepara las etiquetas (igual que los input_ids para entrenamiento de lenguaje)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def preprocess_conversations(examples, tokenizer):
    """Preprocesar y tokenizar las conversaciones completas"""
    formatted_texts = []
    
    for messages in examples['messages']:
        formatted_text = format_conversation(messages)
        formatted_texts.append(formatted_text)
    
    # Tokenizar los textos
    tokenized = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Prepara las etiquetas (igual que los input_ids para entrenamiento de lenguaje)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def get_tokenized_dataset(tokenizer, use_conversation_format=True):
    """Obtener el dataset tokenizado"""
    if use_conversation_format:
        dataset = prepare_conversations_dataset()
        if dataset is None:
            raise ValueError("No se pudo cargar el dataset de conversaciones")
        
        # Procesar el dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_conversations(examples, tokenizer),
            batched=True,
            remove_columns=['messages']
        )
    else:
        dataset = prepare_dataset()
        if dataset is None:
            raise ValueError("No se pudo cargar el dataset")
        
        # Procesar el dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=['instruction', 'response']
        )
    
    return tokenized_dataset

def split_dataset(dataset, test_size=0.1, seed=42):
    """Dividir el dataset en conjuntos de entrenamiento y evaluación"""
    # Calcular tamaños
    dataset_size = len(dataset)
    test_samples = max(1, int(dataset_size * test_size))
    train_samples = dataset_size - test_samples
    
    # Dividir el dataset
    splits = dataset.train_test_split(
        test_size=test_samples,
        train_size=train_samples,
        seed=seed
    )
    
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    
    print(f"Dataset dividido: {train_samples} ejemplos de entrenamiento, {test_samples} ejemplos de prueba")
    
    return train_dataset, test_dataset

def analyze_dataset_stats(dataset):
    """Analizar estadísticas básicas del dataset de matemáticas"""
    if "instruction" in dataset.column_names and "response" in dataset.column_names:
        # Calcular longitudes
        instruction_lengths = [len(text.split()) for text in dataset["instruction"]]
        response_lengths = [len(text.split()) for text in dataset["response"]]
        
        # Estadísticas básicas
        avg_instruction_len = sum(instruction_lengths) / len(instruction_lengths)
        avg_response_len = sum(response_lengths) / len(response_lengths)
        max_instruction_len = max(instruction_lengths)
        max_response_len = max(response_lengths)
        
        print("\n--- Estadísticas del Dataset ---")
        print(f"Total de ejemplos: {len(dataset)}")
        print(f"Longitud promedio de instrucciones: {avg_instruction_len:.1f} palabras")
        print(f"Longitud promedio de respuestas: {avg_response_len:.1f} palabras")
        print(f"Instrucción más larga: {max_instruction_len} palabras")
        print(f"Respuesta más larga: {max_response_len} palabras")
        
        # Detectar patrones comunes en temas matemáticos (ejemplo simplificado)
        math_keywords = ["ecuación", "función", "integral", "derivada", "límite", 
                          "algebra", "geometría", "cálculo", "estadística", "probabilidad"]
        
        keyword_counts = {keyword: 0 for keyword in math_keywords}
        
        for instruction in dataset["instruction"]:
            for keyword in math_keywords:
                if keyword in instruction.lower():
                    keyword_counts[keyword] += 1
        
        # Mostrar distribución de temas
        print("\nDistribución aproximada de temas:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100
            if percentage > 0:
                print(f"  - {keyword}: {percentage:.1f}% ({count} ejemplos)")
    
    elif "messages" in dataset.column_names:
        # Analizar dataset en formato de conversación
        message_counts = [len(conv) for conv in dataset["messages"]]
        avg_messages = sum(message_counts) / len(message_counts)
        
        print("\n--- Estadísticas del Dataset de Conversaciones ---")
        print(f"Total de conversaciones: {len(dataset)}")
        print(f"Promedio de mensajes por conversación: {avg_messages:.1f}")
        print(f"Conversación más larga: {max(message_counts)} mensajes")