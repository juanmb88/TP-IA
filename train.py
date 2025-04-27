"""
Script principal para el fine-tuning del modelo de asistente matemático
con soporte para GPU + CPU
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import Trainer
from datamodule import get_tokenized_dataset, split_dataset, analyze_dataset_stats
from utils import (
    create_directories, 
    load_tokenizer, 
    print_trainable_parameters, 
    save_model_adapter,
    load_model_from_gguf
)
import config

def check_gpu():
    """Verificar disponibilidad de GPU y configurar dispositivo"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"✅ GPU disponible: {device_count} dispositivo(s)")
        for i, name in enumerate(device_names):
            print(f"   - GPU {i}: {name}")
        return True
    else:
        print("⚠️ No se detectó GPU. Usando CPU para entrenamiento (será más lento)")
        return False

def main():
    print("\n=== INICIANDO FINE-TUNING DEL MODELO MATEMÁTICO ===\n")
    
    # Verificar GPU
    has_gpu = check_gpu()
    
    # Crear directorios necesarios
    create_directories()
    
    # Verificar que el modelo GGUF existe
    model_file = config.MODEL_FILE if hasattr(config, 'MODEL_FILE') else None
    if not load_model_from_gguf(config.BASE_MODEL, model_file):
        print("❌ Error: No se puede acceder al modelo. Verificar la ruta en config.py")
        return
    
    print(f"\n1. Preparando configuración para: {config.BASE_MODEL}")
    
    # Para modelos GGUF, usaremos un modelo base de Hugging Face para el fine-tuning
    print("NOTA: Como estamos usando un modelo en formato GGUF (optimizado para inferencia),")
    print("      usaremos un modelo base compatible para el fine-tuning (Mistral).")
    print("      Luego aplicaremos los adaptadores LoRA al modelo final.")
    print("      El modelo base que usaremos es: mistralai/Mistral-7B-v0.1")
    
    # Configuración de cuantización adaptada a disponibilidad de GPU
    print("\n2. Configurando cuantización...")
    if has_gpu:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True if config.BITS == 4 else False,
            load_in_8bit=True if config.BITS == 8 else False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=config.DOUBLE_QUANT,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"  # Distribuir automáticamente entre GPU/CPU
        print("   Configuración optimizada para GPU + CPU")
    else:
        quantization_config = None
        device_map = "cpu"  # Forzar CPU
        print("   Configuración para CPU")
    
    # Cargar el modelo base - usamos Mistral ya que es compatible con Mathstral
    try:
        print("\n3. Cargando modelo base para entrenamiento...")
        
        # Si no se puede autenticar con Hugging Face para Mistral, usar una alternativa
        try:
            model_id = "mistralai/Mistral-7B-v0.1"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            print(f"✅ Modelo {model_id} cargado correctamente")
        except Exception as e:
            print(f"⚠️ Error al cargar Mistral: {str(e)}")
            print("   Intentando con modelo alternativo (gemma-2b)...")
            
            model_id = "google/gemma-2b"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            print(f"✅ Modelo alternativo {model_id} cargado correctamente")
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo base: {str(e)}")
        print("Sugerencia: Verifica tu conexión a internet y espacio en disco.")
        return
    
    # Cargar el tokenizer
    print("\n4. Cargando tokenizador...")
    tokenizer = load_tokenizer(model_id)
    
    # Preparar el modelo para entrenamiento cuantizado (si corresponde)
    print("\n5. Preparando modelo para entrenamiento...")
    if has_gpu and quantization_config:
        model = prepare_model_for_kbit_training(model)
        print("   Modelo preparado para entrenamiento cuantizado")
    else:
        print("   Usando configuración estándar para CPU")
    
    # Configuración de LoRA
    print("\n6. Aplicando configuración LoRA...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.LORA_TARGET_MODULES,
    )
    
    # Obtener modelo con LoRA aplicado
    model = get_peft_model(model, lora_config)
    
    # Mostrar información de parámetros entrenables
    print_trainable_parameters(model)
    
    # Preparar dataset
    print("\n7. Preparando dataset...")
    try:
        tokenized_dataset = get_tokenized_dataset(tokenizer, use_conversation_format=True)
        print(f"✅ Dataset tokenizado con {len(tokenized_dataset)} ejemplos")
        
        # Analizar estadísticas del dataset
        print("\n8. Analizando dataset...")
        dataset_stats = analyze_dataset_stats(tokenized_dataset)
        
        # Dividir en train/test
        train_dataset, eval_dataset = split_dataset(tokenized_dataset, test_size=0.1)
        
    except Exception as e:
        print(f"❌ Error al preparar el dataset: {str(e)}")
        print("Sugerencia: Verifica que el archivo dataset/matematicas.json existe y tiene formato correcto.")
        return
    
    # Configuración de entrenamiento adaptada a GPU/CPU
    print("\n9. Configurando parámetros de entrenamiento...")
    
    # Ajustar batch size automáticamente si hay GPU
    batch_size = config.BATCH_SIZE
    if has_gpu:
        # Si hay GPU, podemos usar el batch definido en config
        print(f"   Usando batch size de {batch_size} (GPU disponible)")
    else:
        # Si no hay GPU, reducir el batch para evitar problemas de memoria
        original_batch = batch_size
        batch_size = max(1, batch_size // 2)
        print(f"   Reduciendo batch size de {original_batch} a {batch_size} (solo CPU)")
    
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_PATH,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.MAX_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        optim="paged_adamw_8bit" if has_gpu else "adamw_torch",
        fp16=has_gpu,  # Solo usar fp16 si hay GPU
        evaluation_strategy="steps",
        eval_steps=config.SAVE_STEPS,
        dataloader_num_workers=4 if has_gpu else 0,  # Usar múltiples workers si hay GPU
    )
    
    # Inicializar Trainer
    print("\n10. Inicializando entrenador...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Iniciar entrenamiento
    print("\n11. ¡INICIANDO ENTRENAMIENTO DEL ASISTENTE MATEMÁTICO!")
    print("=" * 60)
    print(f"Dispositivo: {'GPU + CPU' if has_gpu else 'Solo CPU'}")
    print(f"Épocas totales: {config.NUM_EPOCHS}")
    print(f"Tamaño de lote: {batch_size} (x{config.GRADIENT_ACCUMULATION_STEPS} acumulación)")
    print(f"Tasa de aprendizaje: {config.LEARNING_RATE}")
    print(f"Pasos totales: {min(config.MAX_STEPS, len(train_dataset) * config.NUM_EPOCHS // (batch_size * config.GRADIENT_ACCUMULATION_STEPS))}")
    print("=" * 60)
    print("\nEl entrenamiento puede tardar horas dependiendo de tu hardware.")
    print("Puedes monitorear el progreso en la terminal.")
    print("Presiona Ctrl+C en cualquier momento para detener (se guardará el último checkpoint).")
    print("\nIniciando...")
    
    try:
        trainer.train()
        print("\n✅ ¡Entrenamiento completado exitosamente!")
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        return
    
    # Guardar modelo y adaptador LoRA
    print("\n12. Guardando modelo final...")
    final_model_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    save_model_adapter(model, final_model_path)
    
    # Guardar también el tokenizer con la plantilla de chat
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ ¡Proceso completo! Modelo guardado en {final_model_path}")
    print("\nPuedes usar el modelo fine-tuned para conceptos matemáticos ejecutando:")
    print(f"  python evaluate.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}")
        import traceback
        traceback.print_exc()