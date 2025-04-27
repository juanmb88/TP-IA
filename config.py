"""
Configuración para el proceso de fine-tuning de un modelo matemático
utilizando el modelo Mathstral-7B local desde LM Studio
"""

# Rutas - Ajustadas para tu estructura específica
MODEL_PATH = "models/"  
CHECKPOINT_PATH = "checkpoints/"
DATASET_PATH = "dataset/matematicas.json"

# Modelo base a utilizar - Ruta al modelo Mathstral que encontraste
# Ahora separando el directorio del archivo para evitar duplicación
BASE_MODEL = "C:/Users/User/.cache/lm-studio/models/lmstudio-community/mathstral-7B-v0.1-GGUF"
MODEL_FILE = "mathstral-7B-v0.1-Q4_K_M.gguf"  # Nombre específico del archivo del modelo

# Configuración de entrenamiento - Optimizada para un modelo matemático
LEARNING_RATE = 2e-4      # Tasa de aprendizaje
NUM_EPOCHS = 5            # Aumentado a 5 para mejorar el aprendizaje en matemáticas
BATCH_SIZE = 1            # Tamaño de lote reducido para CPU
GRADIENT_ACCUMULATION_STEPS = 8  # Aumentado para compensar el batch pequeño
WARMUP_STEPS = 100        # Pasos de calentamiento para el optimizador
MAX_STEPS = 1000          # Número máximo de pasos
SAVE_STEPS = 200          # Cada cuántos pasos guardar un checkpoint
LOGGING_STEPS = 50        # Cada cuántos pasos registrar estadísticas
MAX_LENGTH = 512          # Longitud máxima de contexto para tokenización

# Configuración LoRA (Low-Rank Adaptation) - Optimizada para conceptos matemáticos
LORA_R = 16               # Rank
LORA_ALPHA = 32           # Alpha
LORA_DROPOUT = 0.05       # Dropout para regularización
LORA_TARGET_MODULES = [   # Capas del modelo a adaptar
    "q_proj",             # Proyecciones de Query
    "k_proj",             # Proyecciones de Key
    "v_proj",             # Proyecciones de Value
    "o_proj",             # Proyección de Output
    "gate_proj",          # Proyección de Gate
    "up_proj",            # Proyección Up
    "down_proj",          # Proyección Down
]

# Configuración de cuantización (para reducir uso de memoria)
BITS = 4                  # Bits para la cuantización (4 u 8)
DOUBLE_QUANT = True       # Doble cuantización para mayor eficiencia

# Configuración del formato para conversación - Adaptado para Mathstral
CHAT_TEMPLATE = """<s>{{#each messages}}{{#ifEquals role "user"}}[INST] {{content}} [/INST]{{/ifEquals}}{{#ifEquals role "assistant"}}{{content}}</s>{{/ifEquals}}{{/each}}"""

# Nombre del modelo fine-tuned
MODEL_NAME = "matematicas-tutor-mathstral"

# Configuraciones adicionales para matemáticas
INFERENCE_TEMPERATURE = 0.3   # Temperatura baja para respuestas matemáticas precisas
TOP_P = 0.9                   # Filtrado núcleo para generación
REPETITION_PENALTY = 1.2      # Penalización para evitar repeticiones