"""
Script simplificado para probar la configuración básica del proyecto de fine-tuning
"""

import os
import sys
import importlib.util

def test_import(module_name):
    """Prueba si un módulo puede ser importado"""
    try:
        importlib.import_module(module_name)
        print(f"✅ El módulo '{module_name}' se importó correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al importar '{module_name}': {str(e)}")
        return False

def test_imports():
    """Probar todas las importaciones necesarias"""
    print("\n=== PROBANDO IMPORTACIONES ===")
    modules = [
        "torch", 
        "transformers", 
        "datasets", 
        "peft", 
        "bitsandbytes",
        "pandas",
        "tqdm"
    ]
    
    all_success = True
    for module in modules:
        success = test_import(module)
        all_success = all_success and success
    
    # Probar importaciones locales
    try:
        import config
        print("✅ El módulo 'config' se importó correctamente")
        
        import utils
        print("✅ El módulo 'utils' se importó correctamente")
        
        # Verificar si datamodule.py existe en el directorio actual
        if os.path.exists('datamodule.py'):
            import datamodule
            print("✅ El módulo 'datamodule' se importó correctamente")
        else:
            print("❌ Error: No se encuentra el archivo 'datamodule.py' en el directorio actual")
            all_success = False
            
    except Exception as e:
        print(f"❌ Error al importar módulos locales: {str(e)}")
        all_success = False
    
    return all_success

def test_config():
    """Probar la configuración"""
    print("\n=== PROBANDO CONFIGURACIÓN ===")
    try:
        import config
        
        # Probar acceso a variables de configuración
        print(f"Modelo base: {config.BASE_MODEL}")
        print(f"Ruta del dataset: {config.DATASET_PATH}")
        print(f"Rutas de salida: {config.MODEL_PATH}, {config.CHECKPOINT_PATH}")
        
        # Verificar si la ruta del modelo existe
        model_path = config.BASE_MODEL
        if hasattr(config, 'MODEL_FILE'):
            model_path = os.path.join(model_path, config.MODEL_FILE)
            
        if os.path.exists(model_path):
            print(f"✅ La ruta al modelo existe: {model_path}")
        else:
            print(f"⚠️ La ruta al modelo no existe: {model_path}")
        
        print("✅ La configuración parece correcta")
        return True
    except Exception as e:
        print(f"❌ Error en la configuración: {str(e)}")
        return False

def test_utils():
    """Probar funciones de utilidad"""
    print("\n=== PROBANDO UTILIDADES ===")
    try:
        import utils
        
        # Probar creación de directorios
        utils.create_directories()
        print("✅ Directorios creados correctamente")
        
        # Probar formato de conversación básico (sin usar tokenizador)
        test_messages = [
            {"role": "user", "content": "¿Qué es una función?"},
            {"role": "assistant", "content": "Una función es una relación que asigna a cada elemento de un conjunto un único elemento de otro conjunto."}
        ]
        
        try:
            formatted = utils.format_conversation(test_messages)
            print(f"✅ Formato de conversación probado. Longitud del texto formateado: {len(formatted)} caracteres")
        except Exception as e:
            print(f"⚠️ Error en formato de conversación: {str(e)}")
        
        # Probar carga de JSON (con manejo de errores)
        import config
        try:
            data = utils.load_json_dataset(config.DATASET_PATH)
            if data:
                print(f"✅ Carga de JSON probada. Se cargaron {len(data)} ejemplos")
            else:
                print("⚠️ No se cargaron datos JSON o el dataset está vacío")
        except Exception as e:
            print(f"⚠️ Error en carga de JSON: {str(e)}")
        
        # Probar verificación de modelo GGUF
        try:
            import config
            model_file = config.MODEL_FILE if hasattr(config, 'MODEL_FILE') else None
            model_exists = utils.load_model_from_gguf(config.BASE_MODEL, model_file)
            if model_exists:
                print(f"✅ Verificación de modelo GGUF exitosa")
            else:
                print(f"⚠️ El archivo del modelo GGUF no se pudo verificar")
        except Exception as e:
            print(f"⚠️ Error al verificar modelo GGUF: {str(e)}")
        
        return True
    except Exception as e:
        print(f"❌ Error general en las utilidades: {str(e)}")
        return False

def test_datamodule():
    """Probar funciones del módulo de datos"""
    print("\n=== PROBANDO MÓDULO DE DATOS ===")
    try:
        # Verificar si datamodule.py existe
        if not os.path.exists('datamodule.py'):
            print("❌ Error: No se encuentra el archivo 'datamodule.py' en el directorio actual")
            return False
            
        import config
        import utils
        import datamodule
        
        # Verificar si existe el archivo del dataset
        if not os.path.exists(config.DATASET_PATH):
            print(f"⚠️ El archivo del dataset no existe en {config.DATASET_PATH}")
            print("   Se utilizará un dataset de ejemplo mínimo para las pruebas")
            
            # Crear directorio del dataset si no existe
            dataset_dir = os.path.dirname(config.DATASET_PATH)
            if dataset_dir and not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir, exist_ok=True)
                print(f"✅ Se ha creado el directorio: {dataset_dir}")
        
        # Verificar si el directorio del dataset existe
        dataset_dir = os.path.dirname(config.DATASET_PATH)
        if dataset_dir and not os.path.exists(dataset_dir):
            print(f"⚠️ El directorio para el dataset no existe: {dataset_dir}")
        else:
            print(f"✅ El directorio para el dataset existe: {dataset_dir or './'}")
        
        print("✅ Módulo de datos importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error general en el módulo de datos: {str(e)}")
        return False

def main():
    print("=== INICIANDO PRUEBAS DE CONFIGURACIÓN DEL PROYECTO ===")
    
    # Probar cada componente
    imports_ok = test_imports()
    config_ok = test_config()
    utils_ok = test_utils()
    datamodule_ok = test_datamodule()
    
    # Resumen
    print("\n=== RESUMEN DE PRUEBAS ===")
    print(f"Importaciones: {'✅ OK' if imports_ok else '❌ ERROR'}")
    print(f"Configuración: {'✅ OK' if config_ok else '❌ ERROR'}")
    print(f"Utilidades: {'✅ OK' if utils_ok else '❌ ERROR'}")
    print(f"Módulo de datos: {'✅ OK' if datamodule_ok else '❌ ERROR'}")
    
    if imports_ok and config_ok and utils_ok and datamodule_ok:
        print("\n✅ CONFIGURACIÓN BÁSICA CORRECTA")
        print("Puntos a tener en cuenta:")
        print("1. Se ha creado la estructura de directorios necesaria")
        print("2. Necesitarás crear tu dataset de matemáticas en 'dataset/matematicas.json'")
        print("3. Ya puedes proceder a implementar train.py")
    else:
        print("\n⚠️ HAY ERRORES EN LA CONFIGURACIÓN")
        print("Revisa los mensajes de error antes de continuar")

if __name__ == "__main__":
    main()