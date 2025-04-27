"""
Script para probar las importaciones y la configuración de train.py
sin iniciar el entrenamiento completo
"""

import os
import sys
import importlib

def test_train_imports():
    """Probar si las importaciones de train.py funcionan correctamente"""
    print("\n=== PROBANDO IMPORTACIONES DE TRAIN.PY ===")
    
    # Verificar si train.py existe
    if not os.path.exists('train.py'):
        print("❌ Error: No se encuentra el archivo 'train.py' en el directorio actual")
        return False
    
    try:
        # Importar train.py como módulo
        import train
        print("✅ El módulo 'train' se importó correctamente")
        
        # Verificar las funciones importantes
        if hasattr(train, 'main'):
            print("✅ La función 'main' existe en train.py")
        else:
            print("❌ Error: No se encontró la función 'main' en train.py")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error al importar train.py: {str(e)}")
        return False

def test_training_setup():
    """Probar la configuración básica del entrenamiento sin iniciar el proceso completo"""
    print("\n=== PROBANDO CONFIGURACIÓN DE ENTRENAMIENTO ===")
    
    try:
        # Importar lo necesario
        import config
        import utils
        import datamodule
        
        # Probar crear directorios
        utils.create_directories()
        print("✅ Directorios de entrenamiento creados correctamente")
        
        # Verificar tokenizer
        try:
            tokenizer = utils.load_tokenizer("mistralai/Mistral-7B-v0.1")
            print("✅ Tokenizador cargado correctamente")
        except Exception as e:
            print(f"❌ Error al cargar tokenizador: {str(e)}")
            return False
        
        # Intentar cargar dataset
        try:
            print("Intentando cargar dataset para prueba (sin tokenización completa)...")
            data = utils.load_json_dataset(config.DATASET_PATH)
            if data:
                print(f"✅ Dataset cargado con {len(data)} ejemplos")
            else:
                print("⚠️ No se pudo cargar el dataset o está vacío")
        except Exception as e:
            print(f"⚠️ Error al cargar el dataset: {str(e)}")
        
        print("\n✅ Pruebas de configuración de entrenamiento completadas")
        return True
        
    except Exception as e:
        print(f"❌ Error en la configuración del entrenamiento: {str(e)}")
        return False

def main():
    print("=== INICIANDO PRUEBAS DE TRAIN.PY ===")
    
    # Probar componentes
    imports_ok = test_train_imports()
    setup_ok = test_training_setup() if imports_ok else False
    
    # Resumen
    print("\n=== RESUMEN DE PRUEBAS DE TRAIN.PY ===")
    print(f"Importaciones: {'✅ OK' if imports_ok else '❌ ERROR'}")
    print(f"Configuración: {'✅ OK' if setup_ok else '❌ ERROR'}")
    
    if imports_ok and setup_ok:
        print("\n✅ CONFIGURACIÓN DE ENTRENAMIENTO CORRECTA")
        print("El archivo train.py está listo para ejecutarse.")
        print("\nSi quieres iniciar el entrenamiento completo, ejecuta:")
        print("  python train.py")
        print("\nPara crear el archivo de evaluación antes de entrenar, es recomendable.")
    else:
        print("\n⚠️ HAY ERRORES EN LA CONFIGURACIÓN DE ENTRENAMIENTO")
        print("Revisa los mensajes de error antes de continuar")

if __name__ == "__main__":
    main()