"""
Script para diagnosticar la configuración de CUDA y GPU
"""

import torch
import os
import subprocess
import sys

def check_cuda_installation():
    """Verificar instalación de CUDA"""
    print("=== DIAGNÓSTICO DE GPU/CUDA ===\n")
    
    # Verificar si CUDA está disponible según PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch detecta CUDA: {'✅ Sí' if cuda_available else '❌ No'}")
    
    # Verificar versión de PyTorch
    print(f"Versión de PyTorch: {torch.__version__}")
    
    # Verificar si PyTorch fue compilado con CUDA
    cuda_compiled = torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else False
    print(f"PyTorch compilado con CUDA: {'✅ Sí' if cuda_compiled else '❌ No'}")
    
    # Información sobre GPUs disponibles
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Número de GPUs disponibles: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {gpu_name} (Capacidad CUDA: {gpu_capability[0]}.{gpu_capability[1]})")
    else:
        print("No se detectaron GPUs disponibles para PyTorch")
        
        # Buscar GPUs NVIDIA en el sistema
        try:
            if sys.platform == 'win32':
                # En Windows, intentar usar nvidia-smi
                try:
                    nvidia_smi = subprocess.check_output('nvidia-smi', shell=True)
                    print("\nGPUs NVIDIA detectadas en el sistema:")
                    print(nvidia_smi.decode('utf-8'))
                except subprocess.CalledProcessError:
                    print("\nNo se pudo ejecutar 'nvidia-smi'. ¿Está instalado el driver de NVIDIA?")
            elif sys.platform == 'linux':
                # En Linux, intentar lspci
                try:
                    gpu_info = subprocess.check_output('lspci | grep -i nvidia', shell=True)
                    print("\nGPUs NVIDIA detectadas en el sistema:")
                    print(gpu_info.decode('utf-8'))
                except subprocess.CalledProcessError:
                    print("\nNo se detectaron GPUs NVIDIA con 'lspci'")
        except Exception as e:
            print(f"\nError al intentar detectar GPUs del sistema: {str(e)}")
    
    # Verificar variables de entorno CUDA
    cuda_path = os.environ.get('CUDA_PATH', 'No definido')
    cuda_home = os.environ.get('CUDA_HOME', 'No definido')
    
    print("\n=== Variables de entorno CUDA ===")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")
    
    # Resumen y recomendaciones
    print("\n=== DIAGNÓSTICO FINALIZADO ===")
    
    if not cuda_available:
        print("\n⚠️ CUDA no está disponible para PyTorch. Posibles soluciones:")
        print("1. Instalar los drivers NVIDIA más recientes para tu GPU")
        print("2. Reinstalar PyTorch con soporte CUDA usando el comando apropiado de pytorch.org:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Verificar que CUDA Toolkit está instalado (versión compatible con PyTorch)")
        print("4. Asegurarse que la GPU es compatible con la versión de CUDA requerida")
    else:
        print("\n✅ CUDA está correctamente configurado para PyTorch")

    print("\nRecuerda que también necesitas instalar bitsandbytes con soporte GPU:")
    print("pip install bitsandbytes --upgrade")

if __name__ == "__main__":
    check_cuda_installation()