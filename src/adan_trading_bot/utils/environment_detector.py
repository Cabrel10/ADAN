"""Environment detection and GPU optimization."""
import sys
import torch
import logging

logger = logging.getLogger(__name__)


def is_colab():
    """Détecte si le code s'exécute dans Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_gpu_available():
    """Vérifie si un GPU est disponible."""
    return torch.cuda.is_available()


def get_gpu_memory_gb():
    """Retourne la mémoire GPU disponible en GB."""
    if not is_gpu_available():
        return 0.0
    total_mem = torch.cuda.get_device_properties(0).total_memory
    return total_mem / (1024 ** 3)


def get_device():
    """Retourne le device optimal (cuda ou cpu)."""
    if is_gpu_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_optimal_vec_env_type():
    """Retourne le type d'environnement vectorisé optimal.

    Colab + GPU: DummyVecEnv (pas de pickle, GPU optimisé)
    Local + GPU: SubprocVecEnv (multiprocessing, GPU optimisé)
    CPU: DummyVecEnv (plus stable)
    """
    in_colab = is_colab()
    has_gpu = is_gpu_available()

    if in_colab:
        logger.info("🔍 Environnement: Google Colab détecté")
        if has_gpu:
            msg = ("✅ GPU disponible en Colab - "
                   "Utilisation de DummyVecEnv")
            logger.info(msg)
            return "dummy"
        else:
            logger.warning("⚠️ Pas de GPU en Colab")
            return "dummy"
    else:
        logger.info("🔍 Environnement: Local détecté")
        if has_gpu:
            msg = ("✅ GPU disponible localement - "
                   "Utilisation de SubprocVecEnv")
            logger.info(msg)
            return "subproc"
        else:
            logger.warning("⚠️ Pas de GPU localement")
            return "dummy"


def get_optimal_num_workers():
    """Retourne le nombre optimal de workers.

    Colab: 1 worker (DummyVecEnv)
    Local + GPU: 4 workers (SubprocVecEnv)
    CPU: 1 worker (DummyVecEnv)
    """
    in_colab = is_colab()
    has_gpu = is_gpu_available()

    if in_colab:
        logger.info("🔍 Colab: 1 worker (DummyVecEnv)")
        return 1
    elif has_gpu:
        logger.info("✅ Local + GPU: 4 workers (SubprocVecEnv)")
        return 4
    else:
        logger.info("⚠️ CPU only: 1 worker (DummyVecEnv)")
        return 1


def configure_gpu_optimization():
    """Configure les paramètres GPU pour une utilisation optimale."""
    if not is_gpu_available():
        logger.warning("⚠️ Pas de GPU disponible")
        return

    gpu_memory_gb = get_gpu_memory_gb()
    logger.info(f"📊 GPU Memory: {gpu_memory_gb:.1f} GB")

    if hasattr(torch.cuda, 'is_available') and \
            torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✅ TF32 activé pour meilleure performance")

    torch.cuda.empty_cache()
    logger.info("✅ Cache GPU vidé")


def get_environment_config():
    """Retourne la configuration complète de l'environnement."""
    in_colab = is_colab()
    has_gpu = is_gpu_available()
    gpu_memory_gb = get_gpu_memory_gb()
    vec_env_type = get_optimal_vec_env_type()
    num_workers = get_optimal_num_workers()
    device = get_device()

    config = {
        "in_colab": in_colab,
        "has_gpu": has_gpu,
        "gpu_memory_gb": gpu_memory_gb,
        "vec_env_type": vec_env_type,
        "num_workers": num_workers,
        "device": str(device),
        "device_obj": device,
    }

    logger.info(f"📋 Configuration: {config}")
    return config


def print_environment_info():
    """Affiche les informations d'environnement."""
    print("\n" + "="*60)
    print("🔍 INFORMATION D'ENVIRONNEMENT")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {is_gpu_available()}")
    if is_gpu_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {get_gpu_memory_gb():.1f} GB")
    print(f"Colab: {is_colab()}")
    print(f"Type VecEnv optimal: {get_optimal_vec_env_type()}")
    print(f"Nombre de workers: {get_optimal_num_workers()}")
    print("="*60 + "\n")
