#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de statut complet pour ADAN v2.1.
Affiche l'état du système, performances, exchange integration et apprentissage continu.
"""
import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.tree import Tree
from rich.rule import Rule

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

console = Console()

def check_online_learning_capabilities():
    """Vérifie les capacités d'apprentissage continu."""
    learning_status = {
        "scripts_available": False,
        "config_ready": False,
        "models_compatible": False,
        "buffer_support": False
    }
    
    # Scripts d'apprentissage continu
    learning_scripts = [
        "scripts/paper_trade_agent.py",
        "scripts/online_learning_agent.py", 
        "scripts/human_feedback_trading.py"
    ]
    
    learning_status["scripts_available"] = all(os.path.exists(script) for script in learning_scripts)
    
    # Configuration apprentissage continu
    config_files = [
        "config/main_config.yaml",
        "GUIDE_APPRENTISSAGE_CONTINU.md"
    ]
    
    learning_status["config_ready"] = all(os.path.exists(config) for config in config_files)
    
    # Modèles disponibles
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        learning_status["models_compatible"] = len(model_files) > 0
    
    # Support technique
    learning_status["buffer_support"] = os.path.exists("src/adan_trading_bot/exchange_api")
    
    return learning_status

def check_conda_env():
    """Vérifie l'environnement conda."""
    try:
        result = subprocess.run("conda info --envs | grep trading_env", 
                               shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_exchange_connection():
    """Vérifie la connexion exchange."""
    exchange_status = {
        "api_key_set": False,
        "secret_key_set": False,
        "connection_test": False,
        "markets_loaded": False
    }
    
    # Vérifier variables d'environnement
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret_key = os.environ.get("BINANCE_TESTNET_SECRET_KEY")
    
    exchange_status["api_key_set"] = bool(api_key)
    exchange_status["secret_key_set"] = bool(secret_key)
    
    if api_key and secret_key:
        try:
            # Test de connexion rapide
            result = subprocess.run("python test_exchange_connector.py", 
                                   shell=True, capture_output=True, text=True, timeout=30)
            exchange_status["connection_test"] = result.returncode == 0
            
            if result.returncode == 0:
                exchange_status["markets_loaded"] = "markets loaded" in result.stdout.lower()
        except:
            exchange_status["connection_test"] = False
    
    return exchange_status

def check_data_files():
    """Vérifie les fichiers de données."""
    data_status = {}
    
    # Données source
    source_files = [
        "data/new/ADAUSDT_features.parquet",
        "data/new/BNBUSDT_features.parquet", 
        "data/new/BTCUSDT_features.parquet",
        "data/new/ETHUSDT_features.parquet",
        "data/new/XRPUSDT_features.parquet"
    ]
    
    # Données processed multi-timeframe
    processed_files = [
        "data/processed/unified/1m_train_merged.parquet",
        "data/processed/unified/1m_val_merged.parquet",
        "data/processed/unified/1m_test_merged.parquet",
        "data/processed/unified/1h_train_merged.parquet",
        "data/processed/unified/1d_train_merged.parquet"
    ]
    
    # Scalers pour apprentissage continu
    scaler_files = [
        "data/scalers_encoders/scaler_1m.joblib",
        "data/scalers_encoders/scaler_1h.joblib",
        "data/scalers_encoders/scaler_1d.joblib"
    ]
    
    data_status['source'] = {
        'files': len([f for f in source_files if os.path.exists(f)]),
        'total': len(source_files),
        'size_mb': sum(os.path.getsize(f)/(1024*1024) for f in source_files if os.path.exists(f))
    }
    
    data_status['processed'] = {
        'files': len([f for f in processed_files if os.path.exists(f)]),
        'total': len(processed_files),
        'size_mb': sum(os.path.getsize(f)/(1024*1024) for f in processed_files if os.path.exists(f))
    }
    
    data_status['scalers'] = {
        'files': len([f for f in scaler_files if os.path.exists(f)]),
        'total': len(scaler_files),
        'size_mb': sum(os.path.getsize(f)/(1024*1024) for f in processed_files if os.path.exists(f))
    }
    
    return data_status

def check_config_files():
    """Vérifie les fichiers de configuration."""
    config_files = [
        "config/main_config.yaml",
        "config/data_config_cpu.yaml",
        "config/data_config_gpu.yaml", 
        "config/agent_config_cpu.yaml",
        "config/agent_config_gpu.yaml",
        "config/environment_config.yaml",
        "config/logging_config.yaml"
    ]
    
    status = {}
    for config_file in config_files:
        status[os.path.basename(config_file)] = {
            'exists': os.path.exists(config_file),
            'size': os.path.getsize(config_file) if os.path.exists(config_file) else 0,
            'modified': datetime.fromtimestamp(os.path.getmtime(config_file)) if os.path.exists(config_file) else None
        }
    
    return status

def check_models():
    """Vérifie les modèles entraînés."""
    models_dir = Path("models")
    models_status = {}
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.zip"):
            stat = model_file.stat()
            models_status[model_file.name] = {
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'age_hours': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
            }
    
    return models_status

def check_logs():
    """Vérifie les logs récents."""
    log_files = list(Path(".").glob("training_*.log")) + list(Path(".").glob("*.log"))
    recent_logs = []
    
    for log_file in log_files:
        stat = log_file.stat()
        age_hours = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
        
        if age_hours < 24:  # Logs des dernières 24h
            recent_logs.append({
                'name': log_file.name,
                'size_mb': stat.st_size / (1024 * 1024),
                'age_hours': age_hours
            })
    
    return sorted(recent_logs, key=lambda x: x['age_hours'])

def get_system_performance():
    """Analyse les performances système."""
    perf = {}
    
    # Utilisation disque
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        perf['disk'] = {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'usage_pct': (used / total) * 100
        }
    except:
        perf['disk'] = None
    
    # Mémoire (approximation via fichiers temporaires)
    try:
        import psutil
        mem = psutil.virtual_memory()
        perf['memory'] = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'usage_pct': mem.percent
        }
    except:
        perf['memory'] = None
    
    return perf

def analyze_last_training():
    """Analyse le dernier entraînement."""
    logs = check_logs()
    if not logs:
        return None
    
    # Prendre le log le plus récent
    latest_log = logs[0]['name']
    
    try:
        with open(latest_log, 'r') as f:
            content = f.read()
        
        analysis = {}
        
        # Rechercher des métriques
        if "Capital:" in content:
            lines = content.split('\n')
            capital_lines = [l for l in lines if "Capital:" in l and "$" in l]
            if capital_lines:
                try:
                    last_capital_line = capital_lines[-1]
                    capital_part = last_capital_line.split("Capital:")[1].split()[0]
                    analysis['final_capital'] = float(capital_part.replace("$", "").replace(",", ""))
                except:
                    pass
        
        if "ROI:" in content:
            lines = content.split('\n')
            roi_lines = [l for l in lines if "ROI:" in l and "%" in l]
            if roi_lines:
                try:
                    last_roi_line = roi_lines[-1]
                    roi_part = last_roi_line.split("ROI:")[1].split("%")[0].strip()
                    analysis['roi_pct'] = float(roi_part.replace("+", ""))
                except:
                    pass
        
        if "timesteps" in content.lower():
            if "5000" in content or "5,000" in content:
                analysis['training_size'] = "Quick (5K)"
            elif "50000" in content or "50,000" in content:
                analysis['training_size'] = "Standard (50K)"
            elif "100000" in content or "100,000" in content:
                analysis['training_size'] = "Extended (100K)"
            else:
                analysis['training_size'] = "Custom"
        
        analysis['log_age_hours'] = logs[0]['age_hours']
        
        return analysis
        
    except Exception as e:
        return None

def create_status_report():
    """Crée le rapport de statut complet."""
    
    # En-tête principal
    console.print(Rule("[bold blue]🚀 ADAN Trading Agent - Status Report[/bold blue]", style="blue"))
    console.print(f"[dim]Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
    
    # 1. État de l'environnement
    env_table = Table(title="🔧 Environnement Système", title_style="bold green")
    env_table.add_column("Composant", style="dim cyan")
    env_table.add_column("Status", style="bright_white")
    env_table.add_column("Détails", style="dim")
    
    # Conda
    conda_ok = check_conda_env()
    env_table.add_row(
        "Conda Environment", 
        "✅ OK" if conda_ok else "❌ MANQUANT",
        "trading_env actif" if conda_ok else "Exécuter: conda activate trading_env"
    )
    
    # Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    env_table.add_row("Python Version", f"✅ {python_version}", "Compatible")
    
    # Répertoire de travail
    current_dir = os.path.basename(os.getcwd())
    env_table.add_row("Working Directory", f"✅ {current_dir}", f"Path: {os.getcwd()}")
    
    console.print(env_table)
    console.print()
    
    # 2. État des données
    data_status = check_data_files()
    
    data_table = Table(title="📊 État des Données", title_style="bold cyan")
    data_table.add_column("Type", style="dim cyan")
    data_table.add_column("Fichiers", style="bright_white")
    data_table.add_column("Taille", style="green")
    data_table.add_column("Status", style="bright_white")
    
    # Données source
    source_status = "✅ COMPLET" if data_status['source']['files'] == data_status['source']['total'] else "⚠️ PARTIEL"
    data_table.add_row(
        "Données Source",
        f"{data_status['source']['files']}/{data_status['source']['total']}",
        f"{data_status['source']['size_mb']:.1f} MB",
        source_status
    )
    
    # Données traitées  
    processed_status = "✅ PRÊT" if data_status['processed']['files'] == data_status['processed']['total'] else "❌ MANQUANT"
    data_table.add_row(
        "Données Traitées",
        f"{data_status['processed']['files']}/{data_status['processed']['total']}",
        f"{data_status['processed']['size_mb']:.1f} MB", 
        processed_status
    )
    
    console.print(data_table)
    console.print()
    
    # 3. Configuration
    config_status = check_config_files()
    
    config_table = Table(title="⚙️ Configuration", title_style="bold yellow")
    config_table.add_column("Fichier", style="dim cyan")
    config_table.add_column("Status", style="bright_white")
    config_table.add_column("Taille", style="dim")
    
    for config_name, info in config_status.items():
        if info['exists']:
            status = "✅ OK"
            size = f"{info['size']} bytes"
        else:
            status = "❌ MANQUANT"
            size = "-"
        
        config_table.add_row(config_name, status, size)
    
    console.print(config_table)
    console.print()
    
    # 4. Modèles entraînés
    models = check_models()
    
    if models:
        models_table = Table(title="🤖 Modèles Entraînés", title_style="bold magenta")
        models_table.add_column("Modèle", style="dim cyan")
        models_table.add_column("Taille", style="bright_white")
        models_table.add_column("Âge", style="green")
        models_table.add_column("Dernière Modif", style="dim")
        
        for model_name, info in sorted(models.items(), key=lambda x: x[1]['modified'], reverse=True):
            age_text = f"{info['age_hours']:.1f}h" if info['age_hours'] < 24 else f"{info['age_hours']/24:.1f}j"
            models_table.add_row(
                model_name,
                f"{info['size_mb']:.1f} MB",
                age_text,
                info['modified'].strftime('%H:%M')
            )
        
        console.print(models_table)
    else:
        console.print(Panel("[yellow]❌ Aucun modèle entraîné trouvé[/yellow]", title="🤖 Modèles"))
    
    console.print()
    
    # 5. Analyse du dernier entraînement
    training_analysis = analyze_last_training()
    
    if training_analysis:
        perf_table = Table(title="📈 Dernière Performance", title_style="bold green")
        perf_table.add_column("Métrique", style="dim cyan")
        perf_table.add_column("Valeur", style="bright_white")
        perf_table.add_column("Évaluation", style="green")
        
        if 'final_capital' in training_analysis:
            capital = training_analysis['final_capital']
            if capital >= 20:
                eval_text = "🏆 EXCELLENT"
            elif capital >= 15:
                eval_text = "✅ BON"
            elif capital >= 10:
                eval_text = "⚠️ ACCEPTABLE"
            else:
                eval_text = "❌ FAIBLE"
            
            perf_table.add_row("Capital Final", f"${capital:.2f}", eval_text)
        
        if 'roi_pct' in training_analysis:
            roi = training_analysis['roi_pct']
            if roi >= 20:
                eval_text = "🚀 EXCEPTIONNEL"
            elif roi >= 0:
                eval_text = "📈 POSITIF"
            elif roi >= -20:
                eval_text = "⚠️ TOLÉRABLE"
            else:
                eval_text = "📉 CRITIQUE"
            
            perf_table.add_row("ROI", f"{roi:+.2f}%", eval_text)
        
        if 'training_size' in training_analysis:
            perf_table.add_row("Taille Entraînement", training_analysis['training_size'], "ℹ️ INFO")
        
        if 'log_age_hours' in training_analysis:
            age_text = f"{training_analysis['log_age_hours']:.1f}h ago"
            perf_table.add_row("Dernière Session", age_text, "⏰ RÉCENT" if training_analysis['log_age_hours'] < 2 else "📅 ANCIEN")
        
        console.print(perf_table)
    else:
        console.print(Panel("[yellow]❌ Aucune donnée d'entraînement récente trouvée[/yellow]", title="📈 Performance"))
    
    console.print()
    
    # 6. Recommandations
    recommendations = []
    
    # Analyser les problèmes et générer des recommandations
    if not conda_ok:
        recommendations.append("🔧 Activer l'environnement conda: conda activate trading_env")
    
    if data_status['processed']['files'] < data_status['processed']['total']:
        recommendations.append("📊 Générer les données manquantes: python scripts/convert_real_data.py --exec_profile cpu")
    
    if not models:
        recommendations.append("🚀 Démarrer un premier entraînement: python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 5000 --initial_capital 15.0")
    
    if training_analysis and 'final_capital' in training_analysis and training_analysis['final_capital'] < 10:
        recommendations.append("⚠️ Performances faibles - Augmenter le nombre de timesteps ou ajuster les paramètres")
    
    if not recommendations:
        recommendations.append("✅ Système opérationnel - Prêt pour l'entraînement de production!")
    
    # Afficher les recommandations
    if recommendations:
        rec_panel = Panel(
            "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations)),
            title="💡 Recommandations",
            border_style="green"
        )
        console.print(rec_panel)
    
    console.print()
    
    # 7. Commandes rapides
    quick_commands = [
        ("Entraînement Rapide", "python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 5000 --initial_capital 15.0"),
        ("Entraînement Standard", "python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 50000 --initial_capital 15.0"),
        ("Évaluation Modèle", "python scripts/quick_eval.py --model_path models/final_model.zip --profile cpu --capital 15.0"),
        ("Lancement Automatisé", "python run_adan.py --mode auto --quick --capital 15.0"),
        ("Monitoring Temps Réel", "python scripts/monitor_training.py")
    ]
    
    commands_table = Table(title="⚡ Commandes Rapides", title_style="bold blue")
    commands_table.add_column("Action", style="dim cyan")
    commands_table.add_column("Commande", style="bright_white")
    
    for action, command in quick_commands:
        commands_table.add_row(action, f"[dim]{command}[/dim]")
    
    console.print(commands_table)
    
    # Footer
    console.print()
    console.print(Rule("[dim]ADAN Trading Agent - Système de Trading Automatisé avec IA[/dim]", style="dim"))

def main():
    try:
        create_status_report()
        return 0
    except Exception as e:
        console.print(f"[red]❌ Erreur lors de la génération du rapport: {str(e)}[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())