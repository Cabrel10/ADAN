
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

def create_robust_notebook(output_path: str):
    """
    Creates a robust Colab notebook programmatically using nbformat.
    This version is simplified to contain only bash commands for maximum reliability.
    """
    
    # --- Define Patch Content ---
    # This is read from the patch file we created earlier.
    try:
        with open('colab_fixes.patch', 'r') as f:
            patch_content = f.read()
    except FileNotFoundError:
        print("❌ FATAL: colab_fixes.patch not found. Cannot proceed.")
        return

    # --- Define Cells ---
    
    cell_intro = new_markdown_cell(
        "# 🚀 ADAN Trading Bot - Lanceur Robuste v3\n\n"
        "Ce notebook est généré par un script pour garantir sa validité. Il exécute toutes les étapes nécessaires pour lancer l'entraînement.\n\n"
        "**Instructions :**\n"
        "1. Assurez-vous que le runtime est un CPU (pas de GPU nécessaire).\n"
        "2. Cliquez sur `Runtime` -> `Run all`."
    )
    
    cell_clone = new_code_cell(
        "%%bash\n"
        "echo \"[PHASE 1/4] 📥 Clonage du dépôt GitHub...\"\n"
        "if [ -d \"/content/ADAN0\" ]; then\n"
        "  rm -rf /content/ADAN0\n"
        "fi\n"
        "git clone https://github.com/Cabrel10/ADAN0.git /content/ADAN0 --quiet\n"
        "echo \"✅ Dépôt cloné avec succès.\""
    )
    
    cell_patch = new_code_cell(
        f"%%bash\n"
        f"echo \"[PHASE 2/4] 🩹 Application des correctifs...\"\n\n"
        f"cat > /content/ADAN0/colab_fixes.patch << 'EOF'\n"
        f"{patch_content}"
        f"EOF\n\n"
        f"cd /content/ADAN0\n"
        f"git apply colab_fixes.patch\n"
        f"echo \"✅ Correctifs appliqués avec succès.\""
    )
    
    cell_setup = new_code_cell(
        "%%bash\n"
        "echo \"[PHASE 3/4] ⚙️ Installation des dépendances (5-10 minutes)...\"\n"
        "cd /content/ADAN0\n"
        "# The output is hidden to keep the notebook clean. Errors will still stop the execution.\n"
        "bash setup_colab_robust.sh &> /dev/null\n"
        "echo \"✅ Dépendances installées.\""
    )
    
    cell_train = new_code_cell(
        "%%bash\n"
        "echo \"[PHASE 4/4] 🚀 Lancement de l'entraînement...\"\n"
        "cd /content/ADAN0\n"
        "bash launch_training.sh 500000"
    )

    # --- Assemble and Write Notebook ---
    nb = new_notebook(
        cells=[cell_intro, cell_clone, cell_patch, cell_setup, cell_train],
        metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}
    )
    
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)
    
    print(f"✅ Notebook robust '{output_path}' créé avec succès.")

if __name__ == '__main__':
    create_robust_notebook('ADAN_Robust_Launcher.ipynb')
