#!/bin/bash
# Script d'exécution automatique pour finaliser Phase 2

echo "=========================================="
echo "FINALISATION PHASE 2 - CHECKPOINTS 2.5-2.6"
echo "=========================================="

# Checkpoint 2.5: Modification build_observation()
echo -e "\n🔧 CHECKPOINT 2.5: Modification build_observation()"
echo "Localisation: scripts/paper_trading_monitor.py lignes ~1040-1043"
echo ""
echo "ACTION MANUELLE REQUISE:"
echo "1. Ouvrir scripts/paper_trading_monitor.py"
echo "2. Localiser la normalisation manuelle (lignes 1040-1043):"
echo "   mean = window.mean(axis=0)"
echo "   std = window.std(axis=0)"
echo "   window_normalized = (window - mean) / std"
echo ""
echo "3. Remplacer par:"
echo "   # CORRECTION CRITIQUE: Utiliser VecNormalize"
echo "   env = self.worker_envs[worker_id]"
echo "   obs_dict = {'5m': ..., '1h': ..., '4h': ..., 'portfolio_state': ...}"
echo "   obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs_dict.items()}"
echo "   normalized_batch = env.normalize_obs(obs_batch)"
echo "   window_normalized = normalized_batch[tf][0]"
echo ""
echo "4. Ajouter worker_id à la signature:"
echo "   def build_observation(self, worker_id: str, df_5m, df_1h, df_4h):"
echo ""
echo "5. Mettre à jour tous les appels à build_observation()"
echo ""
echo "6. Compiler et valider:"
echo "   python -m py_compile scripts/paper_trading_monitor.py"
echo ""
read -p "Appuyez sur ENTRÉE quand la modification est terminée..."

# Vérifier la compilation
echo -e "\n✓ Vérification compilation..."
python -m py_compile scripts/paper_trading_monitor.py
if [ $? -eq 0 ]; then
    echo "✅ Compilation réussie"
else
    echo "❌ Erreur de compilation - Corriger avant de continuer"
    exit 1
fi

# Checkpoint 2.6: Validation divergence
echo -e "\n🧪 CHECKPOINT 2.6: Validation divergence post-correction"
echo "Exécution du test de divergence..."
python scripts/validate_normalization_coherence.py

if [ $? -eq 0 ]; then
    echo -e "\n=========================================="
    echo "✅ PHASE 2 COMPLÈTE"
    echo "=========================================="
    echo ""
    echo "Résultats:"
    echo "- Checkpoint 2.5: ✅ build_observation() modifiée"
    echo "- Checkpoint 2.6: ✅ Divergence validée"
    echo ""
    echo "Prochaine étape: Phase 3 - Validation Fonctionnelle"
    echo "Exécuter: python scripts/test_inference_basic.py"
else
    echo -e "\n=========================================="
    echo "❌ PHASE 2 ÉCHOUÉE"
    echo "=========================================="
    echo "Revoir Checkpoint 2.5 - Normalisation non correcte"
    exit 1
fi
