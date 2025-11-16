#!/usr/bin/env python3
"""
Script final : combine les vrais portfolios avec les hyperparamètres corrects
"""

import json
from pathlib import Path

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      ✅ RÉSULTATS FINAUX CORRECTS W2 & W3                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Charger les données
    with open('true_portfolios_w2_w3.json', 'r') as f:
        portfolios = json.load(f)
    
    with open('optuna_hyperparams_extracted.json', 'r') as f:
        hyperparams = json.load(f)
    
    final_results = {}
    
    for worker in ['W2', 'W3']:
        if worker not in portfolios or worker not in hyperparams:
            continue
        
        print(f"🔍 {worker}:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        top_3_portfolios = portfolios[worker]['top_3']
        worker_hyperparams = hyperparams[worker]
        
        worker_results = {}
        
        for i, (trial_num, portfolio) in enumerate(top_3_portfolios, 1):
            gain = portfolio - 20.5
            roi = (gain / 20.5) * 100
            
            print(f"\n  {i}. Trial {trial_num}:")
            print(f"     Portfolio: ${portfolio:.2f}")
            print(f"     Gain: +${gain:.2f} (ROI: {roi:.1f}%)")
            
            if trial_num in worker_hyperparams:
                params = worker_hyperparams[trial_num]['hyperparams']
                print(f"     Hyperparamètres ({len(params)}):")
                
                # Grouper par catégorie
                ppo_params = {}
                trading_params = {}
                risk_params = {}
                reward_params = {}
                
                for key, val in params.items():
                    if any(x in key.lower() for x in ['learning', 'batch', 'n_steps', 'gamma', 'ent_coef', 'clip', 'epoch', 'vf_coef', 'max_grad', 'gae']):
                        ppo_params[key] = val
                    elif any(x in key.lower() for x in ['position', 'risk', 'patience', 'confidence']):
                        trading_params[key] = val
                    elif any(x in key.lower() for x in ['stop_loss', 'take_profit', 'hold', 'consecutive']):
                        risk_params[key] = val
                    else:
                        reward_params[key] = val
                
                if ppo_params:
                    print(f"       PPO ({len(ppo_params)}):")
                    for k, v in ppo_params.items():
                        print(f"         • {k}: {v}")
                
                if trading_params:
                    print(f"       Trading ({len(trading_params)}):")
                    for k, v in trading_params.items():
                        print(f"         • {k}: {v}")
                
                if risk_params:
                    print(f"       Risk ({len(risk_params)}):")
                    for k, v in risk_params.items():
                        print(f"         • {k}: {v}")
                
                if reward_params:
                    print(f"       Reward ({len(reward_params)}):")
                    for k, v in reward_params.items():
                        print(f"         • {k}: {v}")
                
                worker_results[trial_num] = {
                    'portfolio': portfolio,
                    'gain': gain,
                    'roi': roi,
                    'hyperparams': params
                }
        
        final_results[worker] = worker_results
        print()
    
    # Sauvegarder
    with open('FINAL_RESULTS_W2_W3_CORRECT.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("✅ Résultats finaux sauvegardés: FINAL_RESULTS_W2_W3_CORRECT.json")

if __name__ == "__main__":
    main()
