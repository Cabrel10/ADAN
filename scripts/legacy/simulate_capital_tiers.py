import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration de la simulation
class SimulationConfig:
    INITIAL_CAPITAL = 50.0  # Commence dans le Tier 2 (30-100)
    NUM_DAYS = 90  # Durée de la simulation
    DAILY_RETURN_MEAN = 0.001  # Rendement journalier moyen (0.1%)
    DAILY_RETURN_STD = 0.02  # Volatilité journalière (2%)
    TRADING_DAYS_PER_YEAR = 252
    
    # Configuration des capital tiers (identique à celle des tests)
    CAPITAL_TIERS = [
        {'name': 'Micro Capital', 'min_capital': 0, 'max_capital': 30, 
         'max_position_size_pct': 90.0, 'leverage': 1.0, 
         'risk_per_trade_pct': 2.0, 'max_drawdown_pct': 15.0},
        {'name': 'Small Capital', 'min_capital': 30, 'max_capital': 100, 
         'max_position_size_pct': 70.0, 'leverage': 1.0, 
         'risk_per_trade_pct': 1.5, 'max_drawdown_pct': 20.0},
        {'name': 'Medium Capital', 'min_capital': 100, 'max_capital': 300, 
         'max_position_size_pct': 60.0, 'leverage': 1.0, 
         'risk_per_trade_pct': 1.0, 'max_drawdown_pct': 25.0},
        {'name': 'High Capital', 'min_capital': 300, 'max_capital': 1000, 
         'max_position_size_pct': 35.0, 'leverage': 1.0, 
         'risk_per_trade_pct': 0.75, 'max_drawdown_pct': 30.0},
        {'name': 'Enterprise', 'min_capital': 1000, 'max_capital': None, 
         'max_position_size_pct': 20.0, 'leverage': 1.0, 
         'risk_per_trade_pct': 0.5, 'max_drawdown_pct': 35.0}
    ]

class PortfolioSimulator:
    def __init__(self, config):
        self.config = config
        self.current_capital = config.INITIAL_CAPITAL
        self.initial_equity = config.INITIAL_CAPITAL
        self.portfolio_value = config.INITIAL_CAPITAL
        self.current_tier = None
        self.history = []
        self.position_size = 0.0
        self.position_value = 0.0
        self.asset_price = 50000.0  # Prix initial de l'actif
        self.update_current_tier()
    
    def update_current_tier(self):
        """Met à jour le tier actuel en fonction du capital"""
        for tier in reversed(self.config.CAPITAL_TIERS):
            if (self.portfolio_value >= tier['min_capital'] and 
                (tier['max_capital'] is None or self.portfolio_value < tier['max_capital'])):
                self.current_tier = tier
                return

    def simulate_day(self, daily_return):
        """Simule une journée de trading"""
        # Mise à jour du prix de l'actif
        previous_price = self.asset_price
        self.asset_price *= (1 + daily_return)
        
        # Mise à jour de la valeur de la position
        if self.position_size > 0:
            self.position_value = self.position_size * self.asset_price
            self.portfolio_value = self.current_capital + self.position_value
        
        # Vérification du drawdown
        self.check_drawdown()
        
        # Décision de trading aléatoire (pour la simulation)
        self.make_trading_decision()
        
        # Enregistrement des métriques
        self.record_metrics(daily_return)
        
        # Mise à jour du tier si nécessaire
        self.update_current_tier()
    
    def check_drawdown(self):
        """Vérifie si le drawdown dépasse la limite du tier actuel"""
        if not self.current_tier:
            return
            
        peak = max(self.initial_equity, self.portfolio_value)
        if peak == 0:
            return
            
        drawdown = (peak - self.portfolio_value) / peak * 100
        max_drawdown = self.current_tier['max_drawdown_pct']
        
        if drawdown > max_drawdown:
            # Liquidation
            self.liquidate(f"Drawdown de {drawdown:.2f}% dépasse la limite de {max_drawdown}%")
    
    def liquidate(self, reason):
        """Liquide la position"""
        self.current_capital = self.portfolio_value
        self.position_size = 0.0
        self.position_value = 0.0
        self.initial_equity = self.current_capital  # Reset peak equity
        print(f"\nLIQUIDATION - {reason}. Nouveau capital: {self.current_capital:.2f} $")
    
    def make_trading_decision(self):
        """Prend une décision de trading aléatoire pour la simulation"""
        if not self.current_tier:
            return
            
        # Décision aléatoire: 30% de chance d'acheter, 30% de vendre, 40% de ne rien faire
        decision = np.random.choice(["buy", "sell", "hold"], p=[0.3, 0.3, 0.4])
        
        if decision == "buy" and self.position_size == 0:
            # Calcul de la taille de position basée sur le risque
            risk_amount = self.portfolio_value * (self.current_tier['risk_per_trade_pct'] / 100)
            position_size = risk_amount / (self.asset_price * 0.02)  # Stop-loss à 2%
            
            # Limite de taille de position
            max_position_value = self.portfolio_value * (self.current_tier['max_position_size_pct'] / 100)
            max_position_size = max_position_value / self.asset_price
            position_size = min(position_size, max_position_size)
            
            # Vérification que nous avons assez de capital
            if position_size * self.asset_price <= self.current_capital:
                self.position_size = position_size
                self.position_value = position_size * self.asset_price
                self.current_capital -= self.position_value
                
        elif decision == "sell" and self.position_size > 0:
            # Vente de la position
            self.current_capital += self.position_value
            self.position_size = 0.0
            self.position_value = 0.0
    
    def record_metrics(self, daily_return):
        """Enregistre les métriques pour analyse"""
        self.history.append({
            'date': len(self.history),
            'portfolio_value': self.portfolio_value,
            'cash': self.current_capital,
            'position_value': self.position_value,
            'asset_price': self.asset_price,
            'tier': self.current_tier['name'] if self.current_tier else 'None',
            'daily_return': daily_return
        })

def run_simulation():
    # Configuration
    config = SimulationConfig()
    
    # Initialisation du simulateur
    simulator = PortfolioSimulator(config)
    
    print(f"Démarrage de la simulation avec un capital initial de ${config.INITIAL_CAPITAL:.2f}")
    print(f"Durée: {config.NUM_DAYS} jours de trading")
    print("-" * 50)
    
    # Simulation
    for day in tqdm(range(config.NUM_DAYS), desc="Simulation en cours"):
        # Génération d'un rendement aléatoire
        daily_return = np.random.normal(
            config.DAILY_RETURN_MEAN, 
            config.DAILY_RETURN_STD
        )
        
        # Simulation d'une journée
        simulator.simulate_day(daily_return)
    
    # Conversion en DataFrame pour l'analyse
    df = pd.DataFrame(simulator.history)
    
    # Affichage des résultats
    print("\n" + "="*50)
    print(f"Résultats finaux après {config.NUM_DAYS} jours")
    print(f"Valeur finale du portefeuille: ${simulator.portfolio_value:.2f}")
    print(f"Rendement total: {((simulator.portfolio_value / config.INITIAL_CAPITAL) - 1) * 100:.2f}%")
    print(f"Dernier tier: {simulator.current_tier['name'] if simulator.current_tier else 'Aucun'}")
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    
    # Graphique de la valeur du portefeuille
    plt.subplot(2, 1, 1)
    plt.plot(df['date'], df['portfolio_value'], label='Valeur du portefeuille')
    plt.title('Évolution de la valeur du portefeuille')
    plt.xlabel('Jours')
    plt.ylabel('Valeur ($)')
    plt.grid(True)
    
    # Ajout des zones de tiers
    for tier in config.CAPITAL_TIERS:
        plt.axhspan(tier['min_capital'], 
                   tier['max_capital'] if tier['max_capital'] else df['portfolio_value'].max() * 1.1,
                   alpha=0.1, label=f"Tier: {tier['name']}")
    
    plt.legend()
    
    # Graphique du tier actuel
    plt.subplot(2, 1, 2)
    tier_mapping = {tier['name']: i for i, tier in enumerate(config.CAPITAL_TIERS)}
    df['tier_num'] = df['tier'].map(tier_mapping)
    plt.step(df['date'], df['tier_num'], where='post')
    plt.yticks(ticks=range(len(tier_mapping)), labels=tier_mapping.keys())
    plt.title('Tier de capital actuel')
    plt.xlabel('Jours')
    plt.ylabel('Tier')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Sauvegarde des graphiques
    plt.savefig('simulation_results.png')
    print("\nGraphiques sauvegardés dans 'simulation_results.png'")
    
    # Affichage des transitions de tier
    tier_changes = df[df['tier'] != df['tier'].shift(1)]
    if not tier_changes.empty:
        print("\nTransitions de tier:")
        for _, row in tier_changes.iterrows():
            print(f"Jour {int(row['date'])}: Passage au tier {row['tier']} (Valeur: ${row['portfolio_value']:.2f})")
    
    return df

if __name__ == "__main__":
    run_simulation()
