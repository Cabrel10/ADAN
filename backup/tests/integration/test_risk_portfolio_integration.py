import unittest

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.risk_management.risk_assessor import RiskAssessor


class TestRiskPortfolioIntegration(unittest.TestCase):
    """Test d'intégration entre le gestionnaire de risque et le portefeuille."""

    def setUp(self):
        # Configuration de base
        self.config = {
            "risk_management": {
                "max_position_size": 0.1,
                "max_portfolio_risk": 0.02,
                "var_confidence": 0.95,
                "var_horizon": 1,
            },
            "trading_rules": {
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "max_leverage": 10,
            },
            "assets": ["BTC", "ETH"],
            "commission": 0.001,
        }

        # Initialisation des composants
        self.risk_assessor = RiskAssessor(self.config)
        self.portfolio = PortfolioManager(self.config)

        # Données de marché simulées
        self.market_data = {
            "prices": {"BTC": 50000, "ETH": 3000},
            "volumes": {"BTC": 1000, "ETH": 5000},
            "spreads": {"BTC": 10, "ETH": 5},
        }

    def test_risk_assessment_after_trade(self):
        """Teste l'évaluation du risque après une transaction."""
        # Mise à jour des données de marché
        self.risk_assessor.update_market_data(self.market_data)

        # Ouvre une position dans le portefeuille
        self.portfolio.open_position("BTC", 50000, 1.0)

        # Vérifie que le risque est correctement évalué
        risk_metrics = self.risk_assessor.assess_portfolio_risk(
            {
                "total_value": self.portfolio.total_capital,
                "equity": self.portfolio.total_capital, # Changed from equity to total_capital
                "used_margin": self.portfolio.used_margin,
                "positions": {
                    "BTC": {
                        "size": 1.0,
                        "value": 50000,
                        "entry_price": 50000,
                        "current_price": 50000,
                        "pnl": 0.0,
                    }
                },
            }
        )

        # Vérifications de base
        self.assertIn("total_value", risk_metrics)
        self.assertIn("equity", risk_metrics)
        self.assertIn("leverage", risk_metrics)
        self.assertIn("drawdown", risk_metrics)
        self.assertIn("risk_level", risk_metrics)

    def test_position_sizing_with_risk(self):
        """Teste le calcul de la taille de position en fonction du risque."""
        # Définit la valeur du portefeuille
        self.portfolio.portfolio_value = 100000

        # Calcule la taille de position optimale
        position_size = self.risk_assessor.calculate_position_size(
            "BTC", 50000, 49000, 1.0
        )

        # Vérifie que la taille est raisonnable
        self.assertGreater(position_size, 0.0)
        max_allowed = (
            self.portfolio.portfolio_value * 0.1
        )  # max_position_size = 10%
        self.assertLessEqual(position_size * 50000, max_allowed)

    def test_stop_loss_trigger(self):
        """Teste le déclenchement du stop-loss."""
        # Ouvre une position avec un stop-loss
        self.portfolio.open_position("BTC", 50000, 1.0) # Removed stop_loss argument

        # Vérifie le déclenchement du stop-loss
        should_close = self.risk_assessor.check_stop_loss("BTC", 48000)
        self.assertTrue(should_close)

    def test_risk_limits(self):
        """Teste la mise à jour des limites de risque."""
        # Teste la mise à jour des limites de risque
        new_limits = {"max_position_size": 0.2, "max_portfolio_risk": 0.05}
        self.risk_assessor.update_risk_parameters(new_limits)

        # Vérifie que les limites ont été mises à jour
        limits = self.risk_assessor.get_risk_limits()
        self.assertEqual(limits["max_position_size"], 0.2)
        self.assertEqual(limits["max_portfolio_risk"], 0.05)


if __name__ == "__main__":
    unittest.main()
