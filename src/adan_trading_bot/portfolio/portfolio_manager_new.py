    def get_active_tier(self):
        """
        Détermine le palier de capital actif en fonction de la valeur du portefeuille.
        
        Returns:
            dict: Le palier de capital actif avec toutes ses propriétés.
            
        Raises:
            RuntimeError: Si aucun palier valide n'est trouvé.
        """
        if not self.capital_tiers:
            raise RuntimeError("Aucun palier de capital n'est défini dans la configuration.")
            
        current_value = self.get_portfolio_value()
        
        # Parcourir les paliers du plus élevé au plus bas
        for tier in sorted(self.capital_tiers, key=lambda x: x['min_capital'], reverse=True):
            if current_value >= tier['min_capital']:
                logger.debug(
                    "Palier actif: %s (capital: %.2f >= %.2f)",
                    tier['name'], current_value, tier['min_capital']
                )
                return tier
        
        # Si on arrive ici, utiliser le palier le plus bas
        min_tier = min(self.capital_tiers, key=lambda x: x['min_capital'])
        logger.warning(
            "La valeur du portefeuille (%.2f) est inférieure au palier minimum (%.2f). "
            "Utilisation du palier: %s",
            current_value, min_tier['min_capital'], min_tier['name']
        )
        return min_tier
