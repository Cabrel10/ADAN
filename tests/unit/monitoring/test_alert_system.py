#!/usr/bin/env python3
"""
Tests unitaires pour le système d'alerte intelligent.
"""

import unittest
from unittest.mock import patch, MagicMock

from adan_trading_bot.monitoring.alert_system import (
    AlertSystem, AlertRule, Alert, AlertLevel, AlertChannel
)

class TestAlertSystem(unittest.TestCase):
    """Tests pour le système d'alerte."""

    def setUp(self):
        """Initialisation des tests."""
        # Configuration minimale pour les tests
        self.config = {
            'notification_channels': ['log'],
            'max_history': 100,
            'email': {'enabled': False},
            'slack': {'enabled': False},
            'webhook': {'enabled': False}
        }

        # Créer un mock pour le gestionnaire de notifications
        self.mock_notification_handler = MagicMock()
        self.mock_notification_handler.send_notification.return_value = True

        # Patcher le constructeur de NotificationHandler pour retourner notre mock
        self.notification_handler_patcher = patch(
            'adan_trading_bot.monitoring.alert_system.NotificationHandler',
            return_value=self.mock_notification_handler
        )
        self.notification_handler_patcher.start()

        # Initialiser le système d'alerte
        self.alert_system = AlertSystem(self.config)

        # Règle de test
        self.test_rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get('value', 0) > 10,
            message_template="La valeur est {value} (seuil: 10)",
            level=AlertLevel.WARNING,
            cooldown=60
        )

        # Règle critique
        self.critical_rule = AlertRule(
            name="critical_rule",
            condition=lambda ctx: ctx.get('error', False),
            message_template="Erreur critique: {error}",
            level=AlertLevel.CRITICAL
        )

        # Ajout des règles au système
        self.alert_system.add_rule(self.test_rule)
        self.alert_system.add_rule(self.critical_rule)

    def test_add_and_remove_rule(self):
        """Teste l'ajout et la suppression de règles."""
        # Vérifier que les règles sont bien ajoutées
        self.assertIn('test_rule', self.alert_system.rules)
        self.assertIn('critical_rule', self.alert_system.rules)

        # Supprimer une règle
        self.assertTrue(self.alert_system.remove_rule('test_rule'))
        self.assertNotIn('test_rule', self.alert_system.rules)

        # Essayer de supprimer une règle inexistante
        self.assertFalse(self.alert_system.remove_rule('nonexistent_rule'))

    def test_enable_disable_rule(self):
        """Teste l'activation et la désactivation de règles."""
        # Désactiver la règle
        self.assertTrue(self.alert_system.enable_rule('test_rule', False))
        self.assertFalse(self.alert_system.rules['test_rule'].enabled)

        # Réactiver la règle
        self.assertTrue(self.alert_system.enable_rule('test_rule', True))
        self.assertTrue(self.alert_system.rules['test_rule'].enabled)

        # Essayer avec une règle inexistante
        self.assertFalse(self.alert_system.enable_rule('nonexistent_rule', True))

    def test_evaluate_rules(self):
        """Teste l'évaluation des règles."""
        # Contexte qui ne déclenche aucune alerte
        context = {'value': 5}
        alerts = self.alert_system.evaluate_rules(context)
        self.assertEqual(len(alerts), 0)

        # Réinitialiser le mock
        self.mock_notification_handler.send_notification.reset_mock()

        # Contexte qui déclenche une alerte warning
        context = {'value': 15}
        alerts = self.alert_system.evaluate_rules(context)

        # Vérifier qu'une alerte a été générée
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)

        # Vérifier que l'alerte a été enregistrée dans l'historique
        self.assertEqual(len(self.alert_system.alert_history), 1)

        # Attendre que le thread de traitement ait fini de traiter l'alerte
        self.alert_system.alert_queue.join()

        # Vérifier que la notification a été envoyée via le gestionnaire
        self.mock_notification_handler.send_notification.assert_called_once()

        # Vérifier que la notification a été envoyée avec les bons paramètres
        call_args = self.mock_notification_handler.send_notification.call_args[0]
        self.assertEqual(len(call_args), 2)
        self.assertIsInstance(call_args[0], Alert)
        self.assertEqual(call_args[1], [AlertChannel.LOG])

    def tearDown(self):
        """Nettoyage après les tests."""
        if hasattr(self, 'alert_system'):
            self.alert_system.shutdown()

        # Arrêter le patcher
        if hasattr(self, 'notification_handler_patcher'):
            self.notification_handler_patcher.stop()

    def test_alert_cooldown(self):
        """Teste le cooldown des alertes."""
        # Première alerte
        context = {'value': 15}
        alerts = self.alert_system.evaluate_rules(context)
        self.assertEqual(len(alerts), 1)

        # Attendre que le thread de traitement ait fini de traiter l'alerte
        self.alert_system.alert_queue.join()

        # Vérifier que la notification a été envoyée
        self.mock_notification_handler.send_notification.assert_called_once()

        # Réinitialiser le mock
        self.mock_notification_handler.send_notification.reset_mock()

        # Même contexte avant la fin du cooldown - ne devrait pas déclencher d'alerte
        alerts = self.alert_system.evaluate_rules(context)
        self.assertEqual(len(alerts), 0)

        # Vérifier qu'aucune nouvelle notification n'a été envoyée
        self.mock_notification_handler.send_notification.assert_not_called()

        # Simuler la fin du cooldown
        self.alert_system.rules['test_rule'].last_triggered = time.time() - 61  # 61 secondes plus tard

        # Maintenant, l'alerte devrait être à nouveau déclenchée
        alerts = self.alert_system.evaluate_rules(context)
        self.assertEqual(len(alerts), 1)

        # Attendre que le thread de traitement ait fini de traiter la nouvelle alerte
        self.alert_system.alert_queue.join()

        # Vérifier que la notification a été envoyée à nouveau
        self.mock_notification_handler.send_notification.assert_called_once()

    def test_get_alerts(self):
        """Teste la récupération des alertes avec filtrage."""
        # Ajouter quelques alertes de test
        context1 = {'value': 15}
        context2 = {'error': 'Erreur de connexion'}

        self.alert_system.evaluate_rules(context1)  # Warning
        self.alert_system.evaluate_rules(context2)  # Critical

        # Toutes les alertes
        all_alerts = self.alert_system.get_alerts()
        self.assertEqual(len(all_alerts), 2)

        # Alertes critiques uniquement
        critical_alerts = self.alert_system.get_alerts(level=AlertLevel.CRITICAL)
        self.assertEqual(len(critical_alerts), 1)
        self.assertEqual(critical_alerts[0].level, AlertLevel.CRITICAL)

        # Acquitter une alerte
        alert_id = critical_alerts[0].id
        self.assertTrue(self.alert_system.acknowledge_alert(alert_id))

        # Vérifier le filtrage par statut d'acquittement
        acknowledged = self.alert_system.get_alerts(acknowledged=True)
        self.assertEqual(len(acknowledged), 1)
        self.assertEqual(acknowledged[0].id, alert_id)

        non_acknowledged = self.alert_system.get_alerts(acknowledged=False)
        self.assertEqual(len(non_acknowledged), 1)
        self.assertNotEqual(non_acknowledged[0].id, alert_id)

    @patch('adan_trading_bot.monitoring.alert_system.requests.post')
    def test_webhook_notification(self, mock_post):
        """Teste l'envoi de notifications via webhook."""
        print("\n=== Début du test_webhook_notification ===")

        # Configurer le mock pour la requête HTTP
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Créer une configuration avec webhook activé
        config = {
            'notification_channels': ['webhook'],
            'webhook': {
                'enabled': True,
                'url': 'http://example.com/webhook',
                'auth_token': 'test-token'
            },
            'log_level': 'DEBUG'
        }

        print("Configuration:", config)

        # Arrêter le patch du gestionnaire de notifications s'il est actif
        if hasattr(self, 'notification_handler_patcher'):
            self.notification_handler_patcher.stop()

        # Créer une instance de AlertSystem avec la configuration
        alert_system = AlertSystem(config)

        # Créer une alerte de test
        alert = Alert(
            id='test_alert',
            title='Test Alert',
            message='This is a test alert',
            level=AlertLevel.WARNING
        )

        # Envoyer la notification via webhook
        alert_system.notification_handler.send_notification(
            alert,
            [AlertChannel.WEBHOOK]
        )

        # Vérifier que la méthode a été appelée une fois
        mock_post.assert_called_once()

        # Vérifier les arguments de l'appel à requests.post
        args, kwargs = mock_post.call_args
        # L'URL est passée comme premier argument positionnel
        self.assertEqual(args[0], 'http://example.com/webhook')
        # Vérifier les en-têtes
        self.assertEqual(
            kwargs['headers']['Content-Type'],
            'application/json'
        )
        self.assertEqual(
            kwargs['headers']['Authorization'],
            'Bearer test-token'
        )
        # Vérifier le timeout
        self.assertEqual(kwargs['timeout'], 10)
        # Vérifier le contenu du JSON
        self.assertEqual(kwargs['json']['alert']['id'], 'test_alert')

        print("\n=== Fin du test_webhook_notification ===")

        # Réactiver le patch du gestionnaire de notifications pour les autres tests
        if hasattr(self, 'notification_handler_patcher'):
            self.notification_handler_patcher.start()

    def test_shutdown(self):
        """Teste l'arrêt propre du système d'alerte."""
        # Vérifier que le thread est en cours d'exécution
        self.assertTrue(self.alert_system.processing_thread.is_alive())

        # Arrêter le système
        self.alert_system.shutdown()

        # Vérifier que le thread s'est arrêté
        self.alert_system.processing_thread.join(timeout=1)
        self.assertFalse(self.alert_system.processing_thread.is_alive())

class TestAlertRule(unittest.TestCase):
    """Tests pour la classe AlertRule."""

    def test_condition_evaluation(self):
        """Teste l'évaluation des conditions des règles."""
        # Règle simple
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get('value', 0) > 10,
            message_template="Test",
            level=AlertLevel.INFO
        )

        # Tester la condition
        self.assertTrue(rule.condition({'value': 15}))
        self.assertFalse(rule.condition({'value': 5}))

    def test_message_formatting(self):
        """Teste le formatage des messages d'alerte."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            message_template="Valeur: {value}, Seuil: {threshold}",
            level=AlertLevel.INFO
        )

        # Tester le formatage du message
        context = {'value': 15, 'threshold': 10}
        self.assertEqual(
            rule.format_message(context),
            "Valeur: 15, Seuil: 10"
        )

    def test_cooldown(self):
        """Teste le système de cooldown des alertes."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            message_template="Test",
            level=AlertLevel.INFO,
            cooldown=60  # 1 minute
        )

        # Première évaluation - devrait déclencher
        rule.last_triggered = None
        self.assertTrue(rule.should_trigger({}))

        # Juste après - ne devrait pas déclencher (cooldown)
        self.assertFalse(rule.should_trigger({}))

        # Simuler le passage du temps (plus que le cooldown)
        rule.last_triggered = time.time() - 61
        self.assertTrue(rule.should_trigger({}))

if __name__ == '__main__':
    unittest.main()
