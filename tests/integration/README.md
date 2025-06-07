# Tests d'Intégration

Ce répertoire contient les tests d'intégration pour vérifier l'interaction entre les différents modules du projet ADAN.

## Contenu

* `test_training_pipeline.py`: Tests pour le pipeline complet d'entraînement, de l'environnement à l'agent

## Objectif

Contrairement aux tests unitaires qui vérifient les composants individuels, les tests d'intégration s'assurent que les différents modules fonctionnent correctement ensemble. Ils permettent de:

1. Détecter les problèmes d'interface entre les composants
2. Vérifier les flux de données à travers le système
3. Tester les scénarios d'utilisation réels de bout en bout
4. Identifier les problèmes de performance ou de ressources

## Approche

Les tests d'intégration utilisent généralement:
- Des environnements simplifiés mais réalistes
- Des jeux de données réduits mais représentatifs
- Des configurations similaires à celles de production
- Des assertions sur les résultats finaux plutôt que sur les étapes intermédiaires

## Bonnes Pratiques

1. Gardez les tests d'intégration ciblés sur des scénarios spécifiques
2. Utilisez des fixtures pour préparer l'environnement de test
3. Nettoyez les ressources après chaque test (fichiers temporaires, etc.)
4. Documentez clairement ce que chaque test vérifie
5. Exécutez ces tests régulièrement, mais moins fréquemment que les tests unitaires en raison de leur durée d'exécution plus longue
