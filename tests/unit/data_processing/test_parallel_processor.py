"""
Tests unitaires pour le module parallel_processor.py
"""

import os
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from adan_trading_bot.data_processing.parallel_processor import (
    ParallelProcessor,
    parallel_apply,
    batch_process
)


class TestParallelProcessor(unittest.TestCase):
    """Tests pour la classe ParallelProcessor"""

    def test_initialization(self):
        """Test de l'initialisation de ParallelProcessor"""
        # Test avec les valeurs par défaut
        pp = ParallelProcessor()
        self.assertIsNotNone(pp)
        self.assertEqual(pp.n_workers, os.cpu_count() or 1)
        self.assertEqual(pp.prefer, 'threads')
        self.assertEqual(pp.chunksize, 1)
        self.assertTrue(pp.show_progress)
        self.assertEqual(pp.description, 'Processing')

        # Test avec des paramètres personnalisés
        pp = ParallelProcessor(
            n_workers=2,
            prefer='processes',
            chunksize=5,
            show_progress=False,
            description='Test',
            some_other_param=42
        )
        self.assertEqual(pp.n_workers, 2)
        self.assertEqual(pp.prefer, 'processes')
        self.assertEqual(pp.chunksize, 5)
        self.assertFalse(pp.show_progress)
        self.assertEqual(pp.description, 'Test')
        self.assertEqual(pp.executor_kwargs.get('some_other_param'), 42)

    def test_chunked(self):
        """Test de la méthode _chunked"""
        pp = ParallelProcessor()

        # Test avec une taille de chunk de 2
        items = [1, 2, 3, 4, 5]
        chunks = list(pp._chunked(items, 2))
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

        # Test avec une taille de chunk plus grande que la liste
        chunks = list(pp._chunked(items, 10))
        self.assertEqual(chunks, [items])

        # Test avec une taille de chunk de 1
        chunks = list(pp._chunked(items, 1))
        self.assertEqual(chunks, [[1], [2], [3], [4], [5]])

    def test_process_chunk(self):
        """Test de la méthode _process_chunk"""
        pp = ParallelProcessor()

        # Fonction de test
        def test_func(x, y, z=0):
            return x + y + z

        # Test avec des tuples d'arguments
        chunk = [(1, 2), (3, 4), (5, 6)]
        results = pp._process_chunk(test_func, chunk, z=10)
        self.assertEqual(results, [13, 17, 21])

        # Test avec des arguments simples
        chunk = [1, 2, 3]
        results = pp._process_chunk(lambda x: x * 2, chunk)
        self.assertEqual(results, [2, 4, 6])

    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_map_threads(self, mock_executor):
        """Test de la méthode map avec des threads"""
        # Configurer le mock
        mock_future = MagicMock()
        mock_future.result.return_value = [1, 4, 9]
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

        # Tester avec des threads
        pp = ParallelProcessor(n_workers=2, prefer='threads')

        # Fonction de test
        def square(x):
            return x * x

        # Appeler map
        results = pp.map(square, [1, 2, 3])

        # Vérifier les résultats
        self.assertEqual(results, [1, 4, 9])

        # Vérifier que le bon nombre de workers a été utilisé
        mock_executor.assert_called_once_with(max_workers=2)

    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_map_processes(self, mock_executor):
        """Test de la méthode map avec des processus"""
        # Configurer le mock
        mock_future = MagicMock()
        mock_future.result.return_value = [1, 4, 9]
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

        # Tester avec des processus
        pp = ParallelProcessor(n_workers=3, prefer='processes')

        # Fonction de test
        def square(x):
            return x * x

        # Appeler map
        results = pp.map(square, [1, 2, 3])

        # Vérifier les résultats
        self.assertEqual(results, [1, 4, 9])

        # Vérifier que le bon nombre de workers a été utilisé
        mock_executor.assert_called_once_with(max_workers=3)

    def test_apply_dataframe(self):
        """Test de la méthode apply avec un DataFrame"""
        # Créer un DataFrame de test
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        # Tester sans group_by
        pp = ParallelProcessor(n_workers=2, show_progress=False)

        def process_row(row):
            return row['A'] + row['B']

        results = pp.apply(process_row, df)
        self.assertEqual(results, [5, 7, 9])

        # Tester avec group_by
        df['group'] = ['X', 'X', 'Y']

        def process_group(group):
            return group['A'].sum() + group['B'].sum()

        results = pp.apply(process_group, df, group_by='group')
        self.assertEqual(set(results.keys()), {'X', 'Y'})
        self.assertEqual(results['X'], 12)  # (1+2) + (4+5)
        self.assertEqual(results['Y'], 9)   # 3 + 6

    def test_apply_dict(self):
        """Test de la méthode apply avec un dictionnaire"""
        # Créer un dictionnaire de test
        data = {'a': 1, 'b': 2, 'c': 3}

        # Tester avec une fonction simple
        pp = ParallelProcessor(n_workers=2, show_progress=False)

        def process_item(key_value):
            k, v = key_value
            return (k.upper(), v * 2)

        results = pp.apply(process_item, data)
        self.assertEqual(results, {'A': 2, 'B': 4, 'C': 6})

    def test_apply_list(self):
        """Test de la méthode apply avec une liste"""
        # Créer une liste de test
        items = [1, 2, 3, 4, 5]

        # Tester avec une fonction simple
        pp = ParallelProcessor(n_workers=2, show_progress=False)

        def square(x):
            return x * x

        results = pp.apply(square, items)
        self.assertEqual(results, [1, 4, 9, 16, 25])


class TestParallelApply(unittest.TestCase):
    """Tests pour la fonction parallel_apply"""

    def test_parallel_apply(self):
        """Test de la fonction parallel_apply"""
        # Créer un DataFrame de test
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        # Définir une fonction à appliquer
        def sum_columns(row):
            return row['A'] + row['B']

        # Appliquer en parallèle
        results = parallel_apply(
            sum_columns,
            df,
            n_workers=2,
            prefer='threads',
            show_progress=False
        )

        # Vérifier les résultats
        self.assertEqual(results, [11, 22, 33, 44, 55])


class TestBatchProcess(unittest.TestCase):
    """Tests pour la fonction batch_process"""

    def test_batch_process(self):
        """Test de la fonction batch_process"""
        # Créer une liste d'éléments à traiter
        items = list(range(10))

        # Définir une fonction de traitement
        def process_batch(batch):
            return [x * 2 for x in batch]

        # Traiter par lots
        results = batch_process(
            process_batch,
            items,
            batch_size=3,
            n_workers=2,
            show_progress=False
        )

        # Vérifier les résultats
        self.assertEqual(results, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        # Vérifier que les lots sont de la bonne taille
        self.assertEqual(len(results), len(items))

    def test_batch_process_empty(self):
        """Test de batch_process avec une liste vide"""
        results = batch_process(lambda x: x, [], batch_size=10)
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()
