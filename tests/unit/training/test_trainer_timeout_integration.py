import time
import sys
import types
import importlib

import pytest


class _DummyEnv:
    def __init__(self):
        self.saved = False

    def save(self, path):
        self.saved = True

    def close(self):
        pass


def test_trainer_respects_timeout_and_saves_checkpoint(monkeypatch, tmp_path):
    # Stub ppo_agent before importing trainer to avoid importing real dependencies
    calls = {'learn': 0, 'save': 0}

    class DummyAgent:
        def learn(self, total_timesteps, callback):  # noqa: ARG002
            calls['learn'] += 1
            time.sleep(0.5)

        def save(self, path):
            calls['save'] += 1
            return path

    # Ensure parent package module exists
    agent_pkg = types.ModuleType('adan_trading_bot.agent')
    sys.modules['adan_trading_bot.agent'] = agent_pkg
    fake_ppo = types.ModuleType('adan_trading_bot.agent.ppo_agent')
    fake_ppo.create_ppo_agent = lambda **_: DummyAgent()
    sys.modules['adan_trading_bot.agent.ppo_agent'] = fake_ppo

    # Now import trainer (will use our stubbed ppo_agent)
    trainer = importlib.import_module('adan_trading_bot.training.trainer')
    # Patch load_config to a minimal config
    def fake_load_config(_):
        return {
            'total_timesteps': 1_000_000,
            'use_gpu': False,
            'environment': {
                'data': {
                    'data_dir': str(tmp_path),
                    'assets': ['BTC'],
                    'timeframes': ['1h'],
                    'features_per_timeframe': {'1h': ['close']},
                    'chunk_size': 8,
                }
            },
            'best_model_save_path': str(tmp_path / 'models/best'),
            'log_path': str(tmp_path / 'logs'),
            'tensorboard_log': str(tmp_path / 'tb'),
        }

    monkeypatch.setattr(trainer.utils, 'load_config', fake_load_config)

    # Patch create_envs to avoid building real envs
    def fake_create_envs(_config, _env_cfg):
        return _DummyEnv(), _DummyEnv()

    monkeypatch.setattr(trainer, 'create_envs', fake_create_envs)
    # Stub callbacks to avoid touching real environments
    class _DummyCb:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(trainer, 'EvalCallback', _DummyCb)
    monkeypatch.setattr(trainer, 'CheckpointCallback', _DummyCb)

    # Run with short timeout so it triggers
    agent = trainer.train_agent(
        config_path='unused.yaml',
        custom_config={'best_model_save_path': str(tmp_path / 'models/best')},
        callbacks=None,
        timeout=0.2,
    )

    # Ensure training attempted and save called due to timeout
    assert isinstance(agent, DummyAgent)
    assert calls['learn'] == 1
    assert calls['save'] == 1
