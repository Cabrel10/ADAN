from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QPushButton, QHBoxLayout, QFormLayout, QDoubleSpinBox, QGroupBox, QLineEdit, QComboBox
from pathlib import Path
import yaml
from PySide6.QtCore import QProcess # Added QProcess

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration ADAN")
        self.setGeometry(200, 200, 600, 400)

        self.dbe_config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'dbe_config.yaml'
        self.dbe_config = self._load_dbe_config()

        self.main_config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'main_config.yaml'
        self.main_config = self._load_main_config()

        self.environment_config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'environment_config.yaml'
        self.environment_config = self._load_environment_config()

        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # Onglet Général
        self.general_tab = QWidget()
        self.general_tab_layout = QFormLayout(self.general_tab)

        self.symbol_line_edit = QLineEdit()
        self.symbol_line_edit.setText(self.main_config.get('symbol', 'BTC/USDT'))
        self.general_tab_layout.addRow(QLabel("Symbole:"), self.symbol_line_edit)

        self.timeframe_combo_box = QComboBox()
        self.timeframe_combo_box.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
        self.timeframe_combo_box.setCurrentText(self.main_config.get('timeframe', '1h'))
        self.general_tab_layout.addRow(QLabel("Timeframe:"), self.timeframe_combo_box)

        self.tab_widget.addTab(self.general_tab, "Général")

        # Onglet Risque
        self.risk_tab = QWidget()
        self.risk_tab_layout = QFormLayout(self.risk_tab)
        
        self.min_sl_spin = QDoubleSpinBox()
        self.min_sl_spin.setDecimals(4)
        self.min_sl_spin.setSingleStep(0.001)
        self.min_sl_spin.setValue(self.dbe_config.get('risk_parameters', {}).get('min_sl_pct', 0.005))
        self.risk_tab_layout.addRow(QLabel("SL Minimum (%):"), self.min_sl_spin)

        self.max_sl_spin = QDoubleSpinBox()
        self.max_sl_spin.setDecimals(4)
        self.max_sl_spin.setSingleStep(0.001)
        self.max_sl_spin.setValue(self.dbe_config.get('risk_parameters', {}).get('max_sl_pct', 0.10))
        self.risk_tab_layout.addRow(QLabel("SL Maximum (%):"), self.max_sl_spin)

        self.tab_widget.addTab(self.risk_tab, "Risque")

        # Onglet Récompense
        self.reward_tab = QWidget()
        self.reward_tab_layout = QFormLayout(self.reward_tab)

        self.winrate_threshold_spin = QDoubleSpinBox()
        self.winrate_threshold_spin.setDecimals(2)
        self.winrate_threshold_spin.setSingleStep(0.01)
        self.winrate_threshold_spin.setValue(self.dbe_config.get('reward', {}).get('winrate_threshold', 0.55))
        self.reward_tab_layout.addRow(QLabel("Seuil Winrate (%):"), self.winrate_threshold_spin)

        self.inaction_factor_spin = QDoubleSpinBox()
        self.inaction_factor_spin.setDecimals(2)
        self.inaction_factor_spin.setSingleStep(0.01)
        self.inaction_factor_spin.setValue(self.dbe_config.get('reward', {}).get('inaction_factor', 0.15))
        self.reward_tab_layout.addRow(QLabel("Facteur Inaction:"), self.inaction_factor_spin)

        self.min_action_frequency_spin = QDoubleSpinBox()
        self.min_action_frequency_spin.setDecimals(2)
        self.min_action_frequency_spin.setSingleStep(0.01)
        self.min_action_frequency_spin.setValue(self.dbe_config.get('reward', {}).get('min_action_frequency', 0.05))
        self.reward_tab_layout.addRow(QLabel("Fréquence Action Min (%):"), self.min_action_frequency_spin)

        self.tab_widget.addTab(self.reward_tab, "Récompense")

        # Onglet Hyperparamètres
        self.hyperparams_tab = QWidget()
        self.hyperparams_tab_layout = QFormLayout(self.hyperparams_tab)

        # Learning Rate Range
        self.lr_min_spin = QDoubleSpinBox()
        self.lr_min_spin.setDecimals(6)
        self.lr_min_spin.setSingleStep(0.00001)
        self.lr_min_spin.setValue(self.dbe_config.get('learning', {}).get('learning_rate_range', [1e-5, 1e-3])[0])
        self.hyperparams_tab_layout.addRow(QLabel("Learning Rate Min:"), self.lr_min_spin)

        self.lr_max_spin = QDoubleSpinBox()
        self.lr_max_spin.setDecimals(6)
        self.lr_max_spin.setSingleStep(0.00001)
        self.lr_max_spin.setValue(self.dbe_config.get('learning', {}).get('learning_rate_range', [1e-5, 1e-3])[1])
        self.hyperparams_tab_layout.addRow(QLabel("Learning Rate Max:"), self.lr_max_spin)

        # Entropy Coefficient Range
        self.ent_coef_min_spin = QDoubleSpinBox()
        self.ent_coef_min_spin.setDecimals(6)
        self.ent_coef_min_spin.setSingleStep(0.00001)
        self.ent_coef_min_spin.setValue(self.dbe_config.get('learning', {}).get('ent_coef_range', [0.001, 0.1])[0])
        self.hyperparams_tab_layout.addRow(QLabel("Ent. Coef. Min:"), self.ent_coef_min_spin)

        self.ent_coef_max_spin = QDoubleSpinBox()
        self.ent_coef_max_spin.setDecimals(6)
        self.ent_coef_max_spin.setSingleStep(0.00001)
        self.ent_coef_max_spin.setValue(self.dbe_config.get('learning', {}).get('ent_coef_range', [0.001, 0.1])[1])
        self.hyperparams_tab_layout.addRow(QLabel("Ent. Coef. Max:"), self.ent_coef_max_spin)

        # Gamma Range
        self.gamma_min_spin = QDoubleSpinBox()
        self.gamma_min_spin.setDecimals(6)
        self.gamma_min_spin.setSingleStep(0.00001)
        self.gamma_min_spin.setValue(self.dbe_config.get('learning', {}).get('gamma_range', [0.9, 0.999])[0])
        self.hyperparams_tab_layout.addRow(QLabel("Gamma Min:"), self.gamma_min_spin)

        self.gamma_max_spin = QDoubleSpinBox()
        self.gamma_max_spin.setDecimals(6)
        self.gamma_max_spin.setSingleStep(0.00001)
        self.gamma_max_spin.setValue(self.dbe_config.get('learning', {}).get('gamma_range', [0.9, 0.999])[1])
        self.hyperparams_tab_layout.addRow(QLabel("Gamma Max:"), self.gamma_max_spin)

        self.tab_widget.addTab(self.hyperparams_tab, "Hyperparamètres")

        # Onglet Régimes de Marché
        self.regimes_tab = QWidget()
        self.regimes_tab_layout = QVBoxLayout(self.regimes_tab)

        self.regime_fields = {}
        regimes = ["volatile", "sideways", "bull", "bear"]
        for regime in regimes:
            group_box = QGroupBox(regime.capitalize())
            form_layout = QFormLayout()
            
            sl_multiplier_spin = QDoubleSpinBox()
            sl_multiplier_spin.setDecimals(2)
            sl_multiplier_spin.setSingleStep(0.01)
            sl_multiplier_spin.setValue(self.dbe_config.get('modes', {}).get(regime, {}).get('sl_multiplier', 1.0))
            form_layout.addRow(QLabel("SL Multiplier:"), sl_multiplier_spin)

            tp_multiplier_spin = QDoubleSpinBox()
            tp_multiplier_spin.setDecimals(2)
            tp_multiplier_spin.setSingleStep(0.01)
            tp_multiplier_spin.setValue(self.dbe_config.get('modes', {}).get(regime, {}).get('tp_multiplier', 1.0))
            form_layout.addRow(QLabel("TP Multiplier:"), tp_multiplier_spin)

            position_size_multiplier_spin = QDoubleSpinBox()
            position_size_multiplier_spin.setDecimals(2)
            position_size_multiplier_spin.setSingleStep(0.01)
            position_size_multiplier_spin.setValue(self.dbe_config.get('modes', {}).get(regime, {}).get('position_size_multiplier', 1.0))
            form_layout.addRow(QLabel("Position Size Multiplier:"), position_size_multiplier_spin)

            group_box.setLayout(form_layout)
            self.regimes_tab_layout.addWidget(group_box)
            self.regime_fields[regime] = {
                'sl_multiplier': sl_multiplier_spin,
                'tp_multiplier': tp_multiplier_spin,
                'position_size_multiplier': position_size_multiplier_spin
            }

        self.tab_widget.addTab(self.regimes_tab, "Régimes de Marché")

        # Onglet Détection Régime
        self.regime_detection_tab = QWidget()
        self.regime_detection_tab_layout = QFormLayout(self.regime_detection_tab)

        self.vol_thresh_high_spin = QDoubleSpinBox()
        self.vol_thresh_high_spin.setDecimals(4)
        self.vol_thresh_high_spin.setSingleStep(0.001)
        self.vol_thresh_high_spin.setValue(self.environment_config.get('volatility_threshold_high', 0.03))
        self.regime_detection_tab_layout.addRow(QLabel("Seuil Volatilité Haute:"), self.vol_thresh_high_spin)

        self.vol_thresh_low_spin = QDoubleSpinBox()
        self.vol_thresh_low_spin.setDecimals(4)
        self.vol_thresh_low_spin.setSingleStep(0.001)
        self.vol_thresh_low_spin.setValue(self.environment_config.get('volatility_threshold_low', 0.01))
        self.regime_detection_tab_layout.addRow(QLabel("Seuil Volatilité Basse:"), self.vol_thresh_low_spin)

        self.adx_thresh_spin = QDoubleSpinBox()
        self.adx_thresh_spin.setDecimals(2)
        self.adx_thresh_spin.setSingleStep(1.0)
        self.adx_thresh_spin.setValue(self.environment_config.get('adx_threshold', 25.0))
        self.regime_detection_tab_layout.addRow(QLabel("Seuil ADX:"), self.adx_thresh_spin)

        self.ema_ratio_thresh_spin = QDoubleSpinBox()
        self.ema_ratio_thresh_spin.setDecimals(4)
        self.ema_ratio_thresh_spin.setSingleStep(0.001)
        self.ema_ratio_thresh_spin.setValue(self.environment_config.get('ema_ratio_threshold', 1.005))
        self.regime_detection_tab_layout.addRow(QLabel("Seuil Ratio EMA:"), self.ema_ratio_thresh_spin)

        self.atr_pct_thresh_spin = QDoubleSpinBox()
        self.atr_pct_thresh_spin.setDecimals(4)
        self.atr_pct_thresh_spin.setSingleStep(0.001)
        self.atr_pct_thresh_spin.setValue(self.environment_config.get('atr_pct_threshold', 0.02))
        self.regime_detection_tab_layout.addRow(QLabel("Seuil ATR (%):"), self.atr_pct_thresh_spin)

        self.tab_widget.addTab(self.regime_detection_tab, "Détection Régime")

        # Boutons Appliquer et Annuler
        button_layout = QHBoxLayout()
        self.apply_save_button = QPushButton("Appliquer & Enregistrer")
        self.cancel_button = QPushButton("Annuler")
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_save_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()

        self.layout.addLayout(button_layout)

        self.apply_save_button.clicked.connect(self._apply_and_save)
        self.cancel_button.clicked.connect(self.reject)

    def _load_dbe_config(self):
        if self.dbe_config_path.exists():
            with open(self.dbe_config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_main_config(self):
        if self.main_config_path.exists():
            with open(self.main_config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_environment_config(self):
        if self.environment_config_path.exists():
            with open(self.environment_config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _apply_and_save(self):
        # Update DBE config
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "risk_parameters", "--key", "min_sl_pct", "--value", str(self.min_sl_spin.value())])
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "risk_parameters", "--key", "max_sl_pct", "--value", str(self.max_sl_spin.value())])

        # Update Reward config
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "reward", "--key", "winrate_threshold", "--value", str(self.winrate_threshold_spin.value())])
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "reward", "--key", "inaction_factor", "--value", str(self.inaction_factor_spin.value())])
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "reward", "--key", "min_action_frequency", "--value", str(self.min_action_frequency_spin.value())])

        # Update Learning config
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "learning", "--key", "learning_rate_range", "--value", f"[{self.lr_min_spin.value()}, {self.lr_max_spin.value()}]"])
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "learning", "--key", "ent_coef_range", "--value", f"[{self.ent_coef_min_spin.value()}, {self.ent_coef_max_spin.value()}]"])
        self._run_cli_command(["update-config", "--type", "dbe", "--section", "learning", "--key", "gamma_range", "--value", f"[{self.gamma_min_spin.value()}, {self.gamma_max_spin.value()}]"])

        # Update Modes config
        for regime, fields in self.regime_fields.items():
            self._run_cli_command(["update-config", "--type", "dbe", "--section", f"modes.{regime}", "--key", "sl_multiplier", "--value", str(fields['sl_multiplier'].value())])
            self._run_cli_command(["update-config", "--type", "dbe", "--section", f"modes.{regime}", "--key", "tp_multiplier", "--value", str(fields['tp_multiplier'].value())])
            self._run_cli_command(["update-config", "--type", "dbe", "--section", f"modes.{regime}", "--key", "position_size_multiplier", "--value", str(fields['position_size_multiplier'].value())])

        # Update Main config
        self._run_cli_command(["update-config", "--type", "main", "--section", ".", "--key", "symbol", "--value", self.symbol_line_edit.text()])
        self._run_cli_command(["update-config", "--type", "main", "--section", ".", "--key", "timeframe", "--value", self.timeframe_combo_box.currentText()])

        # Update Environment config
        self._run_cli_command(["update-config", "--type", "environment", "--section", "environment_parameters", "--key", "volatility_threshold_high", "--value", str(self.vol_thresh_high_spin.value())])
        self._run_cli_command(["update-config", "--type", "environment", "--section", "environment_parameters", "--key", "volatility_threshold_low", "--value", str(self.vol_thresh_low_spin.value())])
        self._run_cli_command(["update-config", "--type", "environment", "--section", "environment_parameters", "--key", "adx_threshold", "--value", str(self.adx_thresh_spin.value())])
        self._run_cli_command(["update-config", "--type", "environment", "--section", "environment_parameters", "--key", "ema_ratio_threshold", "--value", str(self.ema_ratio_thresh_spin.value())])
        self._run_cli_command(["update-config", "--type", "environment", "--section", "environment_parameters", "--key", "atr_pct_threshold", "--value", str(self.atr_pct_thresh_spin.value())])

        print("Configuration mise à jour et enregistrée via CLI.")
        self.accept()

    def _run_cli_command(self, args):
        python_executable = str(Path("/home/morningstar/miniconda3/envs/trading_env/bin/python"))
        cli_script = str(Path(__file__).parent.parent.parent.parent / 'cli' / 'adan_cli.py')
        
        process = QProcess(self) # Use a local process for each command
        process.start(python_executable, [cli_script] + args)
        process.waitForFinished(-1) # Wait indefinitely for the process to finish
        stdout = process.readAllStandardOutput().data().decode().strip()
        stderr = process.readAllStandardError().data().decode().strip()
        
        if stdout: print(f"CLI STDOUT: {stdout}")
        if stderr: print(f"CLI STDERR: {stderr}")
        if process.exitCode() != 0: print(f"CLI command failed with exit code {process.exitCode()}")

        
