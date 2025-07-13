from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QDoubleSpinBox, QPushButton
from pathlib import Path
import yaml

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'dbe_config.yaml'
        self.current_config = self._load_config()

        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        # Paramètres de Risque
        self.sl_base_spin = QDoubleSpinBox()
        self.sl_base_spin.setDecimals(4)
        self.sl_base_spin.setSingleStep(0.001)
        self.sl_base_spin.setValue(self.current_config.get('risk_parameters', {}).get('base_sl_pct', 0.02))

        self.tp_base_spin = QDoubleSpinBox()
        self.tp_base_spin.setDecimals(4)
        self.tp_base_spin.setSingleStep(0.001)
        self.tp_base_spin.setValue(self.current_config.get('risk_parameters', {}).get('base_tp_pct', 0.04))

        self.drawdown_factor_spin = QDoubleSpinBox()
        self.drawdown_factor_spin.setSingleStep(0.1)
        self.drawdown_factor_spin.setValue(self.current_config.get('risk_parameters', {}).get('drawdown_sl_factor', 2.5))

        self.drawdown_threshold_spin = QDoubleSpinBox()
        self.drawdown_threshold_spin.setSingleStep(0.5)
        self.drawdown_threshold_spin.setValue(self.current_config.get('risk_parameters', {}).get('min_drawdown_threshold', 2.0))

        self.form_layout.addRow(QLabel("SL de Base (%):"), self.sl_base_spin)
        self.form_layout.addRow(QLabel("TP de Base (%):"), self.tp_base_spin)
        self.form_layout.addRow(QLabel("Sensibilité Drawdown:"), self.drawdown_factor_spin)
        self.form_layout.addRow(QLabel("Seuil Drawdown (%):"), self.drawdown_threshold_spin)

        self.save_button = QPushButton("Enregistrer Configuration")
        self.save_button.clicked.connect(self._save_config)
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.save_button)
        self.setLayout(self.layout)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config(self):
        self.current_config['risk_parameters'] = {
            'base_sl_pct': self.sl_base_spin.value(),
            'base_tp_pct': self.tp_base_spin.value(),
            'drawdown_sl_factor': self.drawdown_factor_spin.value(),
            'min_drawdown_threshold': self.drawdown_threshold_spin.value()
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.current_config, f, indent=2)
        print("Configuration DBE enregistrée.")
