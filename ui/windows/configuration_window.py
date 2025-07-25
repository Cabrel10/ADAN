from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout, QCheckBox, QGroupBox, QScrollArea, QPushButton, QSlider, QDoubleSpinBox, QLabel, QLineEdit, QComboBox
from PySide6.QtCore import Qt

class ConfigurationWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.setMinimumSize(600, 400)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.create_tabs()

        self.expert_mode_checkbox = QCheckBox("Mode Expert")
        self.expert_mode_checkbox.stateChanged.connect(self.toggle_expert_mode)
        self.layout.addWidget(self.expert_mode_checkbox)

        self.toggle_expert_mode(self.expert_mode_checkbox.checkState())

    def create_tabs(self):
        # A2.1 Timeframes & Indicateurs
        self.timeframes_tab = QWidget()
        self.timeframes_layout = QVBoxLayout(self.timeframes_tab)
        self.tabs.addTab(self.timeframes_tab, "Timeframes & Indicateurs")

        self.create_timeframes_tab()

        # A2.2 DBE / Risk Engine
        self.dbe_tab = QWidget()
        self.dbe_layout = QVBoxLayout(self.dbe_tab)
        self.tabs.addTab(self.dbe_tab, "DBE / Risk Engine")

        self.create_dbe_tab()

        # A2.3 Algorithme PPO (Mode Expert)
        self.ppo_tab = QWidget()
        self.ppo_layout = QVBoxLayout(self.ppo_tab)
        self.tabs.addTab(self.ppo_tab, "Algorithme PPO (Expert)")

        self.create_ppo_tab()

        # A2.4 Orchestration Parallèle
        self.orchestration_tab = QWidget()
        self.orchestration_layout = QVBoxLayout(self.orchestration_tab)
        self.tabs.addTab(self.orchestration_tab, "Orchestration Parallèle")

        self.create_orchestration_tab()

    def create_timeframes_tab(self):
        timeframes_group = QGroupBox("Active Timeframes")
        timeframes_layout = QFormLayout(timeframes_group)

        self.tf_5m_checkbox = QCheckBox("5m")
        self.tf_1h_checkbox = QCheckBox("1h")
        self.tf_4h_checkbox = QCheckBox("4h")

        timeframes_layout.addRow(self.tf_5m_checkbox)
        timeframes_layout.addRow(self.tf_1h_checkbox)
        timeframes_layout.addRow(self.tf_4h_checkbox)

        self.timeframes_layout.addWidget(timeframes_group)

        indicators_group = QGroupBox("Indicators")
        indicators_layout = QVBoxLayout(indicators_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.indicators = [
            "SMA", "RSI", "MACD", "Bollinger Bands", "Stochastic Oscillator", "ATR", "Ichimoku Cloud",
            "VWAP", "OBV", "ADX", "CCI", "ROC", "Williams %R", "MFI", "Chaikin Money Flow",
            "Awesome Oscillator", "Ultimate Oscillator", "TRIX", "PPO", "KST", "DPO", "Coppock Curve"
        ]

        for indicator in self.indicators:
            indicator_layout = QVBoxLayout()
            checkbox = QCheckBox(indicator)
            config_button = QPushButton("Configure")
            indicator_layout.addWidget(checkbox)
            indicator_layout.addWidget(config_button)
            scroll_layout.addLayout(indicator_layout)

        scroll_area.setWidget(scroll_widget)
        indicators_layout.addWidget(scroll_area)
        self.timeframes_layout.addWidget(indicators_group)

    def create_dbe_tab(self):
        dbe_group = QGroupBox("DBE / Risk Engine")
        dbe_layout = QFormLayout(dbe_group)

        self.base_sl_spinbox = QDoubleSpinBox()
        self.base_sl_spinbox.setDecimals(4)
        self.base_sl_spinbox.setSingleStep(0.001)
        dbe_layout.addRow(QLabel("Base SL (%):"), self.base_sl_spinbox)

        self.drawdown_sl_factor_spinbox = QDoubleSpinBox()
        self.drawdown_sl_factor_spinbox.setSingleStep(0.1)
        dbe_layout.addRow(QLabel("Drawdown SL Factor:"), self.drawdown_sl_factor_spinbox)

        self.volatility_impact_slider = QSlider(Qt.Horizontal)
        dbe_layout.addRow(QLabel("Volatility Impact:"), self.volatility_impact_slider)

        self.dbe_layout.addWidget(dbe_group)

    def create_ppo_tab(self):
        ppo_group = QGroupBox("PPO Parameters")
        ppo_layout = QFormLayout(ppo_group)

        self.learning_rate_input = QLineEdit()
        ppo_layout.addRow(QLabel("Learning Rate:"), self.learning_rate_input)

        self.clip_range_input = QLineEdit()
        ppo_layout.addRow(QLabel("Clip Range:"), self.clip_range_input)

        self.batch_size_input = QLineEdit()
        ppo_layout.addRow(QLabel("Batch Size:"), self.batch_size_input)

        self.ent_coef_input = QLineEdit()
        ppo_layout.addRow(QLabel("Entropy Coefficient:"), self.ent_coef_input)

        self.vf_coef_input = QLineEdit()
        ppo_layout.addRow(QLabel("Value Function Coefficient:"), self.vf_coef_input)

        self.ppo_layout.addWidget(ppo_group)

    def create_orchestration_tab(self):
        orchestration_group = QGroupBox("Parallel Orchestration")
        orchestration_layout = QFormLayout(orchestration_group)

        self.num_instances_spinbox = QDoubleSpinBox()
        self.num_instances_spinbox.setDecimals(0)
        self.num_instances_spinbox.setMinimum(1)
        orchestration_layout.addRow(QLabel("Number of Instances:"), self.num_instances_spinbox)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Conservative", "Balanced", "Aggressive", "Adaptive"])
        orchestration_layout.addRow(QLabel("Profile:"), self.profile_combo)

        self.cpu_input = QLineEdit()
        orchestration_layout.addRow(QLabel("CPU Allocation:"), self.cpu_input)

        self.memory_input = QLineEdit()
        orchestration_layout.addRow(QLabel("Memory Allocation:"), self.memory_input)

        self.orchestration_layout.addWidget(orchestration_group)

    def toggle_expert_mode(self, state):
        self.tabs.setTabVisible(2, state == Qt.Checked)

