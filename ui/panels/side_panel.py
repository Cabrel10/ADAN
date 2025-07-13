from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QGroupBox, QFormLayout, QTextEdit, QCheckBox
from PySide6.QtCore import Qt, Signal, QDateTime # Added QDateTime

class SidePanel(QWidget):
    chart_series_toggled = Signal(str, bool)
    backtest_speed_changed = Signal(int) # New signal for backtest speed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Monitoring Group
        self.monitoring_group = QGroupBox("Monitoring")
        self.monitoring_layout = QFormLayout(self.monitoring_group)
        self.portfolio_value_label = QLabel("N/A")
        self.drawdown_label = QLabel("N/A")
        self.risk_mode_label = QLabel("N/A")
        self.monitoring_layout.addRow("Valeur Portefeuille:", self.portfolio_value_label)
        self.monitoring_layout.addRow("Drawdown:", self.drawdown_label)
        self.monitoring_layout.addRow("Mode Risque:", self.risk_mode_label)
        self.layout.addWidget(self.monitoring_group)

        # Chart Series Group
        self.chart_series_group = QGroupBox("Séries Graphique")
        self.chart_series_layout = QVBoxLayout(self.chart_series_group)

        self.candles_checkbox = QCheckBox("Candles")
        self.candles_checkbox.setChecked(True)
        self.candles_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("candles", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.candles_checkbox)

        self.volume_checkbox = QCheckBox("Volume")
        self.volume_checkbox.setChecked(True)
        self.volume_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("volume", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.volume_checkbox)

        self.rsi_checkbox = QCheckBox("RSI")
        self.rsi_checkbox.setChecked(True)
        self.rsi_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("rsi", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.rsi_checkbox)

        self.macd_checkbox = QCheckBox("MACD")
        self.macd_checkbox.setChecked(True)
        self.macd_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("macd", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.macd_checkbox)

        self.bollinger_checkbox = QCheckBox("Bollinger Bands")
        self.bollinger_checkbox.setChecked(True)
        self.bollinger_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("bollinger", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.bollinger_checkbox)

        self.layout.addWidget(self.chart_series_group)

        # Logs Group
        self.logs_group = QGroupBox("Logs")
        self.logs_layout = QVBoxLayout(self.logs_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.document().setMaximumBlockCount(1000) # Limit log lines
        self.logs_layout.addWidget(self.log_display)
        self.layout.addWidget(self.logs_group)

        # Backtest Controls Group
        self.backtest_group = QGroupBox("Contrôles Backtest")
        self.backtest_layout = QFormLayout(self.backtest_group)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.valueChanged.connect(self.backtest_speed_changed.emit) # Connect slider to signal
        self.backtest_layout.addRow("Vitesse Simulation:", self.speed_slider)
        self.layout.addWidget(self.backtest_group)

        self.layout.addStretch(1)

    def update_monitoring(self, portfolio_value, drawdown, risk_mode):
        self.portfolio_value_label.setText(f"{portfolio_value:.2f}")
        self.drawdown_label.setText(f"{drawdown:.2%}")
        self.risk_mode_label.setText(risk_mode)

    def append_log(self, message):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_display.append(f"[{timestamp}] {message}")
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum()) # Auto-scroll