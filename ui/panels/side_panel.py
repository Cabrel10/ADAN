from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, 
                               QGroupBox, QFormLayout, QTextEdit, QCheckBox, 
                               QTableWidget, QTableWidgetItem, QPushButton, QComboBox, QHBoxLayout, QProgressBar, QFileDialog)
from PySide6.QtCore import Qt, Signal, QDateTime
from PySide6.QtGui import QColor
import pyqtgraph as pg

class SidePanel(QWidget):
    chart_series_toggled = Signal(str, bool)
    backtest_speed_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.all_logs = []

        # Monitoring Group
        self.monitoring_group = QGroupBox(self.tr("Monitoring"))
        self.monitoring_layout = QFormLayout(self.monitoring_group)
        self.portfolio_value_label = QLabel("N/A")
        self.pnl_label = QLabel("N/A") # Added for PnL display
        self.drawdown_label = QLabel("N/A")
        self.drawdown_gauge = QProgressBar() # Added for drawdown visualization
        self.drawdown_gauge.setRange(0, 100) # Percentage
        self.drawdown_gauge.setTextVisible(True)
        
        self.risk_mode_indicator = QLabel()
        self.risk_mode_indicator.setFixedSize(20, 20)
        self.risk_mode_indicator.setStyleSheet("border-radius: 10px; background-color: gray;")
        self.risk_mode_label = QLabel("N/A") # Keep the label for text
        
        risk_mode_layout = QHBoxLayout()
        risk_mode_layout.addWidget(self.risk_mode_indicator)
        risk_mode_layout.addWidget(self.risk_mode_label)
        
        self.monitoring_layout.addRow(self.tr("Valeur Portefeuille:"), self.portfolio_value_label)
        self.monitoring_layout.addRow(self.tr("P&L:"), self.pnl_label) # Added P&L row
        self.monitoring_layout.addRow(self.tr("Drawdown:"), self.drawdown_gauge)
        self.monitoring_layout.addRow(self.tr("Mode Risque:"), risk_mode_layout)

        # A1.2 Métriques Portfolio - Positions ouvertes
        self.open_positions_table = QTableWidget()
        self.open_positions_table.setColumnCount(4)
        self.open_positions_table.setHorizontalHeaderLabels([
            self.tr("Symbole"), self.tr("Quantité"), self.tr("Prix Entrée"), self.tr("Prix Actuel")
        ])
        self.open_positions_table.horizontalHeader().setStretchLastSection(True)
        self.monitoring_layout.addRow(self.tr("Positions Ouvertes:"), self.open_positions_table)

        # A1.2 Métriques Portfolio - Exposure % par paire
        self.exposure_chart = pg.PlotWidget()
        self.exposure_chart.setFixedHeight(150)
        self.monitoring_layout.addRow(self.tr("Exposition par Paire:"), self.exposure_chart)

        # A1.3 Statut DBE - SL% et TP% en temps réel
        self.sl_pct_label = QLabel("N/A")
        self.tp_pct_label = QLabel("N/A")
        self.monitoring_layout.addRow(self.tr("SL Actuel (%):"), self.sl_pct_label)
        self.monitoring_layout.addRow(self.tr("TP Actuel (%):"), self.tp_pct_label)

        self.layout.addWidget(self.monitoring_group)

        # A1.3 Statut DBE - Historique des changements de mode et bouton override
        self.dbe_control_group = QGroupBox(self.tr("Contrôle DBE"))
        self.dbe_control_layout = QVBoxLayout(self.dbe_control_group)
        self.dbe_history_label = QLabel(self.tr("Historique Modes DBE:"))
        self.dbe_history_display = QTextEdit()
        self.dbe_history_display.setReadOnly(True)
        self.dbe_history_display.document().setMaximumBlockCount(50)
        self.dbe_control_layout.addWidget(self.dbe_history_label)
        self.dbe_control_layout.addWidget(self.dbe_history_display)
        self.override_dbe_button = QPushButton(self.tr("Override Manuel DBE"))
        self.override_dbe_button.clicked.connect(self._on_override_dbe)
        self.dbe_control_layout.addWidget(self.override_dbe_button)
        self.layout.addWidget(self.dbe_control_group)

        # Chart Series Group
        self.chart_series_group = QGroupBox(self.tr("Séries Graphique"))
        self.chart_series_layout = QVBoxLayout(self.chart_series_group)
        self.candles_checkbox = QCheckBox(self.tr("Candles"))
        self.candles_checkbox.setChecked(True)
        self.candles_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("candles", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.candles_checkbox)
        self.volume_checkbox = QCheckBox(self.tr("Volume"))
        self.volume_checkbox.setChecked(True)
        self.volume_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("volume", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.volume_checkbox)
        self.rsi_checkbox = QCheckBox(self.tr("RSI"))
        self.rsi_checkbox.setChecked(True)
        self.rsi_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("rsi", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.rsi_checkbox)
        self.macd_checkbox = QCheckBox(self.tr("MACD"))
        self.macd_checkbox.setChecked(True)
        self.macd_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("macd", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.macd_checkbox)
        self.bollinger_checkbox = QCheckBox(self.tr("Bollinger Bands"))
        self.bollinger_checkbox.setChecked(True)
        self.bollinger_checkbox.stateChanged.connect(lambda state: self.chart_series_toggled.emit("bollinger", state == Qt.Checked))
        self.chart_series_layout.addWidget(self.bollinger_checkbox)
        self.layout.addWidget(self.chart_series_group)

        # Logs Group
        self.logs_group = QGroupBox(self.tr("Logs"))
        self.logs_layout = QVBoxLayout(self.logs_group)
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        self.log_filter_combo.currentTextChanged.connect(self.filter_logs)
        self.logs_layout.addWidget(self.log_filter_combo)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.document().setMaximumBlockCount(1000)
        self.logs_layout.addWidget(self.log_display)
        self.export_logs_button = QPushButton(self.tr("Export Logs"))
        self.export_logs_button.clicked.connect(self.export_logs)
        self.logs_layout.addWidget(self.export_logs_button)

        # Backtest Controls Group
        self.backtest_group = QGroupBox(self.tr("Contrôles Backtest"))
        self.backtest_layout = QFormLayout(self.backtest_group)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.valueChanged.connect(self.backtest_speed_changed.emit)
        self.backtest_layout.addRow(self.tr("Vitesse Simulation:"), self.speed_slider)
        self.layout.addWidget(self.backtest_group)

        self.layout.addStretch(1)

    def update_monitoring(self, portfolio_value, pnl, drawdown, risk_mode, open_positions, exposure, sl_pct, tp_pct):
        self.portfolio_value_label.setText(f"{portfolio_value:.2f} $")

        self.pnl_label.setText(f"{pnl:.2f} $")
        if pnl >= 0:
            self.pnl_label.setStyleSheet("color: green")
        else:
            self.pnl_label.setStyleSheet("color: red")

        self.drawdown_gauge.setValue(int(drawdown * 100))
        
        if risk_mode != self.risk_mode_label.text():
            self.append_dbe_history(f"Mode DBE changé: {self.risk_mode_label.text()} -> {risk_mode}")
        
        self.risk_mode_label.setText(risk_mode)
        color = "gray"
        if risk_mode == "NORMAL":
            color = "green"
        elif risk_mode == "DEFENSIVE":
            color = "orange"
        elif risk_mode == "AGGRESSIVE":
            color = "red"
        self.risk_mode_indicator.setStyleSheet(f"border-radius: 10px; background-color: {color};")

        self.open_positions_table.setRowCount(len(open_positions))
        for i, position in enumerate(open_positions):
            self.open_positions_table.setItem(i, 0, QTableWidgetItem(str(position.get('symbol', ''))))
            self.open_positions_table.setItem(i, 1, QTableWidgetItem(f"{position.get('amount', 0):.8f}"))
            self.open_positions_table.setItem(i, 2, QTableWidgetItem(f"{position.get('entry_price', 0):.2f}"))
            self.open_positions_table.setItem(i, 3, QTableWidgetItem(f"{position.get('current_price', 0):.2f}"))
        
        # Update exposure chart
        self.exposure_chart.clear()
        if open_positions:
            symbols = [pos['symbol'] for pos in open_positions]
            exposures = [pos['amount'] * pos['current_price'] for pos in open_positions]
            total_exposure = sum(exposures)
            if total_exposure > 0:
                exposure_pct = [exp / total_exposure for exp in exposures]
                x = range(len(symbols))
                bg = pg.BarGraphItem(x=x, height=exposure_pct, width=0.6, brush='b')
                self.exposure_chart.addItem(bg)
                ax = self.exposure_chart.getAxis('bottom')
                ticks = [list(zip(x, symbols))]
                ax.setTicks(ticks)

        self.sl_pct_label.setText(f"{sl_pct:.2%}")
        self.tp_pct_label.setText(f"{tp_pct:.2%}")

    def export_logs(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Logs", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.log_display.toPlainText())

    def append_dbe_history(self, message):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.dbe_history_display.append(f"[{timestamp}] {message}")
        self.dbe_history_display.verticalScrollBar().setValue(self.dbe_history_display.verticalScrollBar().maximum())

    def _on_override_dbe(self):
        self.append_log("Bouton 'Override Manuel DBE' cliqué. Fonctionnalité à implémenter.", "INFO")

    def append_log(self, message, level="INFO"):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.all_logs.append((level, log_entry))
        
        current_filter = self.log_filter_combo.currentText()
        if current_filter == "ALL" or current_filter == level:
            self.log_display.append(log_entry)
            self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def filter_logs(self, level):
        self.log_display.clear()
        if level == "ALL":
            for _, log_entry in self.all_logs:
                self.log_display.append(log_entry)
        else:
            for log_level, log_entry in self.all_logs:
                if log_level == level:
                    self.log_display.append(log_entry)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())
