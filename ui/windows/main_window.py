from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QStatusBar, QToolBar, QLabel, QMenuBar, QMenu, QHBoxLayout, QMessageBox, QComboBox # Added QMessageBox and QComboBox
from PySide6.QtCore import QTimer, QProcess # Added QTimer and QProcess
from PySide6.QtNetwork import QTcpServer, QHostAddress # Added QTcpServer and QHostAddress
from pathlib import Path # Added Path
import random # Added random for dummy data
import json # Added json

from ..widgets.chart_widget import ChartWidget
from ..panels.side_panel import SidePanel
from ..widgets.config_dialog import ConfigDialog # Import ConfigDialog
from .data_manager_dialog import DataManagerDialog # Import DataManagerDialog

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ADAN Trading Bot")
        self.setGeometry(100, 100, 1200, 800)

        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QHBoxLayout(self.central_widget) # Changed to QHBoxLayout

        # ChartWidget and SidePanel
        self.chart_widget = ChartWidget()
        self.side_panel = SidePanel()
        
        self.main_layout.addWidget(self.chart_widget)
        self.main_layout.addWidget(self.side_panel)

        # Connect SidePanel signals to ChartWidget slots
        self.side_panel.chart_series_toggled.connect(self.chart_widget.toggle_series)
        self.side_panel.backtest_speed_changed.connect(self._on_backtest_speed_changed)

        # Setup monitoring update timer
        self.monitoring_timer = QTimer(self)
        self.monitoring_timer.setInterval(1000) # Update every 1 second
        self.monitoring_timer.timeout.connect(self._update_monitoring_data)
        self.monitoring_timer.start()

        # Setup TCP Server for real-time metrics
        self.tcp_server = QTcpServer(self)
        self.tcp_server.newConnection.connect(self._on_new_connection)
        self.PORT = 65432
        if not self.tcp_server.listen(QHostAddress.LocalHost, self.PORT):
            self.side_panel.append_log(f"Error: Could not start TCP server: {self.tcp_server.errorString()}")
        else:
            self.side_panel.append_log(f"TCP server listening on port {self.PORT}")

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("Fichier")
        file_menu.addAction("Quitter", self.close)

        # Data Menu
        data_menu = menu_bar.addMenu("Données")
        data_menu.addAction("Télécharger les données", self._download_data)
        data_menu.addAction("Traiter les données", self._process_data)

        # View Menu
        view_menu = menu_bar.addMenu("Vue")

        # Tools Menu
        tools_menu = menu_bar.addMenu("Outils")
        tools_menu.addAction("Configuration", self._open_config_dialog) # Add action to open ConfigDialog
        tools_menu.addAction("Gestionnaire de Données", self._open_data_manager_dialog)

        # Help Menu
        help_menu = menu_bar.addMenu("Aide")
        help_menu.addAction("À propos", self._show_about_dialog)

    def _create_tool_bar(self):
        tool_bar = self.addToolBar("Barre d'outils")
        tool_bar.addAction("Sélection de paire", self._select_pair)
        
        # Pair ComboBox
        self.pair_combo = QComboBox()
        self.available_pairs = [path.stem for path in Path("/home/morningstar/Documents/trading/ADAN/data/raw/5m").glob("*.csv")]
        self.pair_combo.addItems(self.available_pairs)
        self.pair_combo.setCurrentText('ARBUSDT') # Default to ARBUSDT
        self.pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        tool_bar.addWidget(QLabel("Paire:"))
        tool_bar.addWidget(self.pair_combo)
        
        # Timeframe ComboBox
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
        self.timeframe_combo.setCurrentText('5m') # Default to 5m
        self.timeframe_combo.currentIndexChanged.connect(self._on_timeframe_changed)
        tool_bar.addWidget(QLabel("Timeframe:"))
        tool_bar.addWidget(self.timeframe_combo)

        tool_bar.addAction("Play Backtest", self._play_backtest)
        tool_bar.addAction("Pause Backtest", self._pause_backtest)
        tool_bar.addAction("Plot Live", self._plot_live) # Added Plot Live action

    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Prêt")

    def _open_config_dialog(self):
        config_dialog = ConfigDialog(self)
        config_dialog.exec()

    def _open_data_manager_dialog(self):
        data_manager_dialog = DataManagerDialog(self)
        data_manager_dialog.exec()

    def _download_data(self):
        # For now, hardcode some values for testing
        symbol = "BTC/USDT"
        timeframe = "1h"
        exchange = "binance"
        self.side_panel.append_log(f"Downloading data for {symbol} {timeframe} from {exchange}...")
        self._run_cli_command(["download-data", "--symbol", symbol, "--timeframe", timeframe, "--exchange", exchange])

    def _process_data(self):
        self.side_panel.append_log("Processing data...")
        self._run_cli_command(["process-data"])

    def _select_pair(self):
        QMessageBox.information(self, "Sélection de Paire", "Fonctionnalité de sélection de paire à implémenter.")

    def _on_pair_changed(self, index):
        selected_pair = self.pair_combo.currentText()
        self.chart_widget.load_data(self.timeframe_combo.currentText(), selected_pair)
        self.chart_widget.current_pair = selected_pair
        self.side_panel.append_log(f"Selected pair changed to: {selected_pair}")

    def _on_timeframe_changed(self, index):
        selected_timeframe = self.timeframe_combo.currentText()
        self.chart_widget.load_data(selected_timeframe, self.chart_widget.current_pair) # Pass timeframe and current pair to ChartWidget

    def _play_backtest(self):
        self.side_panel.append_log("Launching backtest...")
        self._run_cli_command(["run-backtest"])

    def _pause_backtest(self):
        QMessageBox.information(self, "Pause Backtest", "Fonctionnalité de pause de backtest à implémenter.")

    def _plot_live(self):
        self.side_panel.append_log("Starting live plot stream...")
        self._run_cli_command(["plot-live"])

    def _update_monitoring_data(self):
        # Generate dummy data for demonstration
        portfolio_value = random.uniform(10000, 15000)
        drawdown = random.uniform(0.01, 0.10)
        risk_modes = ["Low", "Medium", "High"]
        risk_mode = random.choice(risk_modes)
        self.side_panel.update_monitoring(portfolio_value, drawdown, risk_mode)
        # self.side_panel.append_log(f"Monitoring updated: Value={portfolio_value:.2f}, Drawdown={drawdown:.2%}, Risk={risk_mode}") # Commented out to avoid log spam

    def _on_backtest_speed_changed(self, speed):
        self.side_panel.append_log(f"Backtest speed changed to: {speed}")

    def _run_cli_command(self, args):
        python_executable = str(Path("/home/morningstar/miniconda3/envs/trading_env/bin/python"))
        cli_script = str(Path(__file__).parent.parent.parent / 'cli' / 'adan_cli.py')
        
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._handle_cli_stdout)
        process.readyReadStandardError.connect(self._handle_cli_stderr)
        process.finished.connect(self._handle_cli_finished)

        command = [python_executable, cli_script] + args
        self.side_panel.append_log(f"Executing CLI command: {' '.join(command)}")
        process.start(command[0], command[1:])

    def _handle_cli_stdout(self):
        process = self.sender()
        if process:
            data = process.readAllStandardOutput().data().decode()
            self.side_panel.append_log(f"CLI STDOUT: {data.strip()}")

    def _handle_cli_stderr(self):
        process = self.sender()
        if process:
            data = process.readAllStandardError().data().decode()
            self.side_panel.append_log(f"CLI STDERR: {data.strip()}")

    def _handle_cli_finished(self, exitCode, exitStatus):
        process = self.sender()
        if process:
            self.side_panel.append_log(f"CLI command finished with exit code {exitCode} and status {exitStatus}")
            process.deleteLater() # Clean up the process object

    def _on_new_connection(self):
        self.side_panel.append_log("New TCP connection from CLI.")
        self.client_socket = self.tcp_server.nextPendingConnection()
        self.client_socket.readyRead.connect(self._read_socket_data)

    def _read_socket_data(self):
        while self.client_socket.bytesAvailable():
            data = self.client_socket.readAll().data().decode('utf-8')
            for line in data.splitlines():
                try:
                    metric_data = json.loads(line)
                    self.side_panel.update_monitoring(
                        metric_data.get("portfolio_value", 0),
                        metric_data.get("drawdown", 0),
                        metric_data.get("risk_mode", "N/A")
                    )
                    if "log_message" in metric_data:
                        self.side_panel.append_log(f"CLI Metric: {metric_data['log_message']}")
                except json.JSONDecodeError as e:
                    self.side_panel.append_log(f"Error decoding JSON from socket: {e} - Data: {line}")

    def _show_about_dialog(self):
        QMessageBox.about(self, "À propos d'ADAN Trading Bot", "ADAN Trading Bot\nVersion 1.0\n\nDéveloppé par Gemini CLI (votre assistant)\n\nCe logiciel est un prototype d'interface utilisateur pour le bot de trading ADAN.")

    