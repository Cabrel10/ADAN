from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QStatusBar, 
                               QToolBar, QLabel, QMenuBar, QMenu, QHBoxLayout, 
                               QMessageBox, QComboBox, QApplication, QTabWidget) # Added QMessageBox, QComboBox, QApplication, QTabWidget
from PySide6.QtCore import QTimer, QProcess, QTranslator, QLocale # Added QTimer, QProcess, QTranslator, QLocale
from PySide6.QtNetwork import QTcpServer, QHostAddress # Added QTcpServer and QHostAddress
from pathlib import Path # Added Path
import random # Added random for dummy data
import json # Added json
import sys
import os
import yaml # Added yaml

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.sys.path.append(project_root)

from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

from ..widgets.chart_widget import ChartWidget
from ..panels.side_panel import SidePanel

from .data_manager_dialog import DataManagerDialog
from ..widgets.reporting_widget import ReportingWidget
from .configuration_window import ConfigurationWindow
from .analysis_window import AnalysisWindow

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Setup translator
        self.translator = QTranslator()
        # Assuming translation files are in a 'translations' directory relative to the main_app.py
        # For example, 'translations/adan_fr.qm' or 'translations/adan_en.qm'
        # You would typically load the appropriate .qm file based on user's locale or settings.
        # For demonstration, let's assume we want to load 'adan_en.qm' for English.
        # In a real app, you'd get the locale from QLocale.system().name() or user settings.
        locale_name = QLocale.system().name() # e.g., 'en_US', 'fr_FR'
        translation_file = f"adan_{locale_name}.qm"
        translations_path = Path(__file__).parent.parent.parent / "translations"
        
        if self.translator.load(translation_file, str(translations_path)):
            QApplication.instance().installTranslator(self.translator)
            print(f"Loaded translator: {translation_file}")
        else:
            print(f"Failed to load translator: {translation_file} from {translations_path}")

        self.setWindowTitle(self.tr("ADAN Trading Bot"))
        self.setGeometry(100, 100, 1200, 800)

        self._load_config() # Load configuration at the start
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

        # Create a QTabWidget for different views
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # Add ChartWidget to a tab
        self.chart_tab = QWidget()
        self.chart_tab_layout = QVBoxLayout(self.chart_tab)
        self.chart_tab_layout.addWidget(self.chart_widget)
        self.tab_widget.addTab(self.chart_tab, self.tr("TradingView"))

        # Add ReportingWidget to a tab
        self.reporting_widget = ReportingWidget()
        self.tab_widget.addTab(self.reporting_widget, self.tr("Analyse & Reporting"))

        # Connect SidePanel signals to ChartWidget slots
        self.side_panel.chart_series_toggled.connect(self.chart_widget.toggle_series)
        self.side_panel.backtest_speed_changed.connect(self._on_backtest_speed_changed)

        # Initialize PortfolioManager with default config
        self.portfolio_manager = self._init_portfolio_manager()
        
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
            self.side_panel.append_log(self.tr(f"Error: Could not start TCP server: {self.tcp_server.errorString()}"))
        else:
            self.side_panel.append_log(self.tr(f"TCP server listening on port {self.PORT}"))

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu(self.tr("Fichier"))
        file_menu.addAction(self.tr("Quitter"), self.close)

        # Data Menu
        data_menu = menu_bar.addMenu(self.tr("Données"))
        data_menu.addAction(self.tr("Télécharger les données"), self._download_data)
        data_menu.addAction(self.tr("Traiter les données"), self._process_data)

        # View Menu
        view_menu = menu_bar.addMenu(self.tr("Vue"))

        # Tools Menu
        tools_menu = menu_bar.addMenu(self.tr("Outils"))
        tools_menu.addAction(self.tr("Configuration"), self._open_config_dialog)
        tools_menu.addAction(self.tr("Gestionnaire de Données"), self._open_data_manager_dialog)
        tools_menu.addAction(self.tr("Analyse"), self._open_analysis_window)

        # Help Menu
        help_menu = menu_bar.addMenu(self.tr("Aide"))
        help_menu.addAction(self.tr("À propos"), self._show_about_dialog)

    def _create_tool_bar(self):
        tool_bar = self.addToolBar(self.tr("Barre d'outils"))
        tool_bar.addAction(self.tr("Sélection de paire"), self._select_pair)
        
        # Pair ComboBox
        self.pair_combo = QComboBox()
        self.available_pairs = self.get_available_pairs()
        self.pair_combo.addItems(self.available_pairs)
        if "ARBUSDT" in self.available_pairs:
            self.pair_combo.setCurrentText('ARBUSDT') # Default to ARBUSDT
        self.pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        tool_bar.addWidget(QLabel(self.tr("Paire:")))
        tool_bar.addWidget(self.pair_combo)
        
        # Timeframe ComboBox
        self.timeframe_combo = QComboBox()
        self.available_timeframes = self.get_available_timeframes()
        self.timeframe_combo.addItems(self.available_timeframes)
        if "5m" in self.available_timeframes:
            self.timeframe_combo.setCurrentText('5m') # Default to 5m
        self.timeframe_combo.currentIndexChanged.connect(self._on_timeframe_changed)
        tool_bar.addWidget(QLabel(self.tr("Timeframe:")))
        tool_bar.addWidget(self.timeframe_combo)

        tool_bar.addAction(self.tr("Play Backtest"), self._play_backtest)
        tool_bar.addAction(self.tr("Pause Backtest"), self._pause_backtest)
        tool_bar.addAction(self.tr("Plot Live"), self._plot_live) # Added Plot Live action

    def get_available_pairs(self):
        # Use assets from config
        return self.config.get('data', {}).get('assets', [])

    def get_available_timeframes(self):
        # Use timeframes from config
        return self.config.get('data', {}).get('timeframes', [])

    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(self.tr("Prêt"))

    def _open_config_dialog(self):
        config_window = ConfigurationWindow(self)
        config_window.exec()

    def _open_data_manager_dialog(self):
        data_manager_dialog = DataManagerDialog(self)
        data_manager_dialog.exec()

    def _download_data(self):
        # For now, hardcode some values for testing
        symbol = "BTC/USDT"
        timeframe = "1h"
        exchange = "binance"
        self.side_panel.append_log(self.tr(f"Downloading data for {symbol} {timeframe} from {exchange}..."))
        self._run_cli_command(["download-data", "--symbol", symbol, "--timeframe", timeframe, "--exchange", exchange])

    def _process_data(self):
        self.side_panel.append_log(self.tr("Processing data..."))
        self._run_cli_command(["process-data"])

    def _select_pair(self):
        QMessageBox.information(self, self.tr("Sélection de Paire"), self.tr("Fonctionnalité de sélection de paire à implémenter."))

    def _on_pair_changed(self, index):
        selected_pair = self.pair_combo.currentText()
        self.chart_widget.load_data(self.timeframe_combo.currentText(), selected_pair)
        self.chart_widget.current_pair = selected_pair
        self.side_panel.append_log(self.tr(f"Selected pair changed to: {selected_pair}"))

    def _on_timeframe_changed(self, index):
        selected_timeframe = self.timeframe_combo.currentText()
        self.chart_widget.load_data(selected_timeframe, self.chart_widget.current_pair) # Pass timeframe and current pair to ChartWidget

    def _play_backtest(self):
        self.side_panel.append_log(self.tr("Launching backtest..."))
        self._run_cli_command(["run-backtest"])

    def _pause_backtest(self):
        QMessageBox.information(self, self.tr("Pause Backtest"), self.tr("Fonctionnalité de pause de backtest à implémenter."))

    def _plot_live(self):
        self.side_panel.append_log(self.tr("Starting live plot stream..."))
        self._run_cli_command(["plot-live"])

    def _load_config(self):
        """Load configuration from main_config.yaml."""
        config_path = Path(__file__).parent.parent.parent / "config" / "main_config.yaml"
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.side_panel.append_log(self.tr(f"Error: main_config.yaml not found at {config_path}"))
            self.config = {}
        except yaml.YAMLError as e:
            self.side_panel.append_log(self.tr(f"Error parsing main_config.yaml: {e}"))
            self.config = {}

    def _init_portfolio_manager(self):
        """Initialize the PortfolioManager with configuration from main_config.yaml."""
        self._load_config() # Load config first
        
        initial_balance = self.config.get('environment', {}).get('initial_balance', 10000.0)
        assets = [f"{asset}/USDT" for asset in self.config.get('environment', {}).get('assets', ['BTC', 'ETH', 'ADA'])]
        
        trading_rules = self.config.get('environment', {}).get('trading_rules', {})
        risk_management = self.config.get('environment', {}).get('risk_management', {})

        config = {
            'initial_equity': initial_balance,
            'assets': assets,
            'trading_rules': {
                'min_trade_size': trading_rules.get('min_trade_size', 0.0001),
                'min_notional_value': trading_rules.get('min_notional_value', 10.0),
                'max_notional_value': trading_rules.get('max_notional_value', 100000.0),
                'max_position_size': trading_rules.get('max_position_size', 0.1), # This might need to be derived from risk_management
                'commission_pct': trading_rules.get('commission_pct', 0.001)
            },
            'risk_management': {
                'max_drawdown_pct': risk_management.get('max_drawdown_pct', 0.2),
                'stop_loss_pct': risk_management.get('stop_loss', {}).get('max_stop_loss_pct', 0.02),
                'take_profit_pct': risk_management.get('take_profit', {}).get('max_take_profit_pct', 0.05), # Corrected mapping
                'position_sizing': risk_management.get('position_sizing', {}).get('concentration_limits', {})
            }
        }
        return PortfolioManager(config)

    def _update_monitoring_data(self):
        """Update monitoring data with real portfolio information."""
        try:
            # Get current market prices (simulated for now)
            current_prices = {
                'BTC/USDT': random.uniform(50000, 70000),
                'ETH/USDT': random.uniform(2500, 3500),
                'ADA/USDT': random.uniform(0.4, 0.7)
            }
            
            # Update portfolio with current prices
            self.portfolio_manager.update_market_price(current_prices)
            
            # Get portfolio metrics
            metrics = self.portfolio_manager.get_metrics()
            
            # Calculate PnL
            pnl = metrics['total_capital'] - self.portfolio_manager.config['initial_equity']
            
            # Prepare open positions data for the UI
            open_positions = []
            for asset, position in metrics['positions'].items():
                if position['is_open']:
                    current_price = current_prices.get(asset, 0)
                    open_positions.append({
                        'symbol': asset,
                        'amount': position['size'],
                        'entry_price': position['entry_price'],
                        'current_price': current_price
                    })
            
            # Calculate exposure (total value of positions / total capital)
            exposure = 0.0
            for pos in open_positions:
                exposure += pos['amount'] * pos['current_price']
            exposure = min(exposure / metrics['total_capital'], 1.0) if metrics['total_capital'] > 0 else 0.0
            
            # Get default risk parameters (these would come from your risk management system)
            sl_pct = self.portfolio_manager.config['risk_management'].get('stop_loss_pct', 0.02)
            tp_pct = self.portfolio_manager.config['risk_management'].get('take_profit_pct', 0.05)
            
            # Update the UI with real data
            self.side_panel.update_monitoring(
                portfolio_value=metrics['total_capital'],
                pnl=pnl,
                drawdown=abs(metrics.get('drawdown', 0)),
                risk_mode="NORMAL",  # This would come from your risk management system
                open_positions=open_positions,
                exposure=exposure,
                sl_pct=sl_pct,
                tp_pct=tp_pct
            )
            
        except Exception as e:
            self.side_panel.append_log(f"Error updating monitoring data: {str(e)}")
            # Fall back to dummy data in case of error
            self._update_with_dummy_data()
    
    def _update_with_dummy_data(self):
        """Fallback method to update with dummy data in case of errors."""
        portfolio_value = random.uniform(10000, 15000)
        drawdown = random.uniform(0.01, 0.10)
        risk_mode = "NORMAL"
        open_positions = []
        num_positions = random.randint(0, 3)
        for _ in range(num_positions):
            symbol = random.choice(["BTC/USDT", "ETH/USDT", "ADA/USDT"])
            amount = round(random.uniform(0.001, 1.0), 3)
            entry_price = round(random.uniform(20000, 70000), 2)
            current_price = round(entry_price * random.uniform(0.95, 1.05), 2)
            open_positions.append({
                "symbol": symbol,
                "amount": amount,
                "entry_price": entry_price,
                "current_price": current_price
            })
        exposure = random.uniform(0.0, 1.0)
        sl_pct = 0.02
        tp_pct = 0.05
        pnl = random.uniform(-1000, 1000) # Dummy PnL
        self.side_panel.update_monitoring(portfolio_value, pnl, drawdown, risk_mode, open_positions, exposure, sl_pct, tp_pct)

    def _on_backtest_speed_changed(self, speed):
        self.side_panel.append_log(self.tr(f"Backtest speed changed to: {speed}"))

    def _run_cli_command(self, args):
        python_executable = str(Path("/home/morningstar/miniconda3/envs/trading_env/bin/python"))
        cli_script = str(Path(__file__).parent.parent.parent / 'cli' / 'adan_cli.py')
        
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._handle_cli_stdout)
        process.readyReadStandardError.connect(self._handle_cli_stderr)
        process.finished.connect(self._handle_cli_finished)

        command = [python_executable, cli_script] + args
        self.side_panel.append_log(self.tr(f"Executing CLI command: {' '.join(command)}"))
        process.start(command[0], command[1:])

    def _handle_cli_stdout(self):
        process = self.sender()
        if process:
            data = process.readAllStandardOutput().data().decode()
            self.side_panel.append_log(self.tr(f"CLI STDOUT: {data.strip()}"))

    def _handle_cli_stderr(self):
        process = self.sender()
        if process:
            data = process.readAllStandardError().data().decode()
            self.side_panel.append_log(self.tr(f"CLI STDERR: {data.strip()}"))

    def _handle_cli_finished(self, exitCode, exitStatus):
        process = self.sender()
        if process:
            self.side_panel.append_log(self.tr(f"CLI command finished with exit code {exitCode} and status {exitStatus}"))
            process.deleteLater() # Clean up the process object

    def _on_new_connection(self):
        self.side_panel.append_log(self.tr("New TCP connection from CLI."))
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
                        metric_data.get("pnl", 0), # Added pnl
                        metric_data.get("drawdown", 0),
                        metric_data.get("risk_mode", "N/A"),
                        metric_data.get("open_positions", []),
                        metric_data.get("exposure", 0.0),
                        metric_data.get("sl_pct", 0.0),
                        metric_data.get("tp_pct", 0.0)
                    )
                    if "log_message" in metric_data:
                        self.side_panel.append_log(self.tr(f"CLI Metric: {metric_data['log_message']}"))
                except json.JSONDecodeError as e:
                    self.side_panel.append_log(self.tr(f"Error decoding JSON from socket: {e} - Data: {line}"))

    def _open_analysis_window(self):
        analysis_window = AnalysisWindow(self)
        analysis_window.exec()

    def _show_about_dialog(self):
        QMessageBox.about(self, self.tr("À propos d'ADAN Trading Bot"), self.tr("ADAN Trading Bot\nVersion 1.0\n\nDéveloppé par Gemini CLI (votre assistant)\n\nCe logiciel est un prototype d'interface utilisateur pour le bot de trading ADAN."))

    