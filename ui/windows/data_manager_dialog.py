from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QPushButton, QHBoxLayout, QFormLayout, QLineEdit, QComboBox, QGroupBox, QSpinBox, QTextEdit
from PySide6.QtCore import QProcess, Signal
from pathlib import Path

class DataManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gestionnaire de Données")
        self.setGeometry(200, 200, 800, 600)

        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # Onglet Téléchargement
        self.download_tab = QWidget()
        self.download_layout = QFormLayout(self.download_tab)

        self.symbol_input = QLineEdit()
        self.symbol_input.setText("BTC/USDT")
        self.download_layout.addRow(QLabel("Symbole:"), self.symbol_input)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
        self.timeframe_combo.setCurrentText("1h")
        self.download_layout.addRow(QLabel("Timeframe:"), self.timeframe_combo)

        self.exchange_input = QLineEdit()
        self.exchange_input.setText("binance")
        self.download_layout.addRow(QLabel("Exchange:"), self.exchange_input)

        self.download_button = QPushButton("Télécharger Historique")
        self.download_button.clicked.connect(self._download_historical_data)
        self.download_layout.addRow(self.download_button)

        self.download_log = QTextEdit()
        self.download_log.setReadOnly(True)
        self.download_layout.addRow(QLabel("Logs de Téléchargement:"), self.download_log)

        self.tab_widget.addTab(self.download_tab, "Téléchargement")

        # Onglet Pré-traitement
        self.preprocess_tab = QWidget()
        self.preprocess_layout = QVBoxLayout(self.preprocess_tab)

        self.convert_group = QGroupBox("Convertir Données Brutes")
        self.convert_layout = QHBoxLayout(self.convert_group)

        self.convert_5m_button = QPushButton("Convertir 5m")
        self.convert_5m_button.clicked.connect(lambda: self._process_data_timeframe("5m"))
        self.convert_layout.addWidget(self.convert_5m_button)

        self.convert_1h_button = QPushButton("Convertir 1h")
        self.convert_1h_button.clicked.connect(lambda: self._process_data_timeframe("1h"))
        self.convert_layout.addWidget(self.convert_1h_button)

        self.convert_3h_button = QPushButton("Convertir 3h")
        self.convert_3h_button.clicked.connect(lambda: self._process_data_timeframe("3h"))
        self.convert_layout.addWidget(self.convert_3h_button)

        self.preprocess_layout.addWidget(self.convert_group)

        self.preprocess_log = QTextEdit()
        self.preprocess_log.setReadOnly(True)
        self.preprocess_layout.addWidget(QLabel("Logs de Pré-traitement:"))
        self.preprocess_layout.addWidget(self.preprocess_log)

        self.tab_widget.addTab(self.preprocess_tab, "Pré-traitement")

        # Onglet Fusion et Split
        self.merge_split_tab = QWidget()
        self.merge_split_layout = QVBoxLayout(self.merge_split_tab)

        self.merge_split_button = QPushButton("Fusionner & Splitter Train/Val/Test")
        self.merge_split_button.clicked.connect(self._merge_and_split_data)
        self.merge_split_layout.addWidget(self.merge_split_button)

        self.merge_split_log = QTextEdit()
        self.merge_split_log.setReadOnly(True)
        self.merge_split_layout.addWidget(QLabel("Logs de Fusion/Split:"))
        self.merge_split_layout.addWidget(self.merge_split_log)

        self.tab_widget.addTab(self.merge_split_tab, "Fusion & Split")

        # Onglet Explorer les fichiers (Placeholder)
        self.explore_tab = QWidget()
        self.explore_layout = QVBoxLayout(self.explore_tab)
        self.explore_layout.addWidget(QLabel("Fonctionnalité d'exploration des fichiers à implémenter."))
        self.tab_widget.addTab(self.explore_tab, "Explorer Fichiers")

    def _download_historical_data(self):
        symbol = self.symbol_input.text()
        timeframe = self.timeframe_combo.currentText()
        exchange = self.exchange_input.text()
        self._run_cli_command(["download-data", "--symbol", symbol, "--timeframe", timeframe, "--exchange", exchange], self.download_log)

    def _process_data_timeframe(self, timeframe):
        self.preprocess_log.append(f"Processing data for {timeframe}...")
        # Assuming process_data.py can take a timeframe argument
        self._run_cli_command(["process-data", "--timeframe", timeframe], self.preprocess_log)

    def _merge_and_split_data(self):
        self.merge_split_log.append("Merging and splitting data...")
        # Assuming a new CLI command for merging and splitting
        self._run_cli_command(["merge-data"], self.merge_split_log)

    def _run_cli_command(self, args, log_widget):
        python_executable = str(Path("/home/morningstar/miniconda3/envs/trading_env/bin/python"))
        cli_script = str(Path(__file__).parent.parent.parent.parent / 'ADAN' / 'cli' / 'adan_cli.py')
        
        process = QProcess(self)
        process.readyReadStandardOutput.connect(lambda: self._handle_cli_stdout(process, log_widget))
        process.readyReadStandardError.connect(lambda: self._handle_cli_stderr(process, log_widget))
        process.finished.connect(lambda exitCode, exitStatus: self._handle_cli_finished(process, exitCode, exitStatus, log_widget))

        command = [python_executable, cli_script] + args
        log_widget.append(f"Executing CLI command: {' '.join(command)}")
        process.start(command[0], command[1:])

    def _handle_cli_stdout(self, process, log_widget):
        data = process.readAllStandardOutput().data().decode()
        log_widget.append(f"CLI STDOUT: {data.strip()}")

    def _handle_cli_stderr(self, process, log_widget):
        data = process.readAllStandardError().data().decode()
        log_widget.append(f"CLI STDERR: {data.strip()}")

    def _handle_cli_finished(self, process, exitCode, exitStatus, log_widget):
        log_widget.append(f"CLI command finished with exit code {exitCode} and status {exitStatus}")
        process.deleteLater()
