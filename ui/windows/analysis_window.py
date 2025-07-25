from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QGridLayout, QPushButton, QListWidget, QGroupBox, QFormLayout, QLineEdit, QDateTimeEdit
import pyqtgraph as pg

class AnalysisWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis & Reporting")
        self.setMinimumSize(800, 600)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.create_tabs()

    def create_tabs(self):
        # A3.1 Courbes de Performance
        self.performance_curves_tab = QWidget()
        self.performance_curves_layout = QGridLayout(self.performance_curves_tab)
        self.tabs.addTab(self.performance_curves_tab, "Courbes de Performance")

        self.create_performance_curves_tab()

        # A3.2 Analyse DBE
        self.dbe_analysis_tab = QWidget()
        self.dbe_analysis_layout = QGridLayout(self.dbe_analysis_tab)
        self.tabs.addTab(self.dbe_analysis_tab, "Analyse DBE")

        self.create_dbe_analysis_tab()

        # A3.3 Rapports Automatiques
        self.reporting_tab = QWidget()
        self.reporting_layout = QVBoxLayout(self.reporting_tab)
        self.tabs.addTab(self.reporting_tab, "Rapports Automatiques")

        self.create_reporting_tab()

    def create_performance_curves_tab(self):
        self.equity_curve_widget = pg.PlotWidget(title="Equity Curve")
        self.drawdown_widget = pg.PlotWidget(title="Drawdown")
        self.trades_heatmap_widget = pg.PlotWidget(title="Trades Heatmap")

        self.performance_curves_layout.addWidget(self.equity_curve_widget, 0, 0)
        self.performance_curves_layout.addWidget(self.drawdown_widget, 1, 0)
        self.performance_curves_layout.addWidget(self.trades_heatmap_widget, 0, 1, 2, 1)

    def create_dbe_analysis_tab(self):
        self.dbe_mode_histogram_widget = pg.PlotWidget(title="DBE Mode Histogram")
        self.sltp_evolution_widget = pg.PlotWidget(title="SL/TP Evolution")
        self.performance_correlation_widget = pg.PlotWidget(title="Performance/DBE Correlation")

        self.dbe_analysis_layout.addWidget(self.dbe_mode_histogram_widget, 0, 0)
        self.dbe_analysis_layout.addWidget(self.sltp_evolution_widget, 1, 0)
        self.dbe_analysis_layout.addWidget(self.performance_correlation_widget, 0, 1, 2, 1)

    def create_reporting_tab(self):
        export_group = QGroupBox("Export")
        export_layout = QFormLayout(export_group)

        self.export_pdf_button = QPushButton("Export PDF")
        self.export_csv_button = QPushButton("Export CSV")

        export_layout.addRow(self.export_pdf_button)
        export_layout.addRow(self.export_csv_button)

        self.reporting_layout.addWidget(export_group)

        templates_group = QGroupBox("Templates")
        templates_layout = QVBoxLayout(templates_group)

        self.template_list = QListWidget()
        templates_layout.addWidget(self.template_list)

        self.reporting_layout.addWidget(templates_group)

        scheduling_group = QGroupBox("Scheduling")
        scheduling_layout = QFormLayout(scheduling_group)

        self.schedule_datetime_edit = QDateTimeEdit()
        self.schedule_button = QPushButton("Schedule Report")

        scheduling_layout.addRow(QLabel("Date/Time:"), self.schedule_datetime_edit)
        scheduling_layout.addRow(self.schedule_button)

        self.reporting_layout.addWidget(scheduling_group)
