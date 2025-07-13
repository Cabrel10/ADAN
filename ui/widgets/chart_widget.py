from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import mplfinance as mpf
import pandas as pd
from pathlib import Path
import ta.momentum
import ta.trend # Added ta.trend
import ta.volatility # Added ta.volatility
from ADAN.config.paths import RAW_DATA_DIR # Import RAW_DATA_DIR

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.ax = self.figure.add_subplot(3, 1, 1) # Changed to 3x1 subplot for main chart
        self.ax_rsi = self.figure.add_subplot(3, 1, 2, sharex=self.ax) # RSI subplot
        self.ax_macd = self.figure.add_subplot(3, 1, 3, sharex=self.ax) # MACD subplot
        
        self.current_timeframe = '5m'
        self.current_pair = 'ARBUSDT'
        self.data_df = None # Store the DataFrame
        self.series_visibility = {'candles': True, 'volume': True, 'rsi': True, 'macd': True, 'bollinger': True} # Track visibility
        self.load_data(self.current_timeframe, self.current_pair)

    def load_data(self, timeframe, pair):
        self.current_timeframe = timeframe
        self.current_pair = pair
        file_path = RAW_DATA_DIR / timeframe / f'{pair}.csv'
        try:
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            # Ensure column names are correct for mplfinance
            df.columns = [col.capitalize() for col in df.columns]
            self.data_df = df # Store the loaded DataFrame
            self.update_chart()
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    def update_chart(self):
        if self.data_df is None: return

        self.ax.clear()
        self.ax_rsi.clear()
        self.ax_macd.clear()

        mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)

        addplot = []
        if self.series_visibility['volume']:
            addplot.append(mpf.make_addplot(self.data_df['Volume'], panel=1, type='bar', color='gray', ax=self.ax_rsi))

        # Calculate Bollinger Bands
        if self.series_visibility['bollinger']:
            self.data_df['bb_bbm'] = ta.volatility.bollinger_mavg(self.data_df['Close'])
            self.data_df['bb_bbh'] = ta.volatility.bollinger_hband(self.data_df['Close'])
            self.data_df['bb_bbl'] = ta.volatility.bollinger_lband(self.data_df['Close'])
            addplot.extend([
                mpf.make_addplot(self.data_df['bb_bbm'], ax=self.ax, color='blue'),
                mpf.make_addplot(self.data_df['bb_bbh'], ax=self.ax, color='red'),
                mpf.make_addplot(self.data_df['bb_bbl'], ax=self.ax, color='green')
            ])

        # Plot candlestick chart
        if self.series_visibility['candles']:
            try:
                mpf.plot(self.data_df, type='candle', ax=self.ax, volume=self.ax, show_nontrading=True, style=s, addplot=addplot)
            except Exception as e:
                print(f"Error plotting with mplfinance: {e}")
                print(f"DataFrame columns: {self.data_df.columns.tolist()}")
                print(f"DataFrame head:\n{self.data_df.head()}")
        else:
            self.ax.set_visible(False)

        # Calculate and plot RSI
        if self.series_visibility['rsi']:
            self.data_df['RSI'] = ta.momentum.rsi(self.data_df['Close'], window=14)
            self.ax_rsi.plot(self.data_df.index, self.data_df['RSI'], label='RSI')
            self.ax_rsi.axhline(70, linestyle='--', alpha=0.5, color='red')
            self.ax_rsi.axhline(30, linestyle='--', alpha=0.5, color='green')
            self.ax_rsi.set_ylabel('RSI')
            self.ax_rsi.legend()
        else:
            self.ax_rsi.set_visible(False)

        # Calculate and plot MACD
        if self.series_visibility['macd']:
            self.data_df['macd'] = ta.trend.macd(self.data_df['Close'])
            self.data_df['macd_signal'] = ta.trend.macd_signal(self.data_df['Close'])
            self.data_df['macd_diff'] = ta.trend.macd_diff(self.data_df['Close'])
            self.ax_macd.plot(self.data_df.index, self.data_df['macd'], label='MACD', color='blue')
            self.ax_macd.plot(self.data_df.index, self.data_df['macd_signal'], label='Signal', color='red')
            self.ax_macd.bar(self.data_df.index, self.data_df['macd_diff'], label='Histogram', color='gray', alpha=0.7)
            self.ax_macd.set_ylabel('MACD')
            self.ax_macd.legend()
        else:
            self.ax_macd.set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw()

    def toggle_series(self, series_name, visible):
        self.series_visibility[series_name] = visible
        self.update_chart()
