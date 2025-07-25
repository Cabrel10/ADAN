import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox
from PySide6.QtCore import Qt
from pathlib import Path
import pandas as pd
import ta

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # data must have fields: time, open, close, low, high
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('k'))
        if len(self.data) < 2:
            return

        w = (self.data[1]['time'] - self.data[0]['time']) / 3.
        for d in self.data:
            t = d['time']
            open_price = d['open']
            close_price = d['close']
            low_price = d['low']
            high_price = d['high']

            p.drawLine(pg.QtCore.QPointF(t, low_price), pg.QtCore.QPointF(t, high_price))
            if open_price > close_price:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(pg.QtCore.QRectF(t-w, open_price, w*2, close_price-open_price))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['5m', '1h', '4h'])
        self.timeframe_combo.currentTextChanged.connect(self.load_data)
        self.layout.addWidget(self.timeframe_combo)

        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.current_pair = 'ARBUSDT'
        self.load_data(self.timeframe_combo.currentText())

    def load_data(self, timeframe):
        file_path = Path(f"/home/morningstar/Documents/trading/ADAN/data/raw/{timeframe}/{self.current_pair}.csv")
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.plot_data(df)
        else:
            print(f"Data file not found: {file_path}")

    def plot_data(self, df):
        self.plot_widget.clear()

        # Candlestick chart
        ohlc_data = []
        for index, row in df.iterrows():
            time = row['timestamp'].timestamp()
            open_price = float(row['open'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
        ohlc_data.append({'time': time, 'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price})
        
        if ohlc_data:
            item = CandlestickItem(ohlc_data)
            self.plot_widget.addItem(item)

        # Indicators
        self.add_indicators(df)

        self.plot_widget.autoRange()

    def add_indicators(self, df):
        # SMA
        df['sma'] = ta.trend.sma_indicator(df['close'], window=20)
        self.plot_widget.plot(df['timestamp'].apply(lambda x: x.timestamp()), df['sma'], pen='b', name='SMA')

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        self.plot_widget.plot(df['timestamp'].apply(lambda x: x.timestamp()).to_numpy(), df['rsi'].to_numpy(), pen='g', name='RSI')

    def toggle_series(self, series_name, visible):
        # This function will be used to toggle the visibility of indicators
        pass