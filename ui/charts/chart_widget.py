import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: time, open, close, low, high
        self.generatePicture()

    def generatePicture(self):
        ## pre-generate picture for faster displaying
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('k'))
        w = (self.data[1]['time'] - self.data[0]['time']) / 3.
        for (t, open, close, low, high) in self.data:
            p.drawLine(pg.QtCore.QPointF(t, low), pg.QtCore.QPointF(t, high))
            if open > close:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(pg.QtCore.QRectF(t-w, open, w*2, close-open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn in
        ## order for zoom/pan to work correctly. 
        ## (otherwise, only the visible portion of the item gets updated on resize)
        return pg.QtCore.QRectF(self.picture.boundingRect())

class ChartWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setLabel('left', 'Prix (USDT)')
        self.setLabel('bottom', 'Temps')
        self.setTitle('Graphique des Prix')

        # Exemple de données OHLC (Open, High, Low, Close)
        data = [
            (1, 30, 35, 28, 32),
            (2, 32, 38, 30, 36),
            (3, 36, 37, 33, 34),
            (4, 34, 36, 31, 35),
            (5, 35, 40, 33, 39),
            (6, 39, 41, 37, 38),
            (7, 38, 39, 35, 36),
            (8, 36, 38, 34, 37),
            (9, 37, 42, 35, 41),
            (10, 41, 43, 39, 42),
        ]
        # Convertir en format attendu par CandlestickItem
        ohlc_data = []
        for i, (t, open, high, low, close) in enumerate(data):
            ohlc_data.append({'time': t, 'open': open, 'close': close, 'low': low, 'high': high})

        item = CandlestickItem(ohlc_data)
        self.addItem(item)

        # Ajuster la vue pour que toutes les données soient visibles
        self.autoRange()