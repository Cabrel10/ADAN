"""
Dashboard sections package - individual section renderers
"""

from .header import render_header
from .decision_matrix import render_decision_matrix
from .positions import render_positions
from .closed_trades import render_closed_trades
from .performance import render_performance
from .system_health import render_system_health

__all__ = [
    "render_header",
    "render_decision_matrix",
    "render_positions",
    "render_closed_trades",
    "render_performance",
    "render_system_health",
]
