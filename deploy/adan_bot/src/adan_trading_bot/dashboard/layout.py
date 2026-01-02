"""
Layout manager for ADAN Dashboard

Defines the hierarchical layout structure using Rich Layout.
"""

from rich.layout import Layout
from rich.console import Console


def create_layout() -> Layout:
    """
    Create the main dashboard layout.
    
    Layout structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                         HEADER                              │
    ├─────────────────────────────────────────────────────────────┤
    │                    DECISION MATRIX                          │
    ├─────────────────────────────────────────────────────────────┤
    │              MAIN (LEFT + RIGHT)                           │
    │  ┌─────────────────────┬─────────────────────────────────┐  │
    │  │                     │                                 │  │
    │  │   ACTIVE POSITIONS  │      CLOSED TRADES             │  │
    │  │                     │                                 │  │
    │  ├─────────────────────┼─────────────────────────────────┤  │
    │  │                     │                                 │  │
    │  │   PERFORMANCE       │      SYSTEM HEALTH             │  │
    │  │                     │                                 │  │
    │  └─────────────────────┴─────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    
    Returns:
        Rich Layout object
    """
    # Create main layout
    layout = Layout()
    
    # Split into header, decision matrix, and main sections
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="decision_matrix", size=14),
        Layout(name="main"),
    )
    
    # Split main into left and right columns
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    
    # Split left column into positions and performance
    layout["left"].split_column(
        Layout(name="positions"),
        Layout(name="performance", size=12),
    )
    
    # Split right column into trades and health
    layout["right"].split_column(
        Layout(name="trades"),
        Layout(name="health", size=16),
    )
    
    return layout


def create_compact_layout() -> Layout:
    """
    Create a compact layout for smaller terminals.
    
    Layout structure (stacked vertically):
    ┌─────────────────────────────────────────────────────────────┐
    │                         HEADER                              │
    ├─────────────────────────────────────────────────────────────┤
    │                    DECISION MATRIX                          │
    ├─────────────────────────────────────────────────────────────┤
    │                   ACTIVE POSITIONS                         │
    ├─────────────────────────────────────────────────────────────┤
    │                    CLOSED TRADES                           │
    ├─────────────────────────────────────────────────────────────┤
    │                    PERFORMANCE                             │
    ├─────────────────────────────────────────────────────────────┤
    │                   SYSTEM HEALTH                            │
    └─────────────────────────────────────────────────────────────┘
    
    Returns:
        Rich Layout object for compact display
    """
    # Create main layout
    layout = Layout()
    
    # Split into all sections vertically
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="decision_matrix", size=14),
        Layout(name="positions", size=12),
        Layout(name="trades", size=10),
        Layout(name="performance", size=10),
        Layout(name="health", size=12),
    )
    
    return layout


def get_optimal_layout(console: Console) -> Layout:
    """
    Get the optimal layout based on terminal size.
    
    Args:
        console: Rich Console object
    
    Returns:
        Optimal Layout for current terminal size
    """
    # Get terminal dimensions
    width = console.size.width
    height = console.size.height
    
    # Use compact layout for narrow terminals
    if width < 120 or height < 40:
        return create_compact_layout()
    else:
        return create_layout()


def update_layout_content(layout: Layout, sections: dict) -> None:
    """
    Update layout with rendered sections.
    
    Args:
        layout: Rich Layout object
        sections: Dictionary of section names to rendered content
    """
    # Update each section if it exists in the layout
    for section_name, content in sections.items():
        try:
            if layout.get(section_name) is not None:
                layout[section_name].update(content)
        except (KeyError, AttributeError):
            # Section doesn't exist in this layout, skip it
            pass
