import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys
import json

# Import utils
sys.path.append(os.path.dirname(__file__))
from dashboard_utils import run_evaluation

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "ADAN Evaluation Dashboard"

# ------------------------------
# LAYOUT
# ------------------------------

app.layout = dbc.Container([
    dcc.Store(id='eval-results-store'),
    dbc.Row([
        dbc.Col([
            html.H2("ADAN Dashboard", className="display-4"),
            html.Hr(),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Configuration", className="card-title"),
                    dbc.Label("Asset"),
                    dbc.Input(id="input-asset", type="text", value="XRPUSDT", className="mb-2"),
                    dbc.Label("Capital ($)"),
                    dbc.Input(id="input-capital", type="number", value=20.0, className="mb-2"),
                    dbc.Label("Checkpoint"),
                    dbc.Input(id="input-checkpoint", type="text", value="checkpoints_final/adan_model_checkpoint_640000_steps.zip", className="mb-2"),
                    html.Hr(),
                    html.Label("Période"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="input-start-date", type="text", placeholder="YYYY-MM-DD"), width=6),
                        dbc.Col(dbc.Input(id="input-end-date", type="text", placeholder="YYYY-MM-DD"), width=6),
                    ], className="mb-2"),
                    dbc.Button("Bear Market 2022", id="btn-preset-bear", color="warning", size="sm", className="mb-2 w-100"),
                    dbc.Button("Lancer l'Évaluation", id="btn-run", color="primary", className="w-100"),
                    html.Div(id="status-msg", className="mt-2 text-info")
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trade Inspector", className="card-title"),
                    html.Div(id="trade-inspector-content", children="Lancez l'évaluation.")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Tabs(id="tabs-example", value='tab-overview', children=[
                        dcc.Tab(label='Vue d\'ensemble', value='tab-overview'),
                        dcc.Tab(label='Technique (5m)', value='tab-tech-5m'),
                        dcc.Tab(label='Technique (1h)', value='tab-tech-1h'),
                        dcc.Tab(label='Technique (4h)', value='tab-tech-4h'),
                        dcc.Tab(label='Performance', value='tab-performance'),
                        dcc.Tab(label='Données', value='tab-data'),
                    ]),
                    html.Div(id='tabs-content-example', className="p-3")
                ])
            ])
        ], width=9)
    ])
], fluid=True, className="p-4")

# ------------------------------
# CALLBACKS
# ------------------------------

# Preset Callback
@callback(
    [Output("input-start-date", "value"),
     Output("input-end-date", "value")],
    Input("btn-preset-bear", "n_clicks"),
    prevent_initial_call=True
)
def set_bear_market_dates(n_clicks):
    return "2022-01-01", "2023-01-01"

@callback(
    [Output("eval-results-store", "data"),
     Output("status-msg", "children")],
    Input("btn-run", "n_clicks"),
    [State("input-asset", "value"),
     State("input-capital", "value"),
     State("input-checkpoint", "value"),
     State("input-start-date", "value"),
     State("input-end-date", "value")],
    prevent_initial_call=True
)
def run_eval_callback(n_clicks, asset, capital, checkpoint, start_date, end_date):
    if not n_clicks:
        return no_update, "En attente..."
    
    try:
        # Handle empty dates
        if not start_date or not end_date:
            start_date = None
            end_date = None
            
        results = run_evaluation(asset, capital, checkpoint, start_date=start_date, end_date=end_date)
        if "error" in results:
            return no_update, f"Erreur: {results['error']}"
        
        # Serialization
        results["timestamps"] = [str(ts) for ts in results["timestamps"]]
        for t in results["trades"]:
            t["time"] = str(t["time"])
        
        for tf, df in results["data"].items():
            if not df.empty:
                df_reset = df.reset_index()
                if 'timestamp' not in df_reset.columns:
                     df_reset.rename(columns={'index': 'timestamp'}, inplace=True)
                df_reset['timestamp'] = df_reset['timestamp'].astype(str)
                results["data"][tf] = df_reset.to_dict('records')
            else:
                results["data"][tf] = []

        return results, f"Succès! {len(results['trades'])} trades, Return: {results['metrics']['total_return_pct']:.2f}%"
    except Exception as e:
        return no_update, f"Exception: {str(e)}"

def create_trade_table(trades, timeframe):
    try:
        trades_tf = [t for t in trades if t.get('timeframe') == timeframe]
        if not trades_tf:
            return html.Div(f"Aucun trade pour le timeframe {timeframe}.")
        
        # Format data for table
        table_data = []
        for t in trades_tf:
            ctx = t.get('context', {})
            rsi_val = ctx.get('rsi')
            macd_val = ctx.get('macd')
            
            rsi_str = f"{rsi_val:.1f}" if isinstance(rsi_val, (int, float)) else "N/A"
            macd_str = f"{macd_val:.4f}" if isinstance(macd_val, (int, float)) else "N/A"
            
            context_str = f"RSI: {rsi_str}, MACD: {macd_str}"
            
            table_data.append({
                "Time": t['time'],
                "Type": t['type'],
                "Price": f"{t['price']:.4f}",
                "Size": f"{t['size']:.4f}",
                "PnL": f"{t.get('pnl', 0):.4f}",
                "SL": f"{t.get('sl', 'N/A')}",
                "TP": f"{t.get('tp', 'N/A')}",
                "Context": context_str
            })
        
        df = pd.DataFrame(table_data)
        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, dark=True, responsive=True)
    except Exception as e:
        return html.Div(f"Erreur lors de la création du tableau : {str(e)}", className="text-danger")

def build_candles_with_trades(data_records, trades, timeframe, title):
    """Build a candlestick chart with OPEN/CLOSE markers and optional SL/TP overlays."""
    try:
        df = pd.DataFrame(data_records)
        if df.empty:
            return html.Div(f"Aucune donnée disponible pour {timeframe}.")
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            if 'index' in df.columns:
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                return html.Div(f"Colonnes manquantes pour {timeframe}.")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Basic required OHLC columns
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(set(df.columns)):
            return html.Div(f"OHLC manquants pour {timeframe}.")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name=f"{timeframe}"
        ))

        # Add trade markers
        tf_trades = [t for t in trades if t.get('timeframe') == timeframe]
        if tf_trades:
            open_x = [pd.to_datetime(t['time']) for t in tf_trades if t.get('type') == 'OPEN']
            open_y = [t['price'] for t in tf_trades if t.get('type') == 'OPEN']
            close_x = [pd.to_datetime(t['time']) for t in tf_trades if t.get('type') == 'CLOSE']
            close_y = [t['price'] for t in tf_trades if t.get('type') == 'CLOSE']

            if open_x:
                fig.add_trace(go.Scatter(
                    x=open_x, y=open_y, mode='markers', name='OPEN',
                    marker=dict(symbol='triangle-up', size=10, color='lime')
                ))
            if close_x:
                fig.add_trace(go.Scatter(
                    x=close_x, y=close_y, mode='markers', name='CLOSE',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ))

        fig.update_layout(template='plotly_dark', title=title, height=600, xaxis_rangeslider_visible=True)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Erreur graphique {timeframe}: {str(e)}", className="text-danger")

@callback(
    Output('tabs-content-example', 'children'),
    Input('tabs-example', 'value'),
    Input('eval-results-store', 'data')
)
def render_content(tab, data):
    if not data:
        return html.Div([html.H3("Bienvenue sur le Dashboard ADAN"), html.P("Configurez l'évaluation à gauche.")])

    metrics = data["metrics"]
    
    if tab == 'tab-overview':
        return html.Div([
            html.H3("KPIs Clés"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Capital Final", className="card-title"), html.H2(f"${metrics['final_equity']:.2f}", className="text-success")])], color="dark", inverse=True)),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Return", className="card-title"), html.H2(f"{metrics['total_return_pct']:.2f}%", className="text-info")])], color="dark", inverse=True)),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Win Rate", className="card-title"), html.H2(f"{metrics['win_rate']:.1f}%", className="text-warning")])], color="dark", inverse=True)),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4("Profit Factor", className="card-title"), html.H2(f"{metrics['profit_factor']:.2f}", className="text-primary")])], color="dark", inverse=True)),
            ], className="mb-4"),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=data["timestamps"], y=data["portfolio_values"], mode='lines', name='Equity', line=dict(color='gold'))], layout=go.Layout(template='plotly_dark', title="Évolution du Capital", height=400))),
            dcc.Graph(figure=go.Figure(data=[go.Scatter(x=data["timestamps"], y=data["drawdowns"], mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red'))], layout=go.Layout(template='plotly_dark', title="Drawdown (%)", height=300)))
        ])
    
    elif tab == 'tab-tech-5m':
        return html.Div([
            build_candles_with_trades(data.get("data", {}).get("5m", []), data["trades"], "5m", "Bougies + Trades (5m)"),
            html.Hr(),
            html.H5("Tableau des trades (5m)"),
            create_trade_table(data["trades"], "5m")
        ])

    elif tab == 'tab-tech-1h':
        return html.Div([
            build_candles_with_trades(data.get("data", {}).get("1h", []), data["trades"], "1h", "Bougies + Trades (1h)"),
            html.Hr(),
            html.H5("Tableau des trades (1h)"),
            create_trade_table(data["trades"], "1h")
        ])

    elif tab == 'tab-tech-4h':
        return html.Div([
            build_candles_with_trades(data.get("data", {}).get("4h", []), data["trades"], "4h", "Bougies + Trades (4h)"),
            html.Hr(),
            html.H5("Tableau des trades (4h)"),
            create_trade_table(data["trades"], "4h")
        ])

    elif tab == 'tab-performance':
        trades = data["trades"]
        pnls = [t["pnl"] for t in trades if t["pnl"] != 0]
        fig_hist = go.Figure(data=[go.Histogram(x=pnls, nbinsx=50, marker_color='teal')])
        fig_hist.update_layout(template='plotly_dark', title="Distribution des PnL par Trade")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
        return html.Div([dbc.Row([dbc.Col(dcc.Graph(figure=fig_hist), width=12)])])

    elif tab == 'tab-data':
        trades_df = pd.DataFrame(data["trades"])
        return html.Div([html.H4("Historique des Trades"), dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True, dark=True)])

# Callback for Trade Inspector (Disabled/Modified since no chart click)
@callback(
    Output("trade-inspector-content", "children"),
    Input("eval-results-store", "data") # Just show general info or last trade
)
def display_trade_data(data):
    if not data:
        return "Lancez l'évaluation."
    return "Consultez les onglets 'Technique' pour voir les tableaux détaillés des trades."

if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host='0.0.0.0')
