"""Punto de entrada del tablero (Dash multipágina). Expone `server` para gunicorn."""
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True, title="TBX11K · Segmentación de TB",
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

navbar = dbc.NavbarSimple(
    [dbc.NavItem(dbc.NavLink(p["name"], href=p["path"])) for p in
     sorted(dash.page_registry.values(), key=lambda p: p["order"])],
    brand="TBX11K · LACCA-UNet", color="primary", dark=True, fixed="top")

app.layout = html.Div([navbar, dbc.Container(dash.page_container, className="pt-5 mt-4", fluid="lg"),
                       html.Footer("Mujica Camilo y Romero Carlos · Deep Learning 2026 · Uso académico, no clínico",
                                   className="text-center text-muted py-4 small")])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 8050)))
