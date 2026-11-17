import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
from src.data import tabla
from src import plots

dash.register_page(__name__, path="/eda", name="EDA", order=1)

eda = tabla("eda_train_val")

layout = html.Div([
    html.H2("Exploración de datos"),
    html.Div("Las lesiones son pequeñas y variables: eso explica por qué el Dice se acerca a 0,57.", className="insight"),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.splits(tabla("eda_splits"))), md=5),
             dbc.Col([html.Label("Agrupar por"),
                      dcc.Dropdown(id="eda-color", clearable=False, value="tb_type",
                                   options=[{"label": "Tipo de TB", "value": "tb_type"}, {"label": "Partición", "value": "split"}]),
                      dcc.Graph(id="eda-hist")], md=7)]),
    html.Div(id="eda-insight", className="insight"),
    html.H4("Explorador"),
    dbc.Row([dbc.Col([html.Label("Partición"), dcc.Dropdown(id="eda-split", multi=True, value=["train", "val"],
                      options=[{"label": s, "value": s} for s in ["train", "val"]])], md=4),
             dbc.Col([html.Label("Fracción máxima de lesión"),
                      dcc.Slider(id="eda-max", min=0.05, max=0.3, step=0.01, value=0.3,
                                 marks={0.05: "5 %", 0.15: "15 %", 0.3: "30 %"})], md=8)]),
    dcc.Graph(id="eda-scatter"),
    html.Div("Cada gráfico se actualiza con los filtros. Más cajas por imagen suelen implicar más área de lesión.", className="insight"),
    html.H4("Figuras del EDA del libro"),
    dbc.Row([dbc.Col(html.Img(src=dash.get_asset_url(f"eda/{f}"), style={"width": "100%"}), md=6, className="mb-3")
             for f in ["a3_areas.png", "a6_intensidades.png", "a7_shift.png", "a4_shortcuts.png"]]),
    html.Div("Las figuras corresponden al análisis completo (capítulo 2 del libro): áreas de lesión, intensidades, desplazamiento de dominio y atajos.", className="insight"),
])


@callback(Output("eda-hist", "figure"), Output("eda-insight", "children"), Input("eda-color", "value"))
def _hist(color):
    med = eda.mask_frac.median()
    return plots.hist_mask(eda, color), f"La mediana de la lesión es {med:.1%} de la imagen; la mayoría son focos pequeños."


@callback(Output("eda-scatter", "figure"), Input("eda-split", "value"), Input("eda-max", "value"))
def _sc(splits, mx):
    import plotly.express as px
    d = eda[(eda.split.isin(splits or [])) & (eda.mask_frac <= mx)]
    f = px.strip(d, x="n_boxes", y="mask_frac", color="tb_type", title="Fracción de lesión según número de cajas",
                 labels={"n_boxes": "Cajas por imagen", "mask_frac": "Lesión / imagen"})
    f.update_layout(height=380); return f
