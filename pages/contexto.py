import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from src.data import ROOT, leaderboard

dash.register_page(__name__, path="/", name="Contexto", order=0)

lb = leaderboard()
orig = lb[lb.modelo == "lacca_unet"].iloc[0]


def kpi(v, l):
    return dbc.Col(html.Div([html.Div(v, className="v"), html.Div(l, className="l")], className="kpi"), md=3, xs=6, className="mb-3")


layout = html.Div([
    html.Div([html.H1("Segmentación de lesiones de tuberculosis en radiografías de tórax"),
              html.P("Benchmark de 9 arquitecturas y un modelo original (LACCA-UNet) sobre TBX11K, "
                     "con protocolo estadístico y evaluación ampliada.", className="lead mb-0")], className="hero"),
    dbc.Row([kpi("11 200", "radiografías en TBX11K"), kpi("14", "modelos/variantes evaluados"),
             kpi(f"{orig.dice_media:.3f}", "Dice de LACCA-UNet (10 semillas)"), kpi("0,988", "techo humano del GT (Dice)")]),
    html.Div([html.H4("Por qué importa"),
              html.P("La tuberculosis sigue siendo una de las principales causas de muerte por enfermedad infecciosa. "
                     "Localizar las lesiones en la radiografía ayuda a priorizar casos donde no hay radiólogo disponible."),
              html.Div("Implicación: segmentar (no solo clasificar) permite verificar que el modelo mira la lesión y no un atajo.",
                       className="insight")], className="card-sec"),
    html.Div([html.H4("Preguntas de investigación"),
              html.Ul([html.Li("RQ1 · ¿Hay diferencias significativas entre familias de arquitecturas con presupuesto idéntico?"),
                       html.Li("RQ2 · ¿Los transformadores superan a las CNN con lesiones pequeñas (~8,8 % de los píxeles)?"),
                       html.Li("RQ3 · ¿Cuánto error se debe al techo del ground truth y cuánto al modelo?"),
                       html.Li("RQ4 · ¿Grad-CAM coincide con la localización humana?")])], className="card-sec"),
    html.Div([html.H4("Metodología"),
              html.Ol([html.Li("EDA orientado a decisiones y GT de plata con SAM 2 (techo humano Dice 0,988)."),
                       html.Li("HPO de 6 intentos por modelo, 5 semillas base y 5 extra para el original y el top 3."),
                       html.Li("Test sellado: 80 positivas y 40 negativas. Holm sobre 9 valores p, Cliff δ ≥ 0,147, margen 0,01."),
                       html.Li("Calibración, robustez, Grad-CAM y validación externa (Montgomery, Shenzhen).")])], className="card-sec"),
    html.Div([html.H4("Ficha de datos"),
              dcc.Markdown((ROOT / "DATA_STATEMENT.md").read_text(encoding="utf-8")),
              html.H4("Limitaciones y ética", className="mt-3"),
              html.Ul([html.Li("El GT de plata (SAM 2) acota el Dice posible; no hay máscaras de expertos."),
                       html.Li("Resultado principal: LACCA-UNet es competitivo pero no superior al mejor baseline."),
                       html.Li("Uso académico; no sirve para diagnóstico clínico.")]),
              html.P(["Autores: Mujica Camilo y Romero Carlos. ",
                      html.A("Repositorio", href="https://github.com/camilo0709/entregable3"), " · ",
                      html.A("DOI", href="https://doi.org/10.5281/zenodo.24294484"), "."])], className="card-sec"),
])
