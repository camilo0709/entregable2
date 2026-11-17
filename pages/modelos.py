import base64, os
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output, State, callback, no_update
from PIL import Image
from src import plots
from src.data import ROOT, tabla, leaderboard, perfil, NOMBRES
from src import inference as inf

dash.register_page(__name__, path="/modelos", name="Modelos", order=2)

lb = leaderboard()
EJEMPLOS = ["tb0018", "tb0066", "tb0077", "tb0124", "tb0177", "tb0266"]
MODELOS = [{"label": NOMBRES[m], "value": m} for m in lb.modelo]
CON_CAL = set(tabla("calibracion").modelo)
CON_CAM = ["lacca_unet", "attention_unet", "unet_pp", "mednext", "unet"]

cols = [("Modelo", "Modelo"), ("familia", "Familia"), ("n_semillas", "Semillas"), ("dice_media", "Dice (media)"),
        ("dice_sd", "Dice (sd)"), ("ece", "ECE"), ("parametros", "Parámetros"),
        ("gflops_1img", "GFLOPs"), ("latencia_ms_ort_cpu", "Latencia CPU (ms)")]
tab = lb[[c for c, _ in cols]].copy()
tab["dice_media"] = tab.dice_media.round(4); tab["dice_sd"] = tab.dice_sd.round(4); tab["ece"] = tab.ece.round(4)

PSEUDO = """```
LACCA(F):                              # F: mapa de características [C,H,W]
  (θ,d) ← antisimetrizar(θ0,d0)        # eje de simetría aprendido
  F~, M ← reflejar(F, θ, d)            # grid_sample con reflexión y máscara M
  q,k   ← normalizar(Wq F, Wk F~)
  A     ← atención_local(q, k, Wv F~)  # ventana de radio r contra el espejo
  D     ← (Wv F − A) ⊙ M               # residuo de asimetría
  return F + γ · ψ([F, D])             # compuerta ReZero, γ0 = 0
```"""
PRUEBAS = ["Equivarianza exacta al volteo horizontal", "Identidad cuando γ = 0",
           "Gradientes finitos en todos los parámetros", "Salida con la misma forma que la entrada",
           "La red completa sobreajusta un lote pequeño", "Exportación ONNX con ida y vuelta (tolerancia 1e-4)"]

cmp = tabla("comparaciones_original")
cmp = cmp[cmp.prueba.str.startswith("T1")][["rival", "n_img", "dice_orig", "dice_rival", "dif_media", "p_holm", "cliff", "gana_original"]].round(4)


def tarjeta(titulo, hijos, insight=None):
    return html.Div([html.H4(titulo), *hijos] + ([html.Div(insight, className="insight")] if insight else []), className="card-sec")


layout = html.Div([
    html.H2("Modelos"),
    tarjeta("Tabla de líderes", [dash_table.DataTable(
        id="lb", columns=[{"name": n, "id": c} for c, n in cols], data=tab.to_dict("records"),
        sort_action="native", filter_action="native", page_size=14,
        style_table={"overflowX": "auto"}, style_cell={"fontSize": 13, "padding": "6px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#EEF2F7"},
        style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "original"'}, "backgroundColor": "#FFF6EE", "fontWeight": "bold"}])],
        "Dice de test sobre las 80 positivas del conjunto sellado. Ordene o filtre las columnas para comparar."),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.barras_dice(lb)), lg=6), dbc.Col(dcc.Graph(figure=plots.pareto(lb)), lg=6)]),
    html.Div("Los mejores modelos están entre 0,55 y 0,57; el costo en latencia varía hasta 6 veces sin ganar precisión.", className="insight"),

    tarjeta("Predicción en vivo", [
        dbc.Row([
            dbc.Col([html.Label("Modelo"), dcc.Dropdown(id="p-modelo", options=MODELOS, value="lacca_unet", clearable=False),
                     html.Label("Radiografía de ejemplo", className="mt-2"),
                     dcc.Dropdown(id="p-ej", options=[{"label": e, "value": e} for e in EJEMPLOS], value=EJEMPLOS[0], clearable=True),
                     dcc.Upload(id="p-up", children=html.Div(["O arrastre una imagen (PNG/JPEG, ≤ 8 MB)"]),
                                style={"border": "1px dashed #999", "borderRadius": 8, "padding": 14, "textAlign": "center", "marginTop": 8}),
                     html.Label("Umbral", className="mt-2"),
                     dcc.Slider(id="p-umb", min=0.1, max=0.9, step=0.05, value=0.5, marks={0.1: "0,1", 0.5: "0,5", 0.9: "0,9"}),
                     dbc.Checkbox(id="p-temp", label="Probabilidad calibrada (escalado de temperatura, si existe)", value=True, className="mt-2"),
                     dbc.Button("Predecir", id="p-go", color="primary", className="mt-3 w-100"),
                     html.Div(id="p-aviso", className="mt-2")], md=4),
            dbc.Col(dcc.Loading(html.Div(id="p-out")), md=8)])],
        "Inferencia con ONNX Runtime en CPU. La incertidumbre es la entropía binaria de la probabilidad por píxel."),

    tarjeta("Diagnóstico del modelo", [
        dcc.Dropdown(id="d-modelo", options=MODELOS, value="lacca_unet", clearable=False),
        dbc.Row([dbc.Col(dcc.Graph(id="d-caso"), lg=6), dbc.Col(dcc.Graph(id="d-conf"), lg=6)]),
        html.Div(id="d-cal"),
        dcc.Graph(id="d-rob"),
        html.Div(id="d-cam")],
        "Calibración y robustez se midieron con una sola semilla (42) y 5 modelos: son indicativas."),

    tarjeta("Comparación estadística del modelo original", [
        dash_table.DataTable(columns=[{"name": c, "id": c} for c in cmp.columns], data=cmp.to_dict("records"),
                             style_cell={"fontSize": 13, "padding": "6px"}, style_header={"fontWeight": "bold"}),
        html.P("Wilcoxon pareado por imagen con corrección de Holm; Cliff δ como tamaño del efecto.", className="small text-muted mt-2")],
        "LACCA-UNet iguala a Attention U-Net y U-Net++ (p Holm = 1) y supera solo a MedNeXt."),

    tarjeta("Modelo original: LACCA-UNet", [
        dcc.Markdown("**Idea.** Compara cada región del mapa de características con su reflejo respecto a un eje de simetría torácico "
                     "**aprendido**, con atención local y un residuo de asimetría."),
        dcc.Markdown(PSEUDO),
        html.H6("Diferencias respecto a la literatura"),
        html.Ul([html.Li("CE-Net (2020): eje fijo a la línea espinal, para detección."),
                 html.Li("SymFormer (TPAMI 2024): simetría para detección y clasificación en TBX11K."),
                 html.Li("LACCA: eje aprendido y antisimetrizado, segmentación densa, equivarianza exacta al volteo.")]),
        html.H6("Pruebas unitarias (25, todas pasan)"), html.Ul([html.Li(p) for p in PRUEBAS]),
        dbc.Alert([html.B("Veredicto: no superioridad. "),
                   "LACCA-UNet es competitivo (Dice 0,5651 con 10 semillas) pero no supera de forma significativa al mejor baseline; "
                   "las ablaciones no muestran aporte significativo del eje aprendido ni del residuo. Es frágil ante ruido y desenfoque."],
                  color="warning"),
        html.Img(src=dash.get_asset_url("gradcam/lacca_unet.png"), style={"maxWidth": "100%"}),
        html.P("Grad-CAM precalculado (semilla 42).", className="small text-muted")]),

    tarjeta("Ficha del modelo (model card)", [dbc.Accordion([dbc.AccordionItem(
        dcc.Markdown((ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")), title="Ver ficha")], start_collapsed=True)]),
])


def _leer(contenido, ejemplo):
    if contenido:
        raw = base64.b64decode(contenido.split(",", 1)[1])
        return inf.validar_y_preparar(raw), None
    if ejemplo:
        d = ROOT / "assets" / "ejemplos"
        img = np.asarray(Image.open(d / f"{ejemplo}.png").convert("L").resize((512, 512)), dtype=np.float32)
        gt = np.asarray(Image.open(d / f"{ejemplo}_gt.png").convert("L").resize((512, 512), Image.NEAREST)) > 0
        return img, gt
    raise inf.EntradaInvalida("Seleccione un ejemplo o suba una imagen.")


@callback(Output("p-out", "children"), Output("p-aviso", "children"), Input("p-go", "n_clicks"),
          State("p-modelo", "value"), State("p-ej", "value"), State("p-up", "contents"),
          State("p-umb", "value"), State("p-temp", "value"), prevent_initial_call=True)
def predecir(n, modelo, ejemplo, contenido, umbral, usar_t):
    try:
        img, gt = _leer(contenido, None if contenido else ejemplo)
        T = 1.0
        if usar_t and modelo in CON_CAL:
            T = float(tabla("calibracion").set_index("modelo").loc[modelo, "T"])
        prob, mask, inc, ms = inf.predecir(modelo, img, umbral, T)
    except inf.EntradaInvalida as e:
        return no_update, dbc.Alert(str(e), color="danger")
    except Exception as e:  # nunca detener la app
        return no_update, dbc.Alert(f"No se pudo ejecutar el modelo: {e}", color="danger")
    area = mask.mean()
    datos = [f"Latencia: {ms:.0f} ms", f"Área predicha: {area:.1%}", f"Prob. máxima: {prob.max():.2f}"]
    if gt is not None:
        datos.append(f"Dice frente al GT: {inf.dice(mask, gt):.3f}")
    if usar_t and modelo in CON_CAL:
        datos.append(f"Calibrada con T = {T:.2f}")
    sup = plots.superponer(img, mask, gt)
    fig = lambda a, t, **k: dcc.Graph(figure=plots.imagen(a, t, **k), config={"displayModeBar": False})
    return html.Div([
        dbc.Row([dbc.Col(fig(sup, "Máscara predicha (rojo)" + (" y GT (azul)" if gt is not None else "")), md=4),
                 dbc.Col(fig(prob, "Probabilidad de lesión", cmap="Reds", escala=(0, 1)), md=4),
                 dbc.Col(fig(inc, "Incertidumbre (bits)", cmap="Viridis", escala=(0, 1)), md=4)]),
        html.P(" · ".join(datos), className="fw-bold"),
        html.Div("Las zonas de alta incertidumbre rodean los bordes de la lesión; allí conviene la revisión humana.", className="insight")]), None


@callback(Output("d-caso", "figure"), Output("d-conf", "figure"), Output("d-cal", "children"),
          Output("d-rob", "figure"), Output("d-cam", "children"), Input("d-modelo", "value"))
def diagnostico(m):
    import plotly.graph_objects as go
    vacio = go.Figure().update_layout(title="Sin datos para este modelo", height=420)
    caso = plots.dice_por_caso(tabla("dice_por_imagen"), m)
    conf = plots.confiabilidad(tabla("confiabilidad"), m) if m in CON_CAL else vacio
    cal = ""
    if m in CON_CAL:
        c = tabla("calibracion").set_index("modelo").loc[m]
        cal = html.P(f"ECE {c.ece_antes:.4f} → {c.ece_despues:.4f} · Brier {c.brier_antes:.4f} → {c.brier_despues:.4f} · T = {c['T']:.2f}")
    rob = plots.robustez(tabla("robustez"), [m]) if m in CON_CAL else vacio
    cam = html.Img(src=dash.get_asset_url(f"gradcam/{m}.png"), style={"maxWidth": "100%"}) if m in CON_CAM else html.P("Grad-CAM disponible solo para 5 modelos.", className="text-muted")
    return caso, conf, cal, rob, cam
