"""Funciones que construyen las figuras (plantilla Plotly unificada)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

PALETA = ["#1F4E79", "#E07B39", "#2A9D8F", "#7A5C99", "#8C8C8C"]
pio.templates["tb"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Arial", size=13), colorway=PALETA,
    paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=50, r=20, t=50, b=45),
    xaxis=dict(gridcolor="#E6E6E6"), yaxis=dict(gridcolor="#E6E6E6")))
pio.templates.default = "tb"


def barras_dice(df: pd.DataFrame):
    d = df.sort_values("dice_media")
    f = px.bar(d, x="dice_media", y="Modelo", orientation="h", error_x="dice_sd", color="familia",
               title="Dice de test (positivas) · media ± desviación entre semillas",
               labels={"dice_media": "Dice", "Modelo": ""}, hover_data={"n_semillas": True})
    f.update_layout(height=520, legend_title_text="Familia")
    return f


def pareto(df: pd.DataFrame):
    f = px.scatter(df, x="latencia_ms_ort_cpu", y="dice_media", color="familia", text="Modelo",
                   size="parametros", size_max=34, title="Frontera desempeño–costo (CPU, 1 imagen 512×512)",
                   labels={"latencia_ms_ort_cpu": "Latencia en CPU (ms)", "dice_media": "Dice de test"})
    f.update_traces(textposition="top center", textfont_size=10)
    f.update_layout(height=480, legend_title_text="Familia")
    return f


def robustez(df: pd.DataFrame, modelos):
    d = df[df.modelo.isin(modelos)]
    f = px.line(d, x="perturbacion", y="dice_test_pos", color="modelo", markers=True,
                title="Robustez ante perturbaciones (semilla 42)",
                labels={"dice_test_pos": "Dice (positivas)", "perturbacion": ""})
    f.update_layout(height=440); return f


def confiabilidad(df: pd.DataFrame, modelo: str):
    d = df[df.modelo == modelo]; d = d[d.n_antes > 0]
    f = go.Figure()
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta", line=dict(dash="dash", color="#8C8C8C")))
    f.add_trace(go.Scatter(x=d.conf_antes, y=d.acc_antes, mode="lines+markers", name="Antes (T=1)"))
    d2 = d[d.n_despues > 0]
    f.add_trace(go.Scatter(x=d2.conf_despues, y=d2.acc_despues, mode="lines+markers", name="Después (escalado de temperatura)"))
    f.update_layout(title="Diagrama de confiabilidad (píxeles)", xaxis_title="Confianza", yaxis_title="Frecuencia observada", height=420)
    return f


def dice_por_caso(df: pd.DataFrame, modelo: str):
    d = df[["id", modelo]].rename(columns={modelo: "dice"}).sort_values("dice")
    f = px.bar(d, x="id", y="dice", title="Dice por caso de test (media entre semillas)", labels={"id": "Caso", "dice": "Dice"})
    f.update_layout(height=380, xaxis_showticklabels=False); return f


def hist_mask(df: pd.DataFrame, color):
    f = px.histogram(df, x="mask_frac", color=color, nbins=30, barmode="overlay", opacity=0.75,
                     title="Fracción de píxeles de lesión por imagen", labels={"mask_frac": "Lesión / imagen"})
    f.update_layout(height=380); return f


def splits(df: pd.DataFrame):
    f = px.bar(df, x="split", y="n", text="n", title="Imágenes por partición", labels={"split": "", "n": "Imágenes"})
    f.update_layout(height=340); return f


def imagen(arr, titulo="", cmap="gray", escala=None):
    f = px.imshow(arr, color_continuous_scale=cmap, zmin=None if escala is None else escala[0],
                  zmax=None if escala is None else escala[1], title=titulo)
    f.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=escala is not None)
    f.update_xaxes(visible=False); f.update_yaxes(visible=False); return f


def superponer(img, mascara, gt=None):
    """Imagen en gris con la máscara predicha (rojo) y, opcionalmente, el GT (contorno azul)."""
    g = (img - img.min()) / (np.ptp(img) + 1e-6)
    rgb = np.stack([g, g, g], -1)
    rgb[mascara] = 0.55 * rgb[mascara] + 0.45 * np.array([0.9, 0.15, 0.15])
    if gt is not None:
        from scipy.ndimage import binary_erosion
        borde = gt & ~binary_erosion(gt, iterations=2)
        rgb[borde] = np.array([0.1, 0.4, 1.0])
    return (rgb * 255).astype(np.uint8)
