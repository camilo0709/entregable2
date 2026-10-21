"""
================================================================================
REGISTRO DE MODELOS - BENCHMARK ENTREGABLE 2
================================================================================

PROPÓSITO:
    Un único punto de verdad para los 10 modelos del benchmark más los
    3 baselines. Todo el entrenamiento importa de aquí, de modo que el
    protocolo (input 1 canal, salida binaria, misma resolución) es idéntico
    para todos y no puede desviarse por accidente.

PROTOCOLO COMÚN (no negociable):
    - Entrada:  1 canal (radiografía en escala de grises), 512x512
    - Salida:   1 canal logit (binario; sigmoid en inferencia)
    - Pérdida:  BCEWithLogits + DiceLoss  (definida en entrenamiento.py)
    - Umbral:   0.5 sobre probabilidad     (margen de relevancia en metricas)

LOS 10 MODELOS (Ruta B - imágenes):
     1. UNet            (Ronneberger 2015)          CNN
     2. AttentionUNet   (Oktay 2018)                CNN + attention gates
     3. UNetPP          (Zhou 2018)                 CNN (dense skip)
     4. DeepLabV3P      (Chen 2018)                 CNN + ASPP
     5. NnUNet          (Isensee 2021)              CNN (autoconfig)
     6. TransUNet       (Chen 2021)                 CNN + Transformer
     7. SwinUNet        (Cao 2021)                  Swin Transformer
     8. UNetr           (Hatamizadeh 2022)          ViT
     9. SegFormer       (Xie 2021)                  MiT hierarchical
    10. MedNeXt         (Rong 2024) / SegResNet     ConvNeXt / residual

LOS 3 BASELINES (obligatorios, sin entrenamiento o trivial):
     B1. EmptyMask      -> predice máscara vacía       (cota inferior)
     B2. BBoxMask       -> usa el rectángulo TBX11K    (baseline propio)
     B3. UNetFuerte     -> mejor U-Net con tuning      (cota superior CNN)

USO:
    from modelos_tb import crear_modelo, listar_modelos, MODELOS
    m = crear_modelo("unet", in_channels=1, out_channels=1)
================================================================================
"""

import importlib
import traceback

# ============================================================================
# CATÁLOGO
# ============================================================================
# "backend" determina QUÉ librería lo construye.
#   smp       -> segmentation-models-pytorch  (instalado por defecto)
#   monai     -> monai                        (UNETR, SegResNet)
#   custom    -> implementación local en este archivo
#   nnunet    -> nnunetv2 (pipeline completo, entrenamiento especial)
#   lazily    -> dependencia externa pesada; se instala solo si se usa

MODELOS = {
    # ------------------------------------------------------------------ CNN
    "unet": {
        "nombre": "U-Net",
        "backend": "smp",
        "constructor": "Unet",
        "params": {"encoder_name": "resnet34", "encoder_weights": None},
        "familia": "CNN",
        "paper": "Ronneberger et al., MICCAI 2015 (arXiv:1505.04597)",
        "entrenable": True,
    },
    "attention_unet": {
        "nombre": "Attention U-Net",
        "backend": "smp",
        "constructor": "Unet",
        # smp implementa Attention U-Net via attention_type (gates de Oktay)
        "params": {"encoder_name": "resnet34", "encoder_weights": None,
                   "attention_type": "scse"},
        "familia": "CNN + attention",
        "paper": "Oktay et al., MICCAI 2018 (arXiv:1804.03999)",
        "entrenable": True,
    },
    "unet_pp": {
        "nombre": "U-Net++",
        "backend": "smp",
        "constructor": "UnetPlusPlus",   # smp>=0.4 (antes "Unetplusplus")
        "params": {"encoder_name": "resnet34", "encoder_weights": None},
        "familia": "CNN (dense skip)",
        "paper": "Zhou et al., DLMIA 2018 (arXiv:1807.10165)",
        "entrenable": True,
    },
    "deeplabv3p": {
        "nombre": "DeepLabV3+",
        "backend": "smp",
        "constructor": "DeepLabV3Plus",
        "params": {"encoder_name": "resnet50", "encoder_weights": None},
        "familia": "CNN + ASPP",
        "paper": "Chen et al., CVPR 2018",
        "entrenable": True,
        "nota": "Requiere batch >= 2 en entrenamiento: la rama de image "
                "pooling del ASPP usa BN sobre 1x1 y con batch=1 falla.",
    },
    "nnunet": {
        "nombre": "nnU-Net",
        "backend": "nnunet",
        "familia": "CNN (autoconfig)",
        "paper": "Isensee et al., Nature Methods 2021 (doi:10.1038/s41592-020-01008-z)",
        "entrenable": True,
        # nnU-Net configura su propia red y sus propios parches; ver
        # entrenamiento_nnunet.py. Se marca aquí para que el driver lo trate
        # como caso especial y no intente construirlo con smp.
        "nota": "Pipeline propio: requiere dataset en formato nnU-Net.",
    },
    # ------------------------------------------------- CNN + Transformer
    "transunet": {
        "nombre": "TransUNet",
        "backend": "lazily",
        "import_target": "transunet",
        "familia": "CNN + Transformer",
        "paper": "Chen et al., arXiv:2102.04306 (MedIA 2024)",
        "entrenable": True,
        "nota": ("Implementación externa. Alternativa disponible sin "
                 "dependencia: usar smp.Unet con encoder transformer "
                 "(ver fallback en crear_modelo)."),
    },
    "swin_unet": {
        "nombre": "Swin-UNet",
        "backend": "lazily",
        "import_target": "swinunet",
        "familia": "Swin Transformer",
        "paper": "Cao et al., ECCV W. 2022 (arXiv:2105.05537)",
        "entrenable": True,
        "nota": ("Implementación externa. Fallback: smp.Unet con "
                 "encoder='mit_b2' (MiT, mismo family que SegFormer)."),
    },
    # ------------------------------------------------------- Transformer puro
    "unetr": {
        "nombre": "UNETR",
        "backend": "monai",
        "constructor": "UNETR",
        # MONAI >=1.5: pos_embed -> proj_type, residual_connection -> res_block
        "params": {"img_size": 512, "feature_size": 16,
                   "hidden_size": 768, "mlp_dim": 3072,
                   "num_heads": 12, "proj_type": "conv",
                   "norm_name": "instance", "res_block": True,
                   "spatial_dims": 2},
        "familia": "ViT",
        "paper": "Hatamizadeh et al., WACV 2022 (arXiv:2103.10504)",
        "entrenable": True,
    },
    "segformer": {
        "nombre": "SegFormer",
        "backend": "smp",
        "constructor": "Segformer",
        "params": {"encoder_name": "mit_b2", "encoder_weights": None},
        "familia": "MiT jerárquico",
        "paper": "Xie et al., NeurIPS 2021 (arXiv:2105.15203)",
        "entrenable": True,
    },
    # ----------------------------------------------------------- Moderno
    "mednext": {
        "nombre": "MedNeXt",
        "backend": "monai",
        "constructor": "SegResNet",
        # MedNeXt es un ConvNeXt para imagen médica; MONAI no lo incluye
        # todavía, así que se usa SegResNet (residual 3D/2D) como par más
        # cercano disponible por pip. Ver nota en el preregistro.
        "params": {"spatial_dims": 2, "init_filters": 32,
                   "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1),
                   "act": "relu", "norm": "batch"},
        "familia": "ConvNext / residual",
        "paper": "Roy et al., MICCAI 2023 (arXiv:2303.09975)",
        "entrenable": True,
        "nota": "Sustituto: SegResNet (Hatamizadeh 2018). Declarar en el libro.",
    },
    # ------------------------------------------------------------- Baselines
    "empty_mask": {
        "nombre": "Baseline: máscara vacía",
        "backend": "baseline",
        "familia": "Trivial",
        "entrenable": False,
        "predice": "vacio",
        "papel": "Cota inferior. Dice=0 en positivos; sirve de referencia.",
    },
    "bbox_mask": {
        "nombre": "Baseline: caja TBX11K",
        "backend": "baseline",
        "familia": "Trivial",
        "entrenable": False,
        "predice": "bbox",
        "papel": ("Baseline propio del proyecto. Usa el rectángulo del "
                  "anotador original sin refinamiento."),
    },
    "unet_fuerte": {
        "nombre": "Baseline: U-Net fuerte",
        "backend": "smp",
        "constructor": "Unet",
        "params": {"encoder_name": "resnet34", "encoder_weights": None},
        "familia": "CNN (tuned)",
        "entrenable": True,
        "papel": "Cota superior CNN. Mismo U-Net con búsqueda de hiperparámetros.",
        "nota": ("Se entrena con la mejor configuración hallada en val, "
                 "declarada en el preregistro antes de tocar el test."),
    },
}

# Orden en el que se reportan las Tablas A/B/C
ORDEN_BENCHMARK = [
    "unet", "attention_unet", "unet_pp", "deeplabv3p", "nnunet",
    "transunet", "swin_unet", "unetr", "segformer", "mednext",
]
ORDEN_BASELINES = ["empty_mask", "bbox_mask", "unet_fuerte"]

# Semillas por modelo (mínimo 5 exigido por el Entregable)
SEMILLAS = [42, 123, 777, 2024, 31415]


# ============================================================================
# CONSTRUCCIÓN
# ============================================================================

def _importar(nombre):
    """Importa un módulo devolviendo None si no está instalado."""
    try:
        return importlib.import_module(nombre)
    except Exception:
        return None


def _parametros_smp(constructor, in_ch, out_ch, params):
    """Construye el modelo SMP forzando canales de entrada/salida."""
    import segmentation_models_pytorch as smp
    cls = getattr(smp, constructor)
    p = dict(params)
    p["in_channels"] = in_ch
    p["out_channels"] = out_ch
    # sin pesos preentrenados: no hay red ImageNet para 1 canal de entrada
    p.setdefault("encoder_weights", None)
    return cls(**p)


def _parametros_monai(constructor, in_ch, out_ch, params):
    """Construye un modelo MONAI (UNETR / SegResNet)."""
    import monai
    cls = getattr(monai.networks.nets, constructor)
    p = dict(params)
    if constructor == "UNETR":
        p["in_channels"] = in_ch
        p["out_channels"] = out_ch
    elif constructor == "SegResNet":
        p["in_channels"] = in_ch
        p["out_channels"] = out_ch
    return cls(**p)


def _fallback_transformer(in_ch, out_ch, clave):
    """
    Sustituto sin dependencias externas para TransUNet / Swin-UNet.

    Usa un encoder transformer de SMP (MiT de SegFormer) sobre una cabeza
    U-Net. Conserva el espíritu del modelo (atención global) manteniendo el
    protocolo idéntico al resto. Se DEBE declarar en el preregistro.
    """
    import segmentation_models_pytorch as smp
    if clave == "transunet":
        enc = "resnet50"          # CNN + atención via decoder ATM (abajo)
        cls = smp.Unet
        p = {"encoder_name": enc, "encoder_weights": None,
             "decoder_attention_type": "scse"}
    else:  # swin_unet
        cls = smp.Unet
        p = {"encoder_name": "mit_b2", "encoder_weights": None}
    p["in_channels"] = in_ch
    p["out_channels"] = out_ch
    return cls(**p)


def crear_modelo(clave, in_channels=1, out_channels=1, verbose=True):
    """
    Devuelve (modelo, info) para la clave dada.

    Lanza RuntimeError con un mensaje accionable si falta la dependencia,
    en vez de fallar en mitad de una corrida de entrenamiento.
    """
    if clave not in MODELOS:
        raise KeyError(
            f"Modelo desconocido '{clave}'. "
            f"Disponibles: {sorted(MODELOS)}"
        )
    meta = MODELOS[clave]
    backend = meta["backend"]

    # ---------------------------------------------------------- baselines
    if backend == "baseline":
        raise RuntimeError(
            f"'{clave}' es un baseline sin red. No debe construirse: "
            f"se evalúa directamente contra el GT (ver metricas.py)."
        )

    # -------------------------------------------------------------- smp
    if backend == "smp":
        if _importar("segmentation_models_pytorch") is None:
            raise RuntimeError(
                "Falta 'segmentation-models-pytorch'.\n"
                "  Colab:  %pip install segmentation-models-pytorch\n"
                "  Local:  pip install segmentation-models-pytorch"
            )
        m = _parametros_smp(meta["constructor"], in_channels, out_channels,
                            meta.get("params", {}))
        if verbose:
            print(f"✓ {meta['nombre']} construido "
                  f"(encoder {meta.get('params', {}).get('encoder_name', '-')})")
        return m, meta

    # ------------------------------------------------------------- monai
    if backend == "monai":
        if _importar("monai") is None:
            raise RuntimeError(
                "Falta 'monai'.\n"
                "  Colab:  %pip install monai\n"
                "  Local:  pip install monai"
            )
        m = _parametros_monai(meta["constructor"], in_channels, out_channels,
                              meta.get("params", {}))
        if verbose:
            print(f"✓ {meta['nombre']} construido (MONAI {meta['constructor']})")
        return m, meta

    # ----------------------------------------------------------- nnunet
    if backend == "nnunet":
        raise RuntimeError(
            "nnU-Net es un pipeline completo, no una red que se construya "
            "aquí.\nUsa entrenamiento_nnunet.py (dataset en formato nnU-Net)."
        )

    # ----------------------------------------------------------- lazily
    if backend == "lazily":
        mod = _importar(meta["import_target"])
        if mod is not None and hasattr(mod, "ViT"):
            if verbose:
                print(f"✓ {meta['nombre']} construido (implementación externa)")
            return mod.ViT(in_chans=in_channels, class_num=out_channels), meta
        if verbose:
            print(f"⚠ {meta['nombre']}: dependencia externa no disponible.")
            print(f"  Usando sustituto declarado en el preregistro.")
        m = _fallback_transformer(in_channels, out_channels, clave)
        return m, meta

    raise RuntimeError(f"Backend desconocido: {backend}")


# ============================================================================
# INVENTARIO / VERIFICACIÓN
# ============================================================================

def listar_modelos(incluir_baselines=True):
    """Devuelve las claves en orden de reporte."""
    claves = list(ORDEN_BENCHMARK)
    if incluir_baselines:
        claves += list(ORDEN_BASELINES)
    return claves


def verificar_entorno():
    """
    Diagnóstico de qué modelos son realmente entrenables AHORA.

    Devuelve dict {clave: (disponible: bool, detalle: str)}.
    Se ejecuta al inicio de cada corrida para dejar constancia en el log.
    """
    deps = {
        "smp": _importar("segmentation_models_pytorch") is not None,
        "monai": _importar("monai") is not None,
        "nnunet": _importar("nnunetv2") is not None,
        "transunet": _importar("transunet") is not None,
        "swinunet": _importar("swinunet") is not None,
    }
    resultado = {}
    for clave in listar_modelos():
        meta = MODELOS[clave]
        b = meta["backend"]
        if b == "baseline":
            resultado[clave] = (True, "baseline (sin red)")
        elif b == "smp":
            resultado[clave] = (deps["smp"],
                                "ok" if deps["smp"] else "falta smp")
        elif b == "monai":
            resultado[clave] = (deps["monai"],
                                "ok" if deps["monai"] else "falta monai")
        elif b == "nnunet":
            resultado[clave] = (deps["nnunet"],
                                "ok" if deps["nnunet"] else "falta nnunetv2")
        elif b == "lazily":
            ok = deps[meta["import_target"]] if meta["import_target"] in deps else False
            if ok:
                resultado[clave] = (True, "implementación externa")
            else:
                # el sustituto declarado se construye con smp
                resultado[clave] = (deps["smp"],
                                    "sustituto declarado (vía smp)"
                                    if deps["smp"] else "sustituto necesita smp (falta)")
        else:
            resultado[clave] = (False, f"backend {b} desconocido")
    return resultado, deps


def resumen_entorno():
    """Imprime y devuelve el diagnóstico de forma legible."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    diag, deps = verificar_entorno()
    print("=" * 74)
    print("DIAGNÓSTICO DE ENTORNO - MODELOS DEL BENCHMARK")
    print("=" * 74)
    for k, (ok, det) in diag.items():
        marca = "OK  " if ok else "NO  "
        print(f"  [{marca}] {k:16s} {det}")
    print("-" * 74)
    print("Dependencias:", {k: ("✓" if v else "✗") for k, v in deps.items()})
    n_ok = sum(1 for ok, _ in diag.values() if ok)
    print(f"Modelos disponibles: {n_ok}/{len(diag)}")
    return diag, deps


if __name__ == "__main__":
    resumen_entorno()
    print()
    print("Detalle de cada modelo:")
    for clave in listar_modelos():
        m = MODELOS[clave]
        print(f"\n  [{clave}] {m['nombre']}  ({m['familia']})")
        if "paper" in m:
            print(f"      paper:    {m['paper']}")
        if "params" in m:
            enc = m["params"].get("encoder_name", "-")
            print(f"      encoder:  {enc}")
        if "nota" in m:
            print(f"      nota:     {m['nota']}")
        if "papel" in m:
            print(f"      papel:    {m['papel']}")
    print(f"\n  Semillas por modelo: {SEMILLAS}")
    print(f"  Total de corridas planificadas: "
          f"{len(ORDEN_BENCHMARK) + len(ORDEN_BASELINES) - 2} entrenables "
          f"x {len(SEMILLAS)} semillas")
