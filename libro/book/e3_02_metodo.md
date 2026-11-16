# Método: LACCA y LACCA-UNet

## Formulación

Sea $F\in\mathbb R^{C\times H\times W}$. El eje de simetría se parametriza por $(\theta,d)$; se **antisimetriza**
promediando sobre el grupo de flips para que la reflexión sea exactamente equivariante:

$$(\theta,d)=\tfrac12\big[(\theta_0,d_0)-\sigma(\theta_0,d_0)\big]\quad (1)$$

El reflejo $\tilde F=\mathcal R_{\theta,d}F$ se obtiene con `grid_sample` (padding por reflexión) y máscara de
validez $M$. La atención local (ventana de radio $r$) del mapa original contra su espejo es

$$A_p=\sum_{q\in\mathcal N_r(p)}\operatorname{softmax}_q\!\Big(\frac{\hat q_p^\top \hat k_q}{\tau}+b_{p-q}\Big)\,W_vF_q\quad (2)$$

con $q,k$ normalizados ($\epsilon$), temperatura $\tau$ y sesgo posicional simétrico $b$. El residuo de asimetría
y la salida con compuerta ReZero son

$$D=(W_vF-A)\odot M,\qquad F'=F+\gamma\,\psi([F,D]),\ \gamma_0=0\quad (3)$$

LACCA se inserta en las etapas 3 y 4 del encoder ResNet34 de un U-Net.

## Pseudocódigo

```
LACCA(F):
  (θ,d) ← antisimetrizar(θ0,d0)             # ec. (1)
  F~, M ← reflejar(F, θ, d)                 # grid_sample, reflexión
  q,k   ← normalizar(Wq F, Wk F~)
  A     ← atención_local(q,k,Wv F~, r, τ, b) # ec. (2)
  D     ← (Wv F − A) ⊙ M                    # ec. (3)
  return F + γ · ψ(concat(F, D))
```

## Propiedades (verificadas con pruebas)

1. **Equivarianza exacta al flip** de la entrada (error numérico ≤ 1e-6).
2. **Identidad en γ = 0:** $F'=F$ al inicio del entrenamiento, no degrada el preentrenamiento.
3. **Cota de Lipschitz puntual** $\|F'(x)-F'(y)\|\le(1+|\gamma|L_\psi)\|x-y\|$ (solo con eje fijo; con eje aprendido se
   declara como limitación).

## Complejidad

Atención global: $O((HW)^2 d)$. LACCA local: $O(HW\,K\,d)$ con $K=(2r+1)^2$; para $r=1$, $K=9$.
En 512×512 el costo adicional medido es de unos pocos centésimos de GFLOP frente a ~61.7 GFLOPs del U-Net
(ver `resultados/e3/onnx/perfil_modelos.json`).

## Ablaciones

Variantes en el mismo código: `axis_mode=fixed`, `asym_mode=direct`, `attn_mode=global`.
