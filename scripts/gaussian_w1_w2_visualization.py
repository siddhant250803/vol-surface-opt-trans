# pip install numpy plotly pot

import numpy as np
import plotly.graph_objects as go
import ot  # POT: Python Optimal Transport


# ----------------------------
# Density: 2D Gaussian on a grid
# ----------------------------
def gaussian_pdf(points, mean, cov):
    mean = np.asarray(mean).reshape(2)
    cov = np.asarray(cov).reshape(2, 2)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    d = points - mean
    quad = np.einsum("...i,ij,...j->...", d, inv, d)
    return norm * np.exp(-0.5 * quad)


def make_grid(n=45, lim=4.0):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return x, y, X, Y, pts


def normalize_mass(a):
    a = np.clip(a, 0, None)
    s = a.sum()
    return a / s if s > 0 else a


# ----------------------------
# OT: Sinkhorn + barycentric projection map
# ----------------------------
def sinkhorn_barycentric_map(x_pts, y_pts, a, b, cost_matrix, reg=5e-2, numItermax=4000):
    """
    Returns:
      pi: transport plan (n x m)
      T: barycentric map for x -> E[Y|X=x], shape (n,2)
    """
    pi = ot.sinkhorn(a, b, cost_matrix, reg=reg, numItermax=numItermax)
    row_sum = pi.sum(axis=1, keepdims=True)  # should be close to a[:,None]
    row_sum = np.maximum(row_sum, 1e-15)
    T = (pi @ y_pts) / row_sum
    return pi, T


def cost_W2_sqeuclid(X, Y):
    # squared Euclidean cost
    # X: (n,2), Y: (m,2)
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    return X2 + Y2 - 2.0 * (X @ Y.T)


def cost_W1_L1(X, Y):
    # L1 cost
    # broadcasting: (n,1,2) - (1,m,2) -> (n,m,2)
    return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)


# ----------------------------
# Plotly: surfaces + arrow segments
# ----------------------------
def plot_ot_html(
    html_path,
    title,
    x_grid, y_grid, X, Y,
    P, Q,
    src_pts, T_pts,
    arrow_mask,
    mean_src,
    mean_tgt,
    arrow_stride=3,
    zscale=1.0,
    png_path=None,
):
    """
    - P, Q are (n,n) densities on grid for surfaces (will be normalized for display)
    - arrows: from src_pts -> T_pts for points selected by arrow_mask
    - mean_src, mean_tgt: (2,) arrays for source/target Gaussian means (markers)
    """
    # normalize surfaces for display
    Pn = P / (P.max() + 1e-15)
    Qn = Q / (Q.max() + 1e-15)

    # build arrow line segments (downsample + filter)
    idx = np.where(arrow_mask)[0]
    idx = idx[::arrow_stride]

    xs = src_pts[idx, 0]
    ys = src_pts[idx, 1]
    xt = T_pts[idx, 0]
    yt = T_pts[idx, 1]

    # z at start/end: sample from normalized surfaces (nearest neighbor on grid)
    def z_from_surface(Z, px, py):
        xi = np.searchsorted(x_grid, px).clip(1, len(x_grid) - 1)
        yi = np.searchsorted(y_grid, py).clip(1, len(y_grid) - 1)
        xi = np.where(np.abs(x_grid[xi] - px) < np.abs(x_grid[xi - 1] - px), xi, xi - 1)
        yi = np.where(np.abs(y_grid[yi] - py) < np.abs(y_grid[yi - 1] - py), yi, yi - 1)
        return Z[yi, xi]

    z0 = zscale * z_from_surface(Pn, xs, ys)
    z1 = zscale * z_from_surface(Qn, xt, yt)

    # create polyline coordinates with None separators
    Xline, Yline, Zline = [], [], []
    for i in range(len(xs)):
        Xline += [xs[i], xt[i], None]
        Yline += [ys[i], yt[i], None]
        Zline += [z0[i], z1[i], None]

    # z for mean markers (sample from surfaces)
    z_src = float(zscale * z_from_surface(Pn, np.array([mean_src[0]]), np.array([mean_src[1]]))[0])
    z_tgt = float(zscale * z_from_surface(Qn, np.array([mean_tgt[0]]), np.array([mean_tgt[1]]))[0])

    fig = go.Figure()

    # Source: blue tones, slightly more opaque
    fig.add_trace(go.Surface(
        x=X, y=Y, z=zscale * Pn,
        name="Source (μ)",
        colorscale="Blues",
        opacity=0.75,
        showscale=False,
    ))
    # Target: orange/coral tones, distinct
    fig.add_trace(go.Surface(
        x=X, y=Y, z=zscale * Qn,
        name="Target (ν)",
        colorscale="Oranges",
        opacity=0.7,
        showscale=False,
    ))

    # Mean markers: source = blue sphere, target = orange sphere
    fig.add_trace(go.Scatter3d(
        x=[mean_src[0]], y=[mean_src[1]], z=[z_src],
        mode="markers+text",
        marker=dict(size=10, color="#1f77b4", symbol="diamond", line=dict(width=2, color="white")),
        text=["μ"], textposition="top center", textfont=dict(size=14, color="#1f77b4"),
        name="Source mean",
    ))
    fig.add_trace(go.Scatter3d(
        x=[mean_tgt[0]], y=[mean_tgt[1]], z=[z_tgt],
        mode="markers+text",
        marker=dict(size=10, color="#ff7f50", symbol="diamond", line=dict(width=2, color="white")),
        text=["ν"], textposition="top center", textfont=dict(size=14, color="#ff7f50"),
        name="Target mean",
    ))

    # Transport arrows: gradient-like (darker blue)
    fig.add_trace(go.Scatter3d(
        x=Xline, y=Yline, z=Zline,
        mode="lines",
        name="Transport map (source → target)",
        line=dict(width=3, color="#2ca02c"),
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="density (normalized)",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote: {html_path}")
    if png_path:
        try:
            fig.write_image(png_path)
            print(f"Wrote: {png_path}")
        except Exception:
            pass  # kaleido not installed


# ----------------------------
# Main: build distributions, compute W2 + W1 plans, render HTML
# ----------------------------
if __name__ == "__main__":
    # Grid
    xg, yg, X, Y, pts = make_grid(n=55, lim=4.0)

    # Define source/target (edit these to match your narrative)
    m_p = np.array([-1.0, 0.0])
    S_p = np.array([[0.9, 0.30],
                    [0.30, 0.6]])

    m_q = np.array([1.2, 0.3])
    S_q = np.array([[0.5, -0.20],
                    [-0.20, 1.1]])

    # Discrete masses on the same grid
    p_mass = gaussian_pdf(pts, m_p, S_p)
    q_mass = gaussian_pdf(pts, m_q, S_q)
    a = normalize_mass(p_mass)
    b = normalize_mass(q_mass)

    # Surfaces (for display)
    Psurf = p_mass.reshape(X.shape)
    Qsurf = q_mass.reshape(X.shape)

    # Arrow mask: only draw where p is significant (avoid tail hairball)
    # top ~25% of p mass values (tune)
    thr = np.quantile(p_mass, 0.75)
    arrow_mask = (p_mass >= thr)

    # Costs
    C2 = cost_W2_sqeuclid(pts, pts)
    C1 = cost_W1_L1(pts, pts)

    # Sinkhorn (entropic regularization)
    # If arrows still messy, increase reg a bit and/or raise quantile threshold.
    reg_w2 = 7e-2
    reg_w1 = 7e-2

    _, T_w2 = sinkhorn_barycentric_map(pts, pts, a, b, C2, reg=reg_w2)
    _, T_w1 = sinkhorn_barycentric_map(pts, pts, a, b, C1, reg=reg_w1)

    out_dir = "outputs/report/ot_findings"
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Render HTMLs + PNGs (PNG for README; requires kaleido)
    plot_ot_html(
        html_path=f"{out_dir}/ot_w2_map.html",
        title="W2 Map: Source μ → Target ν (Sinkhorn barycentric; cost = ‖x−y‖²)",
        x_grid=xg, y_grid=yg, X=X, Y=Y,
        P=Psurf, Q=Qsurf,
        src_pts=pts, T_pts=T_w2,
        arrow_mask=arrow_mask,
        mean_src=m_p, mean_tgt=m_q,
        arrow_stride=2,
        zscale=1.0,
        png_path=f"{out_dir}/ot_w2_map.png",
    )

    plot_ot_html(
        html_path=f"{out_dir}/ot_w1_map.html",
        title="W1 Map: Source μ → Target ν (Sinkhorn barycentric; cost = ‖x−y‖₁)",
        x_grid=xg, y_grid=yg, X=X, Y=Y,
        P=Psurf, Q=Qsurf,
        src_pts=pts, T_pts=T_w1,
        arrow_mask=arrow_mask,
        mean_src=m_p, mean_tgt=m_q,
        arrow_stride=2,
        zscale=1.0,
        png_path=f"{out_dir}/ot_w1_map.png",
    )
