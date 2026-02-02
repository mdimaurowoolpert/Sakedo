# GNSS Multi‑Point Dashboard (Streamlit) 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as plc
import itertools, io, pickle, gzip, datetime as dt
import re
import tempfile, os, imageio.v2 as imageio


# ‑‑‑ Reference coordinates - Basetime platform
INIT_XYZ = {
    "NW": np.array([309.82, 1090.36, 271.18]),
    "SW": np.array([293.27, 1052.34, 271.23]),
    "SE": np.array([429.08,  991.42, 271.17]),
    "NE": np.array([446.97, 1031.52, 271.24]),
}
ORDER = ["NW", "SW", "SE", "NE"]            
XY0   = np.vstack([INIT_XYZ[p][:2] for p in ORDER])  

# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
MAX_POINTS = 4
PLOT_H     = 500
VARS       = ["Easting", "Northing", "Height"]
DARK_STYLE   = True
MA_OPACITY   = 1.0 if DARK_STYLE else 0.6
MA_WIDTH     = 3   if DARK_STYLE else 1

st.set_page_config(page_title="GNSS Dashboard", layout="wide")
st.title("📡 GNSS Multi‑Point Dashboard (With RBM)")

# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_merge(en_b, ht_b):
    en = pd.read_csv(en_b)
    ht = pd.read_csv(ht_b)
    en = en.rename(columns={en.columns[0]: "Timestamp", en.columns[1]: "Northing", en.columns[3]: "Easting"})
    ht = ht.rename(columns={ht.columns[0]: "Timestamp", ht.columns[1]: "Height", ht.columns[3]: "Temperature"})
    en.Timestamp = pd.to_datetime(en.Timestamp, errors="coerce")
    ht.Timestamp = pd.to_datetime(ht.Timestamp, errors="coerce")
    return en.merge(ht, on="Timestamp").sort_values("Timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def save_session(obj):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        pickle.dump(obj, gz)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def load_session(b):
    with gzip.GzipFile(fileobj=io.BytesIO(b), mode="rb") as gz:
        return pickle.load(gz)

# Moving‑average helpers

def calc_sample_ma(df: pd.DataFrame, n: int):
    out = df.copy(); out[VARS] = out[VARS].rolling(n, min_periods=1).mean(); return out

def calc_time_ma(hr_df: pd.DataFrame, hours: int, mask_real: pd.Series):
    tmp = hr_df.set_index("Timestamp"); tmp[VARS] = tmp[VARS].rolling(f"{hours}H").mean(); out = tmp.reset_index(); out.loc[~mask_real.values, VARS] = np.nan; return out

# 2‑D rigid‑body transform

def _rigid_transform_2d(X: np.ndarray, Y: np.ndarray):
    Xc = X - X.mean(axis=0); Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    t = Y.mean(axis=0) - X.mean(axis=0) @ R
    return R, t

def apply_yaxis(fig, cfg):
    ymin, ymax, dtick = cfg; kwargs = {}
    if ymin is not None and ymax is not None: kwargs["range"] = [ymin, ymax]
    if dtick is not None: kwargs["dtick"] = dtick
    fig.update_yaxes(**kwargs, tickformat=".3f")

# ---------------------------------------------------------
# Session / data upload
# ---------------------------------------------------------
points = {}
with st.sidebar.expander("🚀 Quick Session"):
    qf = st.file_uploader("Load .gnss", type="gnss")
    if qf:
        try:
            points = load_session(qf.read())
            st.success("Session loaded ✓")
        except Exception as e:
            st.error(e)

if not points:
    st.sidebar.header("🔍 Upload point CSVs")
    DEFAULT_NAMES = ["SE", "NE", "NW", "SW"]
    for i in range(1, MAX_POINTS + 1):
        with st.sidebar.expander(f"Point {i}"):
            default_name = DEFAULT_NAMES[i - 1] if i <= len(DEFAULT_NAMES) else f"P{i}"
            pname = st.text_input("Name", default_name, key=f"name{i}")
            f_en  = st.file_uploader("E/N CSV", type="csv", key=f"en{i}")
            f_ht  = st.file_uploader("H/T CSV", type="csv", key=f"ht{i}")
            if f_en and f_ht:
                try:
                    points[pname] = load_and_merge(f_en, f_ht)
                except Exception as e:
                    st.error(e)

if not points:
    st.info("Upload data or load a session to begin.")
    st.stop()

# ---------------------------------------------------------
# Sidebar – common controls (apply to both pages)
# ---------------------------------------------------------
sel_pts   = st.sidebar.multiselect("Points", list(points), default=list(points))
show_temp = st.sidebar.checkbox("Show temperature", False)

with st.sidebar.expander("📊 Moving Averages"):
    use_ma1   = st.checkbox("Sample MA", value=False)
    n_samples = st.slider("N samples", 2, 200, 5, disabled=not use_ma1)
    use_ma2   = st.checkbox("Time‑window MA", value=True)
    n_hours   = st.slider("N hours", 1, 168, 48, disabled=not use_ma2)

with st.sidebar.expander("🧮 Zero‑reference"):
    offset_on = st.checkbox("Enable zero‑reference")
    null_time = None
    if offset_on:
        gmin = min(df.Timestamp.min() for df in points.values())
        null_time = dt.datetime.combine(
            st.date_input("Date", gmin.date()),
            st.time_input("Time", gmin.time())
        )

with st.sidebar.expander("📐 Y-axis scales / grid"):
    st.caption("Leave any box blank for automatic scaling / grid")
    DEFAULT_Y = {
        "Main": {
        "Easting": (-0.02, 0.02, 0.002),
        "Northing": (-0.02, 0.02, 0.002),
        "Height": (-0.04, 0.04, 0.002),
        },
        "CommonΔ": {
        "Easting": (-0.01, 0.01, 0.001),
        "Northing": (-0.01, 0.01, 0.001),
        "Height": (-0.01, 0.01, 0.001),
        },
        "RBMΔ": {
        "Easting": (-0.01, 0.01, 0.001),
        "Northing": (-0.01, 0.01, 0.001),
        "Height": (-0.01, 0.01, 0.001),
        },
        # "Compare" left empty → automated
    }
    y_cfg = {}
    for grp in ["Main", "CommonΔ", "Compare", "RBMΔ"]:
        st.markdown(f"**{grp} plots**")
        y_cfg[grp] = {}
        for v in VARS:
            c1, c2, c3 = st.columns(3)
            d = DEFAULT_Y.get(grp, {}).get(v, (None, None, None))
            vmin_str = "" if d[0] is None else f"{d[0]:.3f}"
            vmax_str = "" if d[1] is None else f"{d[1]:.3f}"
            dtick_str= "" if d[2] is None else f"{d[2]:.3f}"

            ymin_txt = c1.text_input(f"{v} min", key=f"{grp}_{v}_min", value=vmin_str)
            ymax_txt = c2.text_input(f"{v} max", key=f"{grp}_{v}_max", value=vmax_str)
            dtick_txt= c3.text_input(f"{v} grid", key=f"{grp}_{v}_dtick", value=dtick_str)
            try: ymin = float(ymin_txt) if ymin_txt.strip() else None
            except ValueError: ymin = None
            try: ymax = float(ymax_txt) if ymax_txt.strip() else None
            except ValueError: ymax = None
            try: dtick = float(dtick_txt) if dtick_txt.strip() else None
            except ValueError: dtick = None
            if (ymin is not None and ymax is not None) and ymin >= ymax:
                st.warning(f"{grp} {v}: min must be < max — ignored"); ymin = ymax = None
            y_cfg[grp][v] = (ymin, ymax, dtick)

with st.sidebar.expander("🗓️ X‑axis start"):
    use_xstart = st.checkbox("Set start date for all plots")
    x_start = st.date_input("Start date", value=min(df.Timestamp.min() for df in points.values()).date(), disabled=not use_xstart)

legend_cfg = dict(orientation="h", yanchor="top", y=-0.28)

# ---------------------------------------------------------
# Data preparation (shared by both pages)
# ---------------------------------------------------------
base = {p: points[p].copy() for p in sel_pts}
if offset_on and null_time:
    for df in base.values():
        idx = (df.Timestamp - null_time).abs().idxmin(); df[VARS] -= df.loc[idx, VARS].values

ma1, ma2 = {}, {}
for p, df in base.items():
    hr = df.set_index("Timestamp").resample("1H").mean(); mask_r = hr[VARS[0]].notna(); hr[VARS] = hr[VARS].apply(pd.to_numeric, errors='coerce'); hr.interpolate("time", inplace=True); hr.reset_index(inplace=True)
    ma1[p] = calc_sample_ma(df, n_samples) if use_ma1 else None
    ma2[p] = calc_time_ma(hr, n_hours, mask_r) if use_ma2 else None

# Arithmetic common & Δ (merged) ------------------------------------------------
if len(sel_pts) > 1:
    merged = None
    for p in sel_pts:
        src = ma2[p] if use_ma2 else base[p].set_index("Timestamp").resample("1H").mean().interpolate("time").reset_index()
        tmp = src[["Timestamp"] + VARS].copy(); tmp.columns = ["Timestamp"] + [f"{p}_{v[0]}" for v in VARS]
        merged = tmp if merged is None else merged.merge(tmp, on="Timestamp", how="outer")
    merged.sort_values("Timestamp", inplace=True); merged.set_index("Timestamp").interpolate("time").reset_index(inplace=True)
    common = pd.DataFrame({"Timestamp": merged.Timestamp});
    for v in VARS: common[v] = merged.filter(regex=f"_{v[0]}$").mean(axis=1)
    diff = {p: pd.DataFrame({"Timestamp": merged.Timestamp, **{v: merged[f"{p}_{v[0]}"] - common[v] for v in VARS}}) for p in sel_pts}
else:
    merged = None; common, diff = None, {}

if use_ma1: diff_ma1 = {p: calc_sample_ma(df, n_samples) for p, df in diff.items()}
if use_ma2:
    diff_ma2 = {}
    for p, df in diff.items():
        hr = df.set_index("Timestamp").resample("1H").mean(); mask = hr[VARS[0]].notna(); hr[VARS] = hr[VARS].apply(pd.to_numeric, errors='coerce'); hr.interpolate("time", inplace=True); hr.reset_index(inplace=True)
        diff_ma2[p] = calc_time_ma(hr, n_hours, mask)
else:
    diff_ma2 = {}

# Rigid‑Body residuals -----------------------------------------------------------
diff_rbm, rbm_common = {}, None
if merged is not None and set(ORDER).issubset(sel_pts):
    diff_rbm = {p: pd.DataFrame({"Timestamp": merged.Timestamp, **{v: np.nan for v in VARS}}) for p in ORDER}
    rbm_common = pd.DataFrame({"Timestamp": merged.Timestamp, "E_trans": np.nan, "N_trans": np.nan, "H_plane": np.nan})
    A0 = np.c_[np.ones(4), XY0]
    for i, row in merged.iterrows():
        E_abs, N_abs, H_abs = [], [], []
        for p in ORDER:
            dE, dN, dH = row[f"{p}_E"], row[f"{p}_N"], row[f"{p}_H"]
            if pd.isna(dE) or pd.isna(dN) or pd.isna(dH): break
            E_abs.append(float(dE) + INIT_XYZ[p][0]); N_abs.append(float(dN) + INIT_XYZ[p][1]); H_abs.append(float(dH) + INIT_XYZ[p][2])
        else:
            Y = np.column_stack([E_abs, N_abs]); Z = np.array(H_abs)
            R, t = _rigid_transform_2d(XY0, Y); Yfit = XY0 @ R + t; res2d = Y - Yfit
            coeffs, *_ = np.linalg.lstsq(A0, Z, rcond=None); resZ = Z - (A0 @ coeffs)
            for j, p in enumerate(ORDER):
                diff_rbm[p].loc[i, ["Easting", "Northing"]] = res2d[j]; diff_rbm[p].loc[i, "Height"] = resZ[j]
            rbm_common.loc[i, ["E_trans", "N_trans"]] = t; rbm_common.loc[i, "H_plane"] = Z.mean()
    valid_first = [df.dropna(subset=VARS).index.min() for df in diff_rbm.values() if not df.dropna(subset=VARS).empty]
    if valid_first:
        first_idx = min(valid_first)
        for p in ORDER: diff_rbm[p][VARS] -= diff_rbm[p].loc[first_idx, VARS].values

if diff_rbm:
    diff_rbm_ma1 = {p: calc_sample_ma(df, n_samples) for p, df in diff_rbm.items()} if use_ma1 else {}
    diff_rbm_ma2 = {}
    if use_ma2:
        for p, df in diff_rbm.items():
            hr = df.set_index("Timestamp").resample("1H").mean(); mask = hr[VARS[0]].notna(); hr[VARS] = hr[VARS].apply(pd.to_numeric, errors='coerce'); hr.interpolate("time", inplace=True); hr.reset_index(inplace=True)
            diff_rbm_ma2[p] = calc_time_ma(hr, n_hours, mask)

# ---------------------------------------------------------
# Sidebar page selector
# ---------------------------------------------------------
PAGE = st.sidebar.radio("View", ["GNSS Dashboard", "Plan Deformation"])

# ---------------------------------------------------------
# Helper for moving‑average overlays (used in Dashboard page)
# ---------------------------------------------------------

def add_ma(fig, p, v, c):
    if use_ma1 and ma1[p] is not None:
        fig.add_trace(go.Scatter(x=ma1[p].Timestamp, y=ma1[p][v], name=f"{p} {v} MA{n_samples}", line=dict(color=c, dash="dot", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))
    if use_ma2 and ma2[p] is not None:
        fig.add_trace(go.Scatter(x=ma2[p].Timestamp, y=ma2[p][v], name=f"{p} {v} {n_hours}h", line=dict(color=c, dash="dash", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))

def apply_xstart(fig, xmax):
    if use_xstart: fig.update_xaxes(range=[pd.Timestamp(x_start), xmax])


# ---------------------------------------------------------
# Re-usable helper to build a single plan-view figure
# (Used both for the live preview and for video frames)
# ---------------------------------------------------------
def build_plan_view(ts, *, src_label, scale, show_circ, lock_aspect):
    cm_row = common.set_index("Timestamp").loc[pd.Timestamp(ts)]
    trans  = np.array([cm_row.Easting, cm_row.Northing]) * scale

    lbl = re.sub(r"\s+", "", src_label.lower())   # drop blanks & lower-case

    if "raw" in lbl:
        src_dict = diff_rbm                        # instantaneous offsets
    elif re.search(r"\d+h$|hour|hr\b|hrs?\b|h$", lbl):
        src_dict = diff_rbm_ma2                    # MA by elapsed hours
    else:
        src_dict = diff_rbm_ma1                    # MA by N samples

    # Now build the warped rectangle
    warp_xy = []
    for idx, p in enumerate(["SW", "SE", "NE", "NW"]):
        res = (
            src_dict[p]
            .set_index("Timestamp")
            .loc[pd.Timestamp(ts)]
        )
        warp_xy.append(
            rect_xy[idx]
            + trans
            + np.array([res.Easting, res.Northing]) * scale
        )
    warp_xy = np.vstack(warp_xy)

    poly_design = np.vstack([rect_xy, rect_xy[0]])
    poly_trans  = np.vstack([rect_xy + trans, (rect_xy + trans)[0]])
    poly_warp   = np.vstack([warp_xy, warp_xy[0]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=poly_design[:, 0], y=poly_design[:, 1],
                             mode="lines+markers", name="Design",
                             line=dict(color="grey", dash="dot")))
    fig.add_trace(go.Scatter(x=poly_trans[:, 0], y=poly_trans[:, 1],
                             mode="lines+markers", name="Mean-translated",
                             line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=poly_warp[:, 0],  y=poly_warp[:, 1],
                             mode="lines+markers", name="RBM-deformed",
                             line=dict(color="cyan")))

    if show_circ:
        for cx, cy in np.vstack([rect_xy, rect_xy + trans]):
            for rmm in (1, 2, 5, 8):
                r = rmm / 1_000 * scale
                th = np.linspace(0, 2*np.pi, 60)
                fig.add_trace(go.Scatter(x=cx+r*np.cos(th), y=cy+r*np.sin(th),
                                         mode="lines", showlegend=False,
                                         line=dict(color="white", width=0.5)))

    for i, lbl in enumerate(["SW", "SE", "NE", "NW"]):
        fig.add_annotation(x=poly_warp[i, 0], y=poly_warp[i, 1],
                           text=lbl, showarrow=False,
                           font=dict(color="white", size=12))

    if lock_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(title=f"Plan view — {ts:%Y-%m-%d %H:%M} "
                            f"({src_label}, ×{scale})",
                      height=1_000,
                      legend=dict(orientation="h"))
    return fig




# =============================================================================
# PAGE 1 ───────────────────────────────── GNSS DASHBOARD ─────────────────────
# =============================================================================
if PAGE == "GNSS Dashboard":

    # -----------------------------------------------------
    # Main plots (raw)
    # -----------------------------------------------------
    for v in VARS:
        fig = go.Figure(); cyc = itertools.cycle(plc.qualitative.D3)
        for p in sel_pts:
            col = next(cyc)
            fig.add_trace(go.Scatter(x=base[p].Timestamp, y=base[p][v], name=f"{p} {v}", line=dict(color=col), connectgaps=True))
            add_ma(fig, p, v, col)
        if show_temp:
            for p in sel_pts:
                fig.add_trace(go.Scatter(x=base[p].Timestamp, y=base[p].Temperature, name=f"{p} Temp", line=dict(color="grey", dash="dot"), opacity=0.4, connectgaps=True))
        apply_yaxis(fig, y_cfg["Main"][v]); apply_xstart(fig, max(df.Timestamp.max() for df in base.values()))
        fig.update_layout(title=v, height=PLOT_H, hovermode="x unified", legend=legend_cfg)
        st.plotly_chart(fig, use_container_width=True, key=f"main_{v}")

    # -----------------------------------------------------
    # Common vs arithmetic Δ
    # -----------------------------------------------------
    if common is not None:
        st.subheader("Common vs Differential")
        if use_ma1: cm1 = calc_sample_ma(common[["Timestamp"] + VARS], n_samples)
        if use_ma2: cm2 = calc_time_ma(common.set_index("Timestamp").resample("1H").mean().reset_index(), n_hours, common[VARS[0]].notna())
        for v in VARS:
            fig = go.Figure(); fig.add_trace(go.Scatter(x=common.Timestamp, y=common[v], name="Common", line=dict(width=4), connectgaps=True))
            if use_ma1: fig.add_trace(go.Scatter(x=cm1.Timestamp, y=cm1[v], name=f"Common MA{n_samples}", line=dict(dash="dot", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))
            if use_ma2: fig.add_trace(go.Scatter(x=cm2.Timestamp, y=cm2[v], name=f"Common {n_hours}h", line=dict(dash="dash", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))
            cyc = itertools.cycle(plc.qualitative.D3)
            for p in sel_pts:
                col = next(cyc)
                fig.add_trace(go.Scatter(x=diff[p].Timestamp, y=diff[p][v], name=f"{p} Δ{v[0]}", line=dict(color=col), connectgaps=True))
                if use_ma1: fig.add_trace(go.Scatter(x=diff_ma1[p].Timestamp, y=diff_ma1[p][v], name=f"{p} Δ{v[0]} MA{n_samples}", line=dict(color=col, dash="dot"), opacity=0.5, connectgaps=True))
                if use_ma2: fig.add_trace(go.Scatter(x=diff_ma2[p].Timestamp, y=diff_ma2[p][v], name=f"{p} Δ{v[0]} {n_hours}h", line=dict(color=col, dash="dash"), opacity=0.6, connectgaps=True))
            apply_yaxis(fig, y_cfg["CommonΔ"][v]); apply_xstart(fig, common.Timestamp.max())
            fig.update_layout(title=v, height=PLOT_H, hovermode="x unified", legend=legend_cfg); st.plotly_chart(fig, use_container_width=True, key=f"comm_{v}")

    # -----------------------------------------------------
    # Rigid‑Body Δ plots
    # -----------------------------------------------------
    if diff_rbm:
        st.subheader("Rigid‑Body Differential (translation + rotation)")
        for v in VARS:
            fig = go.Figure(); cyc = itertools.cycle(plc.qualitative.Dark24)
            for p in ORDER:
                col = next(cyc)
                fig.add_trace(go.Scatter(x=diff_rbm[p].Timestamp, y=diff_rbm[p][v], name=f"{p} Δ{v[0]} (RBM)", line=dict(color=col), connectgaps=True))
                if use_ma1: fig.add_trace(go.Scatter(x=diff_rbm_ma1[p].Timestamp, y=diff_rbm_ma1[p][v], name=f"{p} Δ{v[0]} (RBM) MA{n_samples}", line=dict(color=col, dash="dot", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))
                if use_ma2: fig.add_trace(go.Scatter(x=diff_rbm_ma2[p].Timestamp, y=diff_rbm_ma2[p][v], name=f"{p} Δ{v[0]} (RBM) {n_hours}h", line=dict(color=col, dash="dash", width=MA_WIDTH), opacity=MA_OPACITY, connectgaps=True))
            if rbm_common is not None:
                if v == "Easting": fig.add_trace(go.Scatter(x=rbm_common.Timestamp, y=rbm_common.E_trans, name="RBM Common E", line=dict(color="black", width=4), connectgaps=True))
                elif v == "Northing": fig.add_trace(go.Scatter(x=rbm_common.Timestamp, y=rbm_common.N_trans, name="RBM Common N", line=dict(color="black", width=4), connectgaps=True))
                else: fig.add_trace(go.Scatter(x=rbm_common.Timestamp, y=rbm_common.H_plane, name="RBM Common H", line=dict(color="black", width=4), connectgaps=True))
            apply_yaxis(fig, y_cfg["RBMΔ"][v]); xmax = rbm_common.Timestamp.max() if rbm_common is not None else diff_rbm[ORDER[0]].Timestamp.max(); apply_xstart(fig, xmax)
            fig.update_layout(title=f"{v} residuals (rigid‑body)", height=PLOT_H, hovermode="x unified", legend=legend_cfg); st.plotly_chart(fig, use_container_width=True, key=f"rbm_{v}")

    # ---------------------------------------------------------
    # 📈 Time‑Window Statistics
    # ---------------------------------------------------------
    with st.expander("📈 Time‑Window Statistics"):
        st.caption("Select a start / end date‑time; stats are hourly‑based.")
        col1, col2 = st.columns(2)

        gmin = min(df.Timestamp.min() for df in base.values())
        gmax = max(df.Timestamp.max() for df in base.values())

        d_from = col1.date_input("Start date", gmin.date(), key="stats_start_date")
        t_from = col1.time_input("Start time", gmin.time(), key="stats_start_time")
        d_to   = col2.date_input("End date",   gmax.date(), key="stats_end_date")
        t_to   = col2.time_input("End time",   gmax.time(), key="stats_end_time")

        if col2.button("Compute stats"):
            t0 = pd.Timestamp(dt.datetime.combine(d_from, t_from))
            t1 = pd.Timestamp(dt.datetime.combine(d_to,   t_to))
            rows = []
            for fam, data_dict in [("Raw", base), ("Δ", diff), ("RBMΔ", diff_rbm)]:
                for p, df in data_dict.items():
                    win = df[(df.Timestamp >= t0) & (df.Timestamp <= t1)].copy()
                    tw = (
                        win.set_index("Timestamp")
                        .resample("2H", origin=t0)
                        .mean()
                    )
                    # % missing relative to the 2-hour schedule
                    missing = tw[VARS[0]].isna().mean() * 100.0
                    # keep only actual readings for stats below
                    tw.dropna(subset=[VARS[0]], inplace=True)
                    for v in VARS:
                        mean = tw[v].mean()
                        std  = tw[v].std()
                        d2h  = tw[v].diff().abs().max()
                        rows.append([fam, p, v, mean, std, missing, d2h])
            st.dataframe(
                pd.DataFrame(rows,
                            columns=["Fam", "Point", "Var", "Mean", "StdDev", "% Missing (2 hr)", "Max Δ2h"])
                .set_index(["Fam", "Point", "Var"])
            )

    # ---------------------------------------------------------
    # 🔧 Custom Compare
    # ---------------------------------------------------------
    with st.expander("🔧 Custom Compare"):
        trace_bank = {}

        # raw -------------------------------------------------
        for p in sel_pts:
            for v in VARS:
                trace_bank[f"{p} {v}"] = (base[p], v)
                if use_ma1:
                    trace_bank[f"{p} {v} MA{n_samples}"] = (ma1[p], v)
                if use_ma2:
                    trace_bank[f"{p} {v} {n_hours}h"]    = (ma2[p], v)

        # arithmetic common & Δ ------------------------------
        if common is not None:
            for v in VARS:
                trace_bank[f"Common {v}"] = (common, v)
                if use_ma1:
                    trace_bank[f"Common {v} MA{n_samples}"] = (cm1, v)
                if use_ma2:
                    trace_bank[f"Common {v} {n_hours}h"]    = (cm2, v)
            for p in sel_pts:
                for v in VARS:
                    lbl = f"{p} Δ{v[0]}"
                    trace_bank[lbl] = (diff[p], v)
                    if use_ma1:
                        trace_bank[f"{lbl} MA{n_samples}"] = (diff_ma1[p], v)
                    if use_ma2:
                        trace_bank[f"{lbl} {n_hours}h"]    = (diff_ma2[p], v)

        # RBM Δ & common -------------------------------------
        if diff_rbm:
            for p in ORDER:
                for v in VARS:
                    lbl = f"{p} Δ{v[0]} RBM"
                    trace_bank[lbl] = (diff_rbm[p], v)
                    if use_ma1:
                        trace_bank[f"{lbl} MA{n_samples}"] = (diff_rbm_ma1[p], v)
                    if use_ma2:
                        trace_bank[f"{lbl} {n_hours}h"]    = (diff_rbm_ma2[p], v)
            trace_bank["RBM Common E"] = (rbm_common.rename(columns={"E_trans": "Easting"}), "Easting")
            trace_bank["RBM Common N"] = (rbm_common.rename(columns={"N_trans": "Northing"}), "Northing")
            trace_bank["RBM Common H"] = (rbm_common.rename(columns={"H_plane": "Height"}),  "Height")
            

        # selector & plotting --------------------------------
        sel_traces = st.multiselect("Series to plot", list(trace_bank))
        rel_zero   = st.checkbox("Relative‑zero within current view")
        if sel_traces:
            fig = go.Figure()
            palette = itertools.cycle(plc.qualitative.Plotly)
            for lbl in sel_traces:
                df, v = trace_bank[lbl]
                x = df.Timestamp
                y = df[v].copy()
                if rel_zero and len(y) > 0:
                    y -= y.iloc[0]
                fig.add_trace(go.Scatter(x=x, y=y, name=lbl,
                                        line=dict(color=next(palette)),
                                        connectgaps=True))
            if use_xstart:
                fig.update_xaxes(range=[pd.Timestamp(x_start),
                                        max(df.Timestamp.max() for df, _ in trace_bank.values())])
            fig.update_layout(height=PLOT_H, hovermode="x unified", legend=legend_cfg)
            st.plotly_chart(fig, use_container_width=True, key="compare_custom")







# =============================================================================
# PAGE 2 ─────────────────────────────── PLAN DEFORMATION ─────────────────────
# =============================================================================
if PAGE == "Plan Deformation":

    need = {"SW", "SE", "NE", "NW"}
    if not (need.issubset(sel_pts) and diff_rbm):
        st.info("Select SW / SE / NE / NW and ensure RBM residuals exist."); st.stop()

    # geometry
    shift_xy = INIT_XYZ["SE"][:2]; XY_ref = {p: INIT_XYZ[p][:2] - shift_xy for p in need}; rect_xy = np.vstack([XY_ref[p] for p in ["SW", "SE", "NE", "NW"]])

    # UI defaults requested
    src_label = st.radio("Data family", ["Raw", f"MA{n_samples}", f"{n_hours}h"], index=2, horizontal=True)
    scale     = st.number_input("Exaggeration scale (×)", 1, 1_000_000, value=1_000, step=10)
    show_circ = st.checkbox("Show scale circles (1-2-5-8 mm)", value=True)
    lock_aspect = st.checkbox("Lock 1 : 1 aspect ratio", value=True)

    # cache heavy merge per src_label
    if "plan_cache" not in st.session_state: st.session_state["plan_cache"] = {}
    if src_label not in st.session_state["plan_cache"]:
        def pick(raw, ma1, ma2):
            if src_label.startswith("MA") and ma1 is not None: return ma1
            if src_label.endswith("h") and ma2 is not None: return ma2
            return raw
        df_list = []
        for p in need:
            src = pick(base[p], ma1.get(p), ma2.get(p))
            part = src[["Timestamp", "Easting", "Northing"]].rename(columns={"Easting": f"{p}_E", "Northing": f"{p}_N"}); df_list.append(part)
        plan_df = df_list[0]
        for nxt in df_list[1:]: plan_df = plan_df.merge(nxt, on="Timestamp", how="inner")
        common_src = pick(common, globals().get("cm1"), globals().get("cm2")); valid = ~common_src[VARS].isna().any(axis=1); plan_df = plan_df[plan_df.Timestamp.isin(common_src[valid].Timestamp)]
        st.session_state["plan_cache"][src_label] = plan_df.copy()
    plan_df = st.session_state["plan_cache"][src_label]
    if plan_df.empty: st.info("No epochs with complete data in the chosen family."); st.stop()

    timestamps = plan_df["Timestamp"].sort_values().dt.to_pydatetime().tolist()
    sel_ts = st.slider("Select epoch", min_value=timestamps[0], max_value=timestamps[-1], value=timestamps[0], format="YYYY-MM-DD HH:mm")

    # lightweight per‑epoch calculations
    cm_row = common.set_index("Timestamp").loc[pd.Timestamp(sel_ts)]; trans = np.array([cm_row.Easting, cm_row.Northing]) * scale
    warp_xy = []
    fig = build_plan_view(
        sel_ts,
        src_label=src_label,
        scale=scale,
        show_circ=show_circ,
        lock_aspect=lock_aspect,
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 🎞️  History video generation
# -----------------------------------------------------
    make_vid = st.button("🎞️  Create history video")
    if make_vid:
        vid_key = (src_label, scale, show_circ, lock_aspect)   # current UI state

        if "plan_vid_cache" not in st.session_state:
            st.session_state["plan_vid_cache"] = {}

        if vid_key not in st.session_state["plan_vid_cache"]:
            with st.spinner("Rendering video … this may take a moment"):
                frames = []
                for t in timestamps:
                    f = build_plan_view(t, src_label=src_label,
                                        scale=scale, show_circ=show_circ,
                                        lock_aspect=lock_aspect)
                    # PNG export (needs kaleido)
                    png = f.to_image(format="png", width=1_000,
                                    height=1_000, scale=2)
                    frames.append(imageio.imread(png))

                # one frame every 2 s  →  fps = 0.5
                tmp = tempfile.NamedTemporaryFile(delete=False,
                                                suffix=".mp4")
                imageio.mimwrite(tmp.name, frames,
                                fps=2, codec="libx264")
                st.session_state["plan_vid_cache"][vid_key] = tmp.name

        vpath = st.session_state["plan_vid_cache"][vid_key]
        st.video(vpath)
        with open(vpath, "rb") as f:
            st.download_button("Download video",
                            f.read(),
                            file_name="plan_history.mp4",
                            mime="video/mp4")


# ---------------------------------------------------------
# 💾 Save session
# ---------------------------------------------------------
with st.sidebar.expander("💾 Save session"):
    if st.button("Generate .gnss"):
        gnss_blob = save_session(points)
        st.download_button("Download session", gnss_blob, file_name="session.gnss", mime="application/octet-stream")
