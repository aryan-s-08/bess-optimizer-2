"""
BESS Profit Optimization — Streamlit Dashboard
Run with:  python -m streamlit run bess_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD, LpStatus

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BESS Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────
BG      = "#0f1114"
SURFACE = "#181b20"
BORDER  = "#2a2d35"
TEXT    = "#d4d8e1"
MUTED   = "#6b7280"
ACCENT  = "#3b82f6"
GREEN   = "#22c55e"
RED     = "#ef4444"
AMBER   = "#f59e0b"
SOC_COL = "#a78bfa"
FONT    = "'Inter', 'Segoe UI', sans-serif"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: {FONT}; background-color: {BG}; color: {TEXT}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  .top-bar {{ display:flex; align-items:baseline; gap:12px; padding:0 0 20px 0;
              border-bottom:1px solid {BORDER}; margin-bottom:24px; }}
  .top-bar h1 {{ font-size:20px; font-weight:600; color:{TEXT}; margin:0; letter-spacing:-0.3px; }}
  .top-bar span {{ font-size:13px; color:{MUTED}; }}

  .kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }}
  .kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:6px; padding:16px 18px; }}
  .kpi-label {{ font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.6px;
                color:{MUTED}; margin-bottom:6px; }}
  .kpi-value {{ font-size:26px; font-weight:700; letter-spacing:-0.5px; color:{TEXT}; }}
  .kpi-value.pos {{ color:{GREEN}; }}
  .kpi-value.neg {{ color:{RED}; }}
  .kpi-sub {{ font-size:11px; color:{MUTED}; margin-top:4px; }}

  .constraint-box {{ background:{SURFACE}; border:1px solid {AMBER}33; border-left:3px solid {AMBER};
                     border-radius:6px; padding:12px 16px; margin-bottom:20px; font-size:13px; color:{TEXT}; }}
  .constraint-box b {{ color:{AMBER}; }}

  .pill {{ display:inline-block; font-size:11px; font-weight:500; padding:2px 8px;
           border-radius:20px; margin-bottom:20px; }}
  .pill.ok   {{ background:rgba(34,197,94,0.12); color:{GREEN}; border:1px solid rgba(34,197,94,0.25); }}
  .pill.warn {{ background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.25); }}

  section[data-testid="stSidebar"] {{ background-color:{SURFACE}; border-right:1px solid {BORDER}; }}
  .sidebar-section {{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;
                      color:{MUTED}; margin:20px 0 10px 0; padding-bottom:6px; border-bottom:1px solid {BORDER}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ font-size:13px; font-weight:500; color:{MUTED};
                                   padding:8px 18px; border-radius:0; background:transparent; }}
  .stTabs [aria-selected="true"] {{ color:{TEXT} !important; border-bottom:2px solid {ACCENT} !important; }}

  .meta-row {{ font-size:12px; color:{MUTED}; margin-bottom:20px; }}
  .meta-row b {{ color:{TEXT}; font-weight:500; }}

  .stDownloadButton > button {{ background:transparent; border:1px solid {BORDER}; color:{TEXT};
                                 font-size:13px; padding:6px 16px; border-radius:4px; }}
  .stDownloadButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
</style>
""", unsafe_allow_html=True)


# ── Optimizer (MW units + demand constraint) ──────────────────
def run_bess_optimization(
    predicted_prices,
    max_power_mw      = 5.0,
    duration_hrs      = 4.0,
    efficiency        = 0.90,
    min_soc_pct       = 0.10,
    initial_soc_pct   = 0.10,
    demand_limit_mw   = 4.0,   # NEW: grid can only absorb 4 MW discharge at a time
):
    """
    LP optimizer for BESS energy arbitrage.

    Units: MW / MWh / €/MWh  (national grid scale)

    Constraints:
      - Max charge power       : max_power_mw
      - Max discharge power    : min(max_power_mw, demand_limit_mw)  ← demand constraint
      - Energy balance         : SoC dynamics with round-trip efficiency
      - SoC bounds             : [min_soc, 100%]
      - No simultaneous charge + discharge
      - Charge when price low, discharge when price high (enforced by objective)
    """
    MAX_MWH        = max_power_mw * duration_hrs          # e.g. 5 MW × 4 h = 20 MWh
    MIN_E          = min_soc_pct * MAX_MWH
    INIT_E         = initial_soc_pct * MAX_MWH
    # Effective discharge cap = battery limit OR demand limit, whichever is tighter
    DISCHARGE_CAP  = min(max_power_mw, demand_limit_mw)
    T              = len(predicted_prices)
    HOURS          = range(T)

    prob = LpProblem("BESS_Arbitrage_MW", LpMaximize)

    p_c = LpVariable.dicts("charge",    HOURS, lowBound=0, upBound=max_power_mw)
    p_d = LpVariable.dicts("discharge", HOURS, lowBound=0, upBound=DISCHARGE_CAP)  # demand-constrained
    soc = LpVariable.dicts("soc",       range(T+1), lowBound=MIN_E, upBound=MAX_MWH)
    b_c = LpVariable.dicts("b_chg",     HOURS, cat="Binary")
    b_d = LpVariable.dicts("b_dis",     HOURS, cat="Binary")

    # Objective: maximise revenue from discharging minus cost of charging
    # This naturally enforces: charge when low, discharge when high
    prob += lpSum(
        p_d[h] * predicted_prices.iloc[h]
      - p_c[h] * predicted_prices.iloc[h]
        for h in HOURS
    )

    prob += soc[0] == INIT_E
    prob += soc[T] >= MIN_E

    for h in HOURS:
        # Energy balance with efficiency losses
        prob += soc[h+1] == soc[h] + p_c[h] * efficiency - p_d[h] * (1.0 / efficiency)
        # Mutual exclusion: can't charge and discharge at same hour
        prob += p_c[h] <= b_c[h] * max_power_mw
        prob += p_d[h] <= b_d[h] * DISCHARGE_CAP
        prob += b_c[h] + b_d[h] <= 1

    prob.solve(PULP_CBC_CMD(msg=0))

    net    = value(prob.objective) or 0.0
    cv     = [p_c[h].varValue or 0.0 for h in HOURS]
    dv     = [p_d[h].varValue or 0.0 for h in HOURS]
    sv_mwh = [soc[h].varValue or MIN_E for h in range(T+1)]
    sv_pct = [100*v/MAX_MWH for v in sv_mwh]

    actions = [
        "Charge"    if c > 0.001 else
        "Discharge" if d > 0.001 else
        "Idle"
        for c, d in zip(cv, dv)
    ]

    df = pd.DataFrame({
        "Hour"           : list(range(T)),
        "Timestamp"      : predicted_prices.index,
        "Price (€/MWh)"  : predicted_prices.values.round(2),
        "Action"         : actions,
        "Charge (MW)"    : [round(v, 3) for v in cv],
        "Discharge (MW)" : [round(v, 3) for v in dv],
        "SoC Start (%)"  : [round(v, 1) for v in sv_pct[:T]],
        "SoC End (%)"    : [round(v, 1) for v in sv_pct[1:]],
        "Revenue (€)"    : [round(d*p, 2) for d, p in zip(dv, predicted_prices.values)],
        "Cost (€)"       : [round(c*p, 2) for c, p in zip(cv, predicted_prices.values)],
    })
    df["Net P&L (€)"] = (df["Revenue (€)"] - df["Cost (€)"]).round(2)

    return {
        "total_profit" : net,
        "schedule_df"  : df,
        "soc_pct"      : sv_pct,
        "status"       : LpStatus[prob.status],
        "max_mwh"      : MAX_MWH,
        "discharge_cap": DISCHARGE_CAP,
    }


# ── Model load ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("bess_xgboost_model.pkl"), joblib.load("bess_feature_columns.pkl")
    except FileNotFoundError:
        return None, None

model, feature_columns = load_model()

# Representative Spain day-ahead price curve (€/MWh)
DEMO_PRICES = np.array([
    38.2, 35.1, 32.5, 30.8, 31.2, 34.7,
    42.1, 55.3, 68.9, 74.2, 72.1, 65.4,
    58.3, 54.7, 56.2, 61.8, 75.4, 89.2,
    92.1, 88.4, 77.6, 65.2, 52.1, 42.8,
])


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Battery Parameters</div>', unsafe_allow_html=True)
    max_power    = st.slider("Max Power (MW)",             1.0, 20.0,  5.0, 0.5)
    duration     = st.slider("Duration (hours)",           1.0,  8.0,  4.0, 0.5)
    efficiency   = st.slider("Round-Trip Efficiency (%)",   70,   99,   90,   1)
    min_soc      = st.slider("Min SoC (%)",                  0,   30,   10,   1)
    init_soc     = st.slider("Initial SoC (%)",              0,  100,   10,   5)

    cap = max_power * duration
    st.markdown(
        f'<div style="font-size:12px;color:{MUTED};margin-top:8px;">'
        f'Capacity: <b style="color:{TEXT}">{cap:.1f} MWh</b></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">Demand Constraint</div>', unsafe_allow_html=True)
    demand_limit = st.slider(
        "Grid Absorption Limit (MW)",
        min_value=1.0, max_value=float(max_power), value=4.0, step=0.5,
        help="Maximum MW the grid can absorb at any single hour (discharge cap)"
    )
    effective_cap = min(max_power, demand_limit)
    st.markdown(
        f'<div style="font-size:12px;color:{MUTED};margin-top:4px;">'
        f'Effective discharge cap: <b style="color:{AMBER}">{effective_cap:.1f} MW</b></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">Price Input</div>', unsafe_allow_html=True)
    price_mode = st.radio(
        "Source", ["Demo prices", "Manual entry", "Upload CSV"],
        label_visibility="collapsed"
    )

    custom_prices = None
    if price_mode == "Manual entry":
        raw = st.text_area("24 comma-separated values (€/MWh)",
                           value=", ".join(map(str, DEMO_PRICES)))
        try:
            vals = [float(x.strip()) for x in raw.split(",")]
            custom_prices = np.array(vals) if len(vals) == 24 else None
            if len(vals) != 24:
                st.error(f"Need 24 values, got {len(vals)}.")
        except Exception:
            st.error("Parse error.")
    elif price_mode == "Upload CSV":
        up = st.file_uploader("CSV with a 'price' column (24 rows)", type=["csv"])
        if up:
            try:
                udf = pd.read_csv(up)
                col = [c for c in udf.columns if "price" in c.lower()][0]
                custom_prices = udf[col].values[:24]
            except Exception as e:
                st.error(str(e))


# ── Price resolution ──────────────────────────────────────────
prices_array  = custom_prices if custom_prices is not None else DEMO_PRICES
price_source  = "Custom" if custom_prices is not None else "Demo — Spain day-ahead"
timestamps    = pd.date_range("2018-03-15 00:00", periods=24, freq="h", tz="UTC")
prices_series = pd.Series(prices_array, index=timestamps)


# ── Solve ─────────────────────────────────────────────────────
with st.spinner("Solving..."):
    result = run_bess_optimization(
        prices_series,
        max_power_mw    = max_power,
        duration_hrs    = duration,
        efficiency      = efficiency / 100,
        min_soc_pct     = min_soc    / 100,
        initial_soc_pct = init_soc   / 100,
        demand_limit_mw = demand_limit,
    )

sched   = result["schedule_df"]
soc_pct = result["soc_pct"]

total_revenue = sched["Revenue (€)"].sum()
total_cost    = sched["Cost (€)"].sum()
net_profit    = result["total_profit"]
peak_soc      = max(soc_pct)
n_charge      = (sched["Action"] == "Charge").sum()
n_discharge   = (sched["Action"] == "Discharge").sum()
n_idle        = (sched["Action"] == "Idle").sum()

# Price-rank check: are we charging in the cheapest hours & discharging in the most expensive?
charge_hours    = sched[sched["Action"] == "Charge"]["Price (€/MWh)"]
discharge_hours = sched[sched["Action"] == "Discharge"]["Price (€/MWh)"]
logic_ok = (
    len(charge_hours) == 0 or len(discharge_hours) == 0 or
    charge_hours.mean() < discharge_hours.mean()
)


# ── Header ────────────────────────────────────────────────────
model_status = "Model loaded — ML predictions active" if model else "No model file — running on demo prices"
pill_class   = "ok" if model else "warn"

st.markdown(f"""
<div class="top-bar">
  <h1>BESS Profit Optimizer</h1>
  <span>Battery Energy Storage · Energy Arbitrage · Spain Grid · MW Scale</span>
</div>
<div class="pill {pill_class}">{model_status}</div>
""", unsafe_allow_html=True)

# Demand constraint info box
constraint_active = demand_limit < max_power
if constraint_active:
    st.markdown(f"""
    <div class="constraint-box">
      <b>Demand constraint active:</b> Grid absorption limited to <b>{demand_limit:.1f} MW</b> per hour.
      Battery max power is {max_power:.1f} MW — discharge is capped at {effective_cap:.1f} MW.
      This reduces peak revenue but reflects real grid absorption limits.
    </div>
    """, unsafe_allow_html=True)


# ── KPI row ───────────────────────────────────────────────────
profit_class = "pos" if net_profit >= 0 else "neg"
logic_note   = "Avg charge price lower than avg discharge price" if logic_ok else "Warning: check price ordering"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Net Arbitrage Profit</div>
    <div class="kpi-value {profit_class}">€{net_profit:,.2f}</div>
    <div class="kpi-sub">After {efficiency}% efficiency losses</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Gross Revenue</div>
    <div class="kpi-value">€{total_revenue:,.2f}</div>
    <div class="kpi-sub">From {n_discharge} discharge hours</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Gross Cost</div>
    <div class="kpi-value">€{total_cost:,.2f}</div>
    <div class="kpi-sub">From {n_charge} charge hours</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Peak State of Charge</div>
    <div class="kpi-value">{peak_soc:.1f}%</div>
    <div class="kpi-sub">Capacity: {cap:.1f} MWh</div>
  </div>
</div>
<div class="meta-row">
  Optimizer: <b>{result['status']}</b> &nbsp;·&nbsp;
  Discharge cap: <b>{result['discharge_cap']:.1f} MW</b> &nbsp;·&nbsp;
  Price spread: <b>€{prices_array.max()-prices_array.min():.2f}/MWh</b> &nbsp;·&nbsp;
  Logic check: <b>{logic_note}</b> &nbsp;·&nbsp;
  Source: <b>{price_source}</b>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────
tab_chart, tab_table, tab_analytics = st.tabs(["Schedule Chart", "Schedule Table", "Analytics"])

CHART_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family=FONT, size=12),
    margin=dict(l=12, r=12, t=36, b=12),
)


# ── Tab 1: Chart ──────────────────────────────────────────────
with tab_chart:
    hours  = list(range(24))
    prices = sched["Price (€/MWh)"].values
    chg    = sched["Charge (MW)"].values
    dis    = sched["Discharge (MW)"].values

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.06,
        subplot_titles=("Price & Charge / Discharge Schedule (MW)", "State of Charge (%)")
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=hours, y=prices, name="Price (€/MWh)",
        line=dict(color=ACCENT, width=2), mode="lines",
    ), row=1, col=1)

    # Charge bars (positive)
    fig.add_trace(go.Bar(
        x=hours, y=chg, name="Charge (MW)",
        marker_color=GREEN, opacity=0.7,
    ), row=1, col=1)

    # Discharge bars (mirrored negative so they go down)
    fig.add_trace(go.Bar(
        x=hours, y=[-v for v in dis], name="Discharge (MW)",
        marker_color=RED, opacity=0.7,
    ), row=1, col=1)

    # Demand limit reference line
    if constraint_active:
        fig.add_hline(
            y=-demand_limit, row=1, col=1,
            line=dict(color=AMBER, width=1, dash="dot"),
            annotation_text=f"Demand limit −{demand_limit:.1f} MW",
            annotation_font=dict(color=AMBER, size=11),
            annotation_position="bottom right",
        )

    # SoC area
    fig.add_trace(go.Scatter(
        x=list(range(25)), y=soc_pct, name="SoC (%)",
        line=dict(color=SOC_COL, width=1.8),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
        mode="lines",
    ), row=2, col=1)

    fig.add_hline(
        y=min_soc, row=2, col=1,
        line=dict(color=RED, width=1, dash="dot"),
        annotation_text=f"Min SoC {min_soc}%",
        annotation_font=dict(color=MUTED, size=11),
        annotation_position="top right",
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=570,
        barmode="overlay",
        legend=dict(orientation="h", x=0, y=1.06,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig.update_yaxes(title_text="€/MWh", row=1, col=1,
                     gridcolor=BORDER, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(title_text="SoC %", row=2, col=1,
                     range=[0, 105], gridcolor=BORDER,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(gridcolor=BORDER, tickfont=dict(color=MUTED),
                     tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)], row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", gridcolor=BORDER, tickfont=dict(color=MUTED),
                     tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Charge vs discharge price comparison
    if len(charge_hours) > 0 and len(discharge_hours) > 0:
        st.markdown(f"""
        <div style="font-size:12px;color:{MUTED};margin-top:-8px;">
          Avg charge price: <b style="color:{GREEN}">€{charge_hours.mean():.2f}/MWh</b>
          &nbsp;·&nbsp;
          Avg discharge price: <b style="color:{RED}">€{discharge_hours.mean():.2f}/MWh</b>
          &nbsp;·&nbsp;
          Spread captured: <b style="color:{TEXT}">€{discharge_hours.mean()-charge_hours.mean():.2f}/MWh</b>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 2: Table ──────────────────────────────────────────────
with tab_table:
    st.markdown(
        f'<div style="font-size:13px;color:{MUTED};margin-bottom:12px;">'
        f'24-hour optimal schedule · Units: MW / MWh · '
        f'Discharge capped at {result["discharge_cap"]:.1f} MW · '
        f'Source: {price_source}</div>',
        unsafe_allow_html=True
    )

    def colour_action(val):
        if val == "Charge":    return f"background-color:#0f2318; color:{GREEN}"
        if val == "Discharge": return f"background-color:#2a0f0f; color:{RED}"
        return f"color:{MUTED}"

    def colour_pnl(val):
        if val > 0: return f"color:{GREEN}; font-weight:600"
        if val < 0: return f"color:{RED}; font-weight:600"
        return f"color:{MUTED}"

    cols = [
        "Hour", "Timestamp", "Price (€/MWh)", "Action",
        "Charge (MW)", "Discharge (MW)",
        "SoC Start (%)", "SoC End (%)",
        "Revenue (€)", "Cost (€)", "Net P&L (€)"
    ]

    styled = (
        sched[cols].style
        .map(colour_action, subset=["Action"])
        .map(colour_pnl,    subset=["Net P&L (€)"])
        .format({
            "Price (€/MWh)"  : "{:.2f}",
            "Charge (MW)"    : "{:.3f}",
            "Discharge (MW)" : "{:.3f}",
            "SoC Start (%)"  : "{:.1f}",
            "SoC End (%)"    : "{:.1f}",
            "Revenue (€)"    : "{:.2f}",
            "Cost (€)"       : "{:.2f}",
            "Net P&L (€)"    : "{:.2f}",
        })
        .set_properties(**{"font-size": "12px", "font-family": FONT})
    )

    st.dataframe(styled, use_container_width=True, height=660)
    st.download_button(
        "Download as CSV",
        sched.to_csv(index=False).encode("utf-8"),
        "bess_schedule.csv", "text/csv",
    )


# ── Tab 3: Analytics ──────────────────────────────────────────
with tab_analytics:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f'<div style="font-size:13px;font-weight:500;color:{TEXT};margin-bottom:10px;">Hourly Net P&L (€)</div>', unsafe_allow_html=True)
        colours = [GREEN if v >= 0 else RED for v in sched["Net P&L (€)"]]
        fig_pnl = go.Figure(go.Bar(
            x=sched["Hour"], y=sched["Net P&L (€)"],
            marker_color=colours, marker_opacity=0.8,
        ))
        fig_pnl.update_layout(
            **CHART_LAYOUT, height=280, showlegend=False,
            yaxis=dict(title="€", gridcolor=BORDER, tickfont=dict(color=MUTED),
                       zeroline=True, zerolinecolor=BORDER),
            xaxis=dict(title="Hour", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col_b:
        st.markdown(f'<div style="font-size:13px;font-weight:500;color:{TEXT};margin-bottom:10px;">Action Distribution</div>', unsafe_allow_html=True)
        ac = sched["Action"].value_counts()
        colour_map = {"Charge": GREEN, "Discharge": RED, "Idle": MUTED}
        fig_pie = go.Figure(go.Pie(
            labels=ac.index, values=ac.values,
            marker=dict(colors=[colour_map.get(l, MUTED) for l in ac.index]),
            hole=0.55, textinfo="label+value",
            textfont=dict(size=12, color=TEXT),
        ))
        fig_pie.update_layout(**CHART_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Price by action histogram
    st.markdown(f'<div style="font-size:13px;font-weight:500;color:{TEXT};margin:16px 0 10px;">Price Distribution by Action</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    for action, colour in [("Charge", GREEN), ("Discharge", RED), ("Idle", MUTED)]:
        subset = sched[sched["Action"] == action]["Price (€/MWh)"]
        if len(subset):
            fig_hist.add_trace(go.Histogram(
                x=subset, name=action,
                marker_color=colour, opacity=0.65, nbinsx=12,
            ))
    fig_hist.update_layout(
        **CHART_LAYOUT, height=260, barmode="overlay",
        xaxis=dict(title="Price (€/MWh)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Hours", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Demand constraint impact comparison
    if constraint_active:
        st.markdown(f'<div style="font-size:13px;font-weight:500;color:{TEXT};margin:16px 0 10px;">Demand Constraint Impact</div>', unsafe_allow_html=True)

        # Quick unconstrained solve for comparison
        result_unc = run_bess_optimization(
            prices_series,
            max_power_mw    = max_power,
            duration_hrs    = duration,
            efficiency      = efficiency / 100,
            min_soc_pct     = min_soc    / 100,
            initial_soc_pct = init_soc   / 100,
            demand_limit_mw = max_power,   # no constraint
        )
        profit_unc  = result_unc["total_profit"]
        profit_loss = profit_unc - net_profit

        c1, c2, c3 = st.columns(3)
        c1.metric("Unconstrained Profit", f"€{profit_unc:,.2f}")
        c2.metric("Constrained Profit",   f"€{net_profit:,.2f}")
        c3.metric("Profit Lost to Constraint", f"−€{profit_loss:,.2f}",
                  delta=f"{-100*profit_loss/profit_unc:.1f}%" if profit_unc else "N/A",
                  delta_color="inverse")

    # Price stats
    st.markdown(f'<div style="font-size:13px;font-weight:500;color:{TEXT};margin:16px 0 10px;">Price Statistics</div>', unsafe_allow_html=True)
    stats = pd.DataFrame({
        "Metric": ["Min", "Max", "Mean", "Spread", "Std Dev"],
        "Value":  [
            f"€{prices_array.min():.2f} / MWh",
            f"€{prices_array.max():.2f} / MWh",
            f"€{prices_array.mean():.2f} / MWh",
            f"€{prices_array.max()-prices_array.min():.2f} / MWh",
            f"€{prices_array.std():.2f} / MWh",
        ]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True, height=212)


# ── Footer ────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:40px;padding-top:16px;border-top:1px solid {BORDER};
            font-size:11px;color:{MUTED};">
  BESS Optimizer &nbsp;·&nbsp; XGBoost + PuLP LP &nbsp;·&nbsp;
  Units: MW / MWh &nbsp;·&nbsp;
  Dataset: Spain Hourly Energy (Kaggle) &nbsp;·&nbsp; Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
