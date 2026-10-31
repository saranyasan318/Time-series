
import streamlit as st

from config import THEME

T = THEME


def inject_css():
    st.markdown(f"""<style>
    .stApp {{background:#FAFBFE;color:{T['navy']};font-family:Inter,'Segoe UI',sans-serif}}
    .block-container {{padding-top:1.6rem;max-width:1280px}}
    section[data-testid="stSidebar"] {{background:linear-gradient(180deg,{T['navy']} 0%,#16294F 100%)}}
    section[data-testid="stSidebar"] * {{color:#E7ECFA !important}}
    section[data-testid="stSidebar"] .stRadio label {{padding:6px 10px;border-radius:10px;transition:background .2s}}
    section[data-testid="stSidebar"] .stRadio label:hover {{background:rgba(255,255,255,.08)}}
    h1,h2,h3 {{color:{T['navy']};letter-spacing:-.015em}}
    .hero {{background:linear-gradient(120deg,{T['navy']} 0%,#2B3F8F 55%,{T['accent']} 100%);border-radius:22px;padding:34px 38px;color:white;margin-bottom:22px;box-shadow:0 10px 30px rgba(11,27,58,.18)}}
    .hero h1 {{color:white;font-size:2.1rem;margin:0 0 6px 0}} .hero p {{color:#DCE4FF;font-size:1.02rem;margin:0;max-width:760px}}
    .kpi {{background:white;border-radius:16px;padding:16px 18px;box-shadow:0 2px 12px rgba(11,27,58,.07);border-left:4px solid {T['accent']};height:100%;transition:all .25s}}
    .kpi:hover {{transform:translateY(-2px);box-shadow:0 8px 20px rgba(11,27,58,.12)}}
    .kpi .l {{font-size:.70rem;color:{T['muted']};text-transform:uppercase;letter-spacing:.07em;font-weight:600}}
    .kpi .v {{font-size:1.45rem;font-weight:700;color:{T['navy']};margin-top:2px;word-break:break-word}}
    .kpi .s {{font-size:.78rem;color:{T['muted']};margin-top:2px}}
    .kpi.good {{border-left-color:#2E9E6B}} .kpi.bad {{border-left-color:#D9534F}} .kpi.warn {{border-left-color:#E0A84F}}
    .note {{background:#EEF3FF;border-radius:12px;padding:11px 15px;color:{T['navy']};font-size:.9rem;margin:6px 0 12px 0}}
    .step {{background:white;border-radius:16px;padding:16px 18px;box-shadow:0 2px 12px rgba(11,27,58,.06);height:100%}}
    .step b {{color:{T['accent']}}} .step .done {{color:#2E9E6B;font-weight:700}} .step .todo {{color:{T['muted']}}}
    .insight {{background:white;border-radius:14px;padding:13px 17px;margin-bottom:10px;box-shadow:0 1px 8px rgba(11,27,58,.06);border-left:5px solid {T['lavender']}}}
    .insight.positive {{border-left-color:#2E9E6B}} .insight.warning {{border-left-color:#E0A84F}} .insight.info {{border-left-color:{T['accent']}}}
    .insight .t {{font-weight:700}} .insight .c {{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:{T['muted']};margin-left:8px}}
    .insight .e {{font-size:.8rem;color:{T['muted']};margin-top:3px}}
    .sub {{color:{T['muted']};margin-top:-8px;margin-bottom:16px}}
    </style>""", unsafe_allow_html=True)


def header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p class="sub">{subtitle}</p>', unsafe_allow_html=True)


def fmt(v, pct=False):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "n/a"
    out = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    return out + ("%" if pct else "")


def kpi(col, label, value, sub=None, tone=""):
    s = f'<div class="s">{sub}</div>' if sub else ""
    col.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div>{s}</div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        kpi(c, *it)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def plot(fig, key=None):
    """st.plotly_chart across Streamlit versions (use_container_width was replaced by width='stretch')."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def download(label, data, filename, mime, key=None):
    try:
        st.download_button(label, data, filename, mime, key=key or filename, width="stretch")
    except TypeError:
        st.download_button(label, data, filename, mime, key=key or filename, use_container_width=True)


def need_data():
    from ui.state import S, ready
    if not ready():
        st.warning("No valid dataset yet. Open **Dataset Upload**, upload a CSV (or load a sample) and resolve any validation errors.")
        st.stop()
    return S["series"]


def need_results():
    from ui.state import S
    need_data()
    if not S["ev"] or not S["fcs"]:
        st.warning("No forecasts yet. Run the **Forecasting Lab** first.")
        st.stop()
    return S["ev"]
