import streamlit as st

st.set_page_config(page_title="Time Series Intelligence", layout="wide", initial_sidebar_state="expanded")

from ui import pages_data as pd_, pages_forecast as pf, pages_q3 as pq3  # noqa: E402  (after set_page_config)
from ui import state as stt  # noqa: E402
from ui.components import inject_css  # noqa: E402

inject_css()
stt.init()

ROUTES = {"Overview Dashboard": pd_.overview, "Dataset Upload": pd_.upload, "Data Validation & Quality": pd_.validation,
          "Exploratory Data Analysis": pd_.exploratory, "Data Preparation": pd_.preparation, "Forecasting Lab": pf.lab,
          "Model Evaluation & Comparison": pf.evaluation, "Prediction Dashboard": pf.prediction,
          "Business Intelligence & Insights": pf.insights, "Reports & Downloads": pf.reports,
          "Q3 – Time Series Decision Challenge": pq3.page}

st.sidebar.markdown("### Time Series Intelligence")
st.sidebar.caption("Forecasting & decision intelligence")
st.sidebar.radio("Navigate", stt.PAGES, key="nav", label_visibility="collapsed")
if stt.S["raw"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Dataset:** {stt.S['dataset_name']}")
    st.sidebar.caption(f"**Series:** {len(stt.S['series']) if stt.S['series'] is not None else '—'} periods · {stt.S['freq']}")
    if stt.S["ev"]:
        st.sidebar.caption(f"**Best model:** {stt.S['ev']['selected']}")

try:
    ROUTES[stt.S["nav"]]()
except Exception as exc:  # last-resort guard: show a readable error instead of a raw traceback wall
    st.error(f"Something went wrong on this page: {type(exc).__name__}: {exc}")
    with st.expander("Technical details"):
        st.exception(exc)
