"""M8: Streamlit dashboard for DrillPilot. Consumes the M7 FastAPI backend's
/predict and /explain (never calls ml.inference or the model directly -- the
backend is the only thing that touches the production model, see
docs/adr/004-inference-input-contract.md and docs/adr/005-shap-endpoint-design.md).

Both confidence flags the backend exposes (known_limitation_zone,
insufficient_history) are rendered as visible warnings, not just fields in a JSON
blob -- a drilling engineer glancing at the dashboard needs to see at a glance when
to trust a prediction less.

Run: streamlit run frontend/streamlit_app/app.py
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from frontend.streamlit_app import api_client, sample_data
from frontend.streamlit_app.formatting import confidence_warnings
from ml.features.pipeline import DEFAULT_ROLLING_WINDOW

DEFAULT_BACKEND_URL = os.environ.get("DRILLPILOT_BACKEND_URL", "http://localhost:8000")


@st.cache_data(show_spinner="Cargando dataset USROP...")
def _cached_dataset() -> pd.DataFrame:
    from ml.features.dataset import load_combined_dataset

    return load_combined_dataset()


def _init_readings_state() -> None:
    if "readings_df" not in st.session_state:
        df = _cached_dataset()
        sample = sample_data.load_sample_window(df, well_id=3, start_row=0, n_rows=10)
        st.session_state["readings_df"] = pd.DataFrame(sample)


def _render_sample_loader() -> None:
    st.subheader("1. Ventana de lecturas")
    st.caption(
        "El modelo necesita una ventana de lecturas del mismo pozo, no una lectura "
        f"aislada (recomendado >= {DEFAULT_ROLLING_WINDOW} filas). Cargá un ejemplo "
        "real del dataset USROP o editá la tabla manualmente."
    )
    cols = st.columns([1, 1, 1, 1])
    well_id = cols[0].selectbox("Pozo", sample_data.AVAILABLE_WELL_IDS, index=3)
    start_row = cols[1].number_input("Fila inicial", min_value=0, value=0, step=1)
    n_rows = cols[2].number_input("Cantidad de filas", min_value=1, value=10, step=1)
    if cols[3].button("Cargar ejemplo real", width="stretch"):
        df = _cached_dataset()
        sample = sample_data.load_sample_window(
            df, well_id=int(well_id), start_row=int(start_row), n_rows=int(n_rows)
        )
        st.session_state["readings_df"] = pd.DataFrame(sample)

    st.session_state["readings_df"] = st.data_editor(
        st.session_state["readings_df"],
        num_rows="dynamic",
        width="stretch",
        key="readings_editor",
    )


def _render_predict_warnings(response: dict[str, Any]) -> None:
    predictions = response["predictions"]
    any_known_limitation_zone = any(p["known_limitation_zone"] for p in predictions)
    for message in confidence_warnings(
        known_limitation_zone=any_known_limitation_zone,
        insufficient_history=response["insufficient_history"],
    ):
        st.warning(message)


def _render_predict_results(response: dict[str, Any]) -> None:
    _render_predict_warnings(response)
    table = pd.DataFrame(response["predictions"])
    table["known_limitation_zone"] = table["known_limitation_zone"].map({True: "⚠️ si", False: "no"})
    table = table.rename(
        columns={
            "md": "MD (m)",
            "predicted_rop": "ROP predicho (m/h)",
            "known_limitation_zone": "Zona de alto error",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)


def _render_explain_results(response: dict[str, Any]) -> None:
    for message in confidence_warnings(
        known_limitation_zone=response["known_limitation_zone"],
        insufficient_history=response["insufficient_history"],
    ):
        st.warning(message)

    st.metric(
        "ROP predicho (m/h)", f"{response['predicted_rop']:.2f}", help=f"MD={response['md']:.2f} m"
    )

    contributions = pd.DataFrame(response["contributions"]).set_index("feature")
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")
    st.bar_chart(contributions["shap_value"])
    st.caption(
        f"Valor base: {response['base_value']:.2f} + suma de contribuciones = "
        f"prediccion ({response['predicted_rop']:.2f}) -- propiedad de aditividad de SHAP."
    )


def main() -> None:
    st.set_page_config(page_title="DrillPilot", page_icon="🛢️", layout="wide")
    st.title("🛢️ DrillPilot -- predicción de ROP")

    with st.sidebar:
        st.header("Backend")
        backend_url = st.text_input("URL del backend", value=DEFAULT_BACKEND_URL)
        if st.button("Verificar conexión"):
            try:
                status = api_client.health(backend_url)
                if status["model_loaded"]:
                    st.success(f"Conectado -- modelo cargado ({status['model_uri']})")
                else:
                    st.warning(
                        "Backend disponible pero el modelo no esta cargado (status=degraded)."
                    )
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                st.error(f"No se pudo conectar al backend: {exc}")

    _init_readings_state()
    _render_sample_loader()

    st.subheader("2. Predicción")
    predict_col, explain_col = st.columns(2)

    if predict_col.button("Predecir", type="primary", width="stretch"):
        readings = sample_data.dataframe_to_readings(st.session_state["readings_df"])
        try:
            response = api_client.predict(backend_url, readings)
            _render_predict_results(response)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            st.error(f"Error en /predict: {exc}")

    if explain_col.button("Explicar (SHAP)", width="stretch"):
        readings = sample_data.dataframe_to_readings(st.session_state["readings_df"])
        try:
            response = api_client.explain(backend_url, readings)
            _render_explain_results(response)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            st.error(f"Error en /explain: {exc}")


main()
