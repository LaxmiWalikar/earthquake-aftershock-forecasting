"""Earthquake Aftershock Forecasting — Streamlit dashboard."""

from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import eaf_engine as eng

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Earthquake Aftershock Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --navy: #12294a;
        --navy-dark: #0b1f3a;
        --teal: #2f6f7e;
        --orange: #c4703a;
        --paper: #f5f2ea;
      }
      .block-container { padding-top: 1.4rem; max-width: 1500px; }
      .hero {
        background: linear-gradient(120deg, var(--navy-dark), var(--navy));
        border-bottom: 4px solid var(--orange);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 16px;
        color: #fff;
      }
      .hero h1 { margin: 0; font-size: 1.8rem; letter-spacing: -0.02em; }
      .hero p { margin: 6px 0 0; color: #c9d8e2; font-size: 1rem; }
      .status-pill {
        display: inline-block; background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.18); border-radius: 999px;
        padding: 4px 10px; color: #dcebf1; font-size: .78rem; margin-top: 8px;
      }
      .banner {
        border-left: 4px solid var(--teal); background: #eef5f6;
        padding: 10px 14px; border-radius: 8px; margin: 8px 0 14px;
        color: #173b46;
      }
      .warnbox {
        border-left: 4px solid var(--orange); background: #fff4ea;
        padding: 10px 14px; border-radius: 8px; margin: 8px 0;
        color: #5b351f;
      }
      [data-testid="stMetricValue"] { font-size: 1.5rem; }
      [data-testid="stMetricLabel"] { font-size: .78rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_mainshock_defaults() -> None:
    rows = st.session_state.get("catalog_rows") or []
    meta = st.session_state.get("catalog_meta") or {}
    main = eng.find_mainshock(rows, st.session_state.get("mainshock_id"))
    if main is None:
        return
    reference = meta.get("latestTime") or main["time"]
    st.session_state.magnitude = round(float(main["mag"]), 1)
    st.session_state.depth = int(eng.clamp(round(main["depth"]), 0, 700))
    st.session_state.elapsed = int(eng.clamp(round((reference - main["time"]) / 3600000.0), 0, 720))
    st.session_state.region = "catalog"


def set_catalog(rows, meta) -> None:
    st.session_state.catalog_rows = rows
    st.session_state.catalog_meta = meta
    options = eng.mainshock_options(rows)
    st.session_state.mainshock_id = options[0]["id"] if options else None
    apply_mainshock_defaults()


def init_state() -> None:
    if "catalog_rows" not in st.session_state:
        rows, meta = eng.load_catalog("live-week")
        set_catalog(rows, meta)
    model = eng.load_model()
    if "engine" not in st.session_state:
        st.session_state.engine = "ml" if model else "baseline"
    for layer in ("layer_grid", "layer_events", "layer_fault"):
        if layer not in st.session_state:
            st.session_state[layer] = True
    if "needs_saved" not in st.session_state:
        st.session_state.needs_saved = None


init_state()
model = eng.load_model()
rows = st.session_state.catalog_rows
meta = st.session_state.catalog_meta
analysis = eng.compute_analysis(rows, st.session_state.get("mainshock_id"), meta.get("latestTime"))

st.markdown(
    """
    <div class="hero">
      <h1>Earthquake Aftershock Forecasting</h1>
      <p>App-first USGS CSV prototype with a bundled Random Forest, transparent baseline, and local relief tools.</p>
      <span class="status-pill">Prototype estimate · not a prediction or public-safety product</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset source")
    st.selectbox(
        "CSV feed",
        options=list(eng.FEEDS.keys()),
        format_func=lambda k: eng.FEEDS[k][0],
        key="catalog_feed",
    )
    if st.button("Load catalog", use_container_width=True):
        loaded_rows, loaded_meta = eng.load_catalog(st.session_state.catalog_feed)
        set_catalog(loaded_rows, loaded_meta)
        st.rerun()

    uploaded = st.file_uploader("Import USGS-format CSV", type=["csv"], key="catalog_upload")
    if uploaded is not None:
        payload = uploaded.getvalue()
        if st.session_state.get("uploaded_bytes") != payload:
            try:
                imported_rows, imported_meta = eng.parse_usgs_csv(
                    payload.decode("utf-8-sig"),
                    uploaded.name,
                    "Local CSV import",
                    "Local CSV import",
                )
                set_catalog(imported_rows, imported_meta)
                st.session_state.uploaded_bytes = payload
                st.rerun()
            except Exception as exc:
                st.error(f"Could not import CSV: {exc}")

    status_kind = "fallback" if meta.get("fallback") else meta.get("kind", "catalog")
    st.caption(f"{status_kind} · {meta.get('count', 0):,} events · {meta.get('label', '')}")
    if meta.get("fallback"):
        st.warning("Live feed unavailable; bundled USGS snapshot is in use.")

    st.divider()
    st.header("Scenario")
    options = eng.mainshock_options(rows)
    option_ids = [row["id"] for row in options]

    def format_event(event_id: str) -> str:
        event = eng.find_mainshock(rows, event_id)
        if not event:
            return event_id
        place = event["place"] or "USGS event"
        return f"M{event['mag']:.1f} · {place[:44]}"

    if option_ids:
        st.selectbox(
            "Mainshock from catalog",
            options=option_ids,
            format_func=format_event,
            key="mainshock_select",
            on_change=lambda: (
                st.session_state.__setitem__("mainshock_id", st.session_state.mainshock_select),
                apply_mainshock_defaults(),
            ),
        )
    else:
        st.info("No catalog rows loaded.")

    st.markdown("**Schematic presets**")
    preset_cols = st.columns(3)
    for col, (key_name, cfg) in zip(preset_cols, eng.PRESETS.items()):
        if col.button(eng.REGIONS[key_name]["name"], use_container_width=True, key=f"preset_{key_name}"):
            st.session_state.magnitude = cfg["magnitude"]
            st.session_state.depth = cfg["depth"]
            st.session_state.elapsed = cfg["elapsed"]
            st.session_state.window = cfg["window"]
            st.session_state.region = key_name
            st.rerun()

    st.slider("Mainshock magnitude", 3.0, 9.0, value=float(st.session_state.magnitude), step=0.1, key="magnitude")
    st.number_input("Depth (km)", 0, 700, value=int(st.session_state.depth), step=1, key="depth")
    st.number_input("Hours since mainshock", 0, 720, value=int(st.session_state.elapsed), step=1, key="elapsed")
    st.radio(
        "Forecast window",
        options=list(eng.WINDOWS.keys()),
        format_func=lambda k: eng.WINDOWS[k]["short"],
        horizontal=True,
        key="window",
    )
    st.selectbox(
        "Region context",
        options=list(eng.REGIONS.keys()),
        format_func=lambda k: eng.REGIONS[k]["name"],
        key="region",
    )

    st.divider()
    st.header("Forecast engine")
    st.selectbox(
        "Active model",
        options=["ml", "baseline"],
        format_func=lambda k: (
            "Trained Random Forest v1.0" if k == "ml" else "Catalog-calibrated baseline v0.2"
        ),
        key="engine",
    )
    if st.session_state.engine == "ml" and model:
        st.caption(f"90 trees · {len(model.get('trees', []))} loaded · evaluated locally")
    elif st.session_state.engine == "ml" and not model:
        st.error("Bundled model unavailable; using baseline.")
        st.session_state.engine = "baseline"

    st.divider()
    st.header("Map layers")
    st.checkbox("Distance grid", key="layer_grid")
    st.checkbox("Catalog events", key="layer_events")
    st.checkbox("Inferred trend", key="layer_fault")


scenario = {
    "magnitude": float(st.session_state.magnitude),
    "depth": float(st.session_state.depth),
    "elapsed": float(st.session_state.elapsed),
    "window": st.session_state.window,
    "region": st.session_state.region,
}
features = eng.build_features(analysis, scenario["magnitude"], scenario["depth"], scenario["elapsed"], scenario["window"])
ml_count = eng.predict_ml(model, features)
baseline_count = eng.predict_baseline(
    analysis, scenario["magnitude"], scenario["depth"], scenario["elapsed"], scenario["window"]
)
expected_total = ml_count if st.session_state.engine == "ml" and ml_count is not None else baseline_count
scenario["expectedTotal"] = expected_total
grid = eng.build_grid(
    expected_total,
    analysis,
    scenario["magnitude"],
    scenario["depth"],
    scenario["elapsed"],
    scenario["region"],
)
rates, window_days = eng.daily_rates(
    expected_total,
    analysis,
    scenario["magnitude"],
    scenario["depth"],
    scenario["elapsed"],
    scenario["window"],
    st.session_state.engine,
)

main = analysis["main"] if analysis else None
if main:
    banner = (
        f"USGS live · {meta.get('label', '')} · {meta.get('count', 0):,} events · "
        f"mainshock M{main['mag']:.1f} · {main['place'] or 'USGS event'}"
    )
else:
    banner = "No catalog available. Load a feed or import a CSV in the sidebar."
st.markdown(f'<div class="banner">{html.escape(banner)}</div>', unsafe_allow_html=True)

tabs = st.tabs(["Forecast workspace", "Model comparison", "Relief & recovery", "Method & limits"])

with tabs[0]:
    engine_label = "Random Forest v1.0" if st.session_state.engine == "ml" and ml_count is not None else "baseline v0.2"
    st.subheader(f"Catalog outlook · {engine_label}")
    if analysis:
        st.caption(
            f"Mainshock M{main['mag']:.1f} · {main['depth']:.0f} km · "
            f"observed window {analysis['availableHours']:.0f} h · radius {analysis['radiusKm']:.0f} km"
        )
    else:
        st.caption("Waiting for a USGS-format catalog.")

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Peak cell estimate",
        f"{grid['peak']['probability'] * 100:.1f}%",
        help="Highest modelled chance of ≥1 M3+ event in a 3 km cell",
    )
    metric_cols[1].metric(
        "Expected aftershocks",
        f"{expected_total:.2f}",
        help="Catalog-scaled expected M≥3 count in the selected window",
    )
    metric_cols[2].metric("High-probability area", f"{grid['highAreaKm2']:.0f} km²", help="Cells at or above 50%")
    metric_cols[3].metric("Top priority zone", grid["zones"][0]["zone"] if grid["zones"] else "—")

    map_col, side_col = st.columns([3, 2])
    with map_col:
        st.markdown("#### Spatial probability surface")
        probs = np.array(grid["probabilities"]).reshape(eng.MAP["rows"], eng.MAP["cols"])
        fig, ax = plt.subplots(figsize=(10, 5.6))
        vmax = max(0.05, float(probs.max()))
        image = ax.imshow(
            probs,
            origin="upper",
            extent=[-60, 60, -37.5, 37.5],
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        if st.session_state.layer_events and analysis:
            early = eng.observed_through(analysis, scenario["elapsed"])
            if early:
                ex = [event["x"] for event in early[:180]]
                ey = [event["y"] for event in early[:180]]
                ax.scatter(ex, ey, s=12, c="#12294a", edgecolors="#fff", linewidths=0.4, zorder=4, label="Observed")
        if st.session_state.layer_fault:
            length = grid["region"]["faultLength"]
            theta = math.radians(grid["region"]["angle"])
            half = length / 2.0
            ax.plot(
                [-half * math.cos(theta), half * math.cos(theta)],
                [-half * math.sin(theta), half * math.sin(theta)],
                color="#fff",
                linestyle="--",
                linewidth=1.6,
                zorder=5,
                label="Inferred trend",
            )
        ax.scatter([grid["peak"]["x"]], [grid["peak"]["y"]], marker="*", s=150, c="#0b1f3a", edgecolors="#fff", zorder=6)
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_title("Estimated chance of ≥1 M≥3 event per 3 km cell")
        if st.session_state.layer_grid:
            ax.set_xticks(np.arange(-60, 61, 15))
            ax.set_yticks(np.arange(-30, 31, 15))
            ax.grid(color="#ffffff", alpha=0.25, linewidth=0.5)
        fig.colorbar(image, ax=ax, label="Probability")
        if analysis and (st.session_state.layer_events or st.session_state.layer_fault):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        st.pyplot(fig)

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=160)
        png_bytes = png_buf.getvalue()
        st.download_button(
            "Export map PNG",
            data=png_bytes,
            file_name=f"aftershock-map-{st.session_state.engine}-{scenario['region']}.png",
            mime="image/png",
        )
        plt.close(fig)

    with side_col:
        st.markdown("#### Priority zones")
        zone_df = pd.DataFrame(grid["zones"][:5])[
            ["zone", "probability", "expected", "level"]
        ].rename(
            columns={
                "zone": "Zone",
                "probability": "P(≥1)",
                "expected": "Expected",
                "level": "Relative priority",
            }
        )
        zone_df["P(≥1)"] = zone_df["P(≥1)"].map(lambda value: f"{value * 100:.1f}%")
        zone_df["Expected"] = zone_df["Expected"].map(lambda value: f"{value:.3f}")
        st.dataframe(zone_df, hide_index=True, use_container_width=True)
        st.caption("Rankings use scenario-relative thresholds. They are not validated alert levels.")

        st.markdown("#### Expected daily rate")
        rate_fig, rate_ax = plt.subplots(figsize=(7, 2.8))
        colors = ["#c4703a" if i < window_days else "#c9d4dc" for i in range(30)]
        rate_ax.bar(range(1, 31), rates, color=colors)
        rate_ax.set_xlabel("Day")
        rate_ax.set_ylabel("M≥3 / day")
        rate_ax.set_title("Omori-type decay scaled to the selected window")
        rate_fig.tight_layout()
        st.pyplot(rate_fig)
        plt.close(rate_fig)

    st.markdown("#### Catalog evidence")
    if analysis and analysis["observed"]:
        evidence_rows = []
        for event in analysis["observed"][:180]:
            row = event["row"]
            evidence_rows.append(
                {
                    "Time (UTC)": eng.iso_utc(row["time"])[:16].replace("T", " "),
                    "Mag": f"{row['mag']:.1f}",
                    "Depth": f"{row['depth']:.0f} km",
                    "Distance": f"{event['distance']:.1f} km",
                    "Place": row["place"] or "USGS event",
                    "USGS": row["id"],
                }
            )
        st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, use_container_width=True, height=240)
    else:
        st.info("No observed aftershock rows are available for this mainshock and time window.")

    st.markdown("#### Exports")
    export_cols = st.columns(3)
    grid_csv = eng.export_grid_csv(grid, scenario, meta, analysis, st.session_state.engine, model)
    export_cols[0].download_button(
        "Export grid CSV",
        data=grid_csv.encode("utf-8"),
        file_name=f"aftershock-grid-{st.session_state.engine}-{scenario['region']}-m{scenario['magnitude']:.1f}-{scenario['window']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    observed_csv = eng.export_observed_csv(analysis, meta, scenario["window"])
    export_cols[1].download_button(
        "Export observed CSV",
        data=observed_csv.encode("utf-8"),
        file_name=f"usgs-observed-{main['id'] if main else 'catalog'}-{scenario['window']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[2].download_button(
        "Export map PNG",
        data=png_bytes,
        file_name=f"aftershock-map-{st.session_state.engine}-{scenario['region']}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown(
        '<div class="warnbox">Prototype estimate only. Not validated, not an alert, and not for emergency response or public-safety decisions.</div>',
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader("Original model comparison")
    st.caption("The original 118-earthquake neural-network view is preserved alongside the full workspace.")
    try:
        meta_original = json.loads((BASE / "data" / "meta.json").read_text(encoding="utf-8"))
        summary_original = json.loads((BASE / "data" / "summary.json").read_text(encoding="utf-8"))
        from plots import actual_map, predicted_map

        order = sorted(meta_original, key=lambda key: -meta_original[key]["mag"])
        selected = st.selectbox(
            "Choose an earthquake",
            order,
            format_func=lambda key: meta_original[key]["name"],
            key="comparison_event",
        )
        selected_meta = meta_original[selected]
        selected_df = pd.read_csv(BASE / "data" / "processed" / f"{selected}.csv")
        score_cols = st.columns(3)
        score_cols[0].metric("Earthquakes in the dataset", summary_original["n_events"])
        score_cols[1].metric(
            "AI score (this earthquake)",
            f"{selected_meta['auc_nn']:.2f}" if "auc_nn" in selected_meta else "n/a",
        )
        score_cols[2].metric(
            "Traditional method score",
            f"{selected_meta['auc_base']:.2f}" if "auc_base" in selected_meta else "n/a",
        )
        st.caption("Score: 1.0 = perfect, 0.5 = random guessing.")
        left, right = st.columns(2)
        with left:
            st.subheader("AI prediction")
            st.pyplot(predicted_map(selected_df, selected_meta, "nn"))
        with right:
            st.subheader("What really happened")
            st.pyplot(actual_map(selected_df, selected_meta))
        st.subheader(f"Overall, across all {summary_original['n_events']} earthquakes")
        overall = st.columns(3)
        overall[0].metric("AI average score", f"{summary_original['auc_nn']:.2f}")
        overall[1].metric("Traditional average score", f"{summary_original['auc_base']:.2f}")
        overall[2].metric("Earthquakes where AI wins", f"{summary_original['ai_wins']:.0%}")
        st.caption(
            "Prototype note: stress values are simplified estimates, and fault direction and length are approximated."
        )
    except Exception as exc:
        st.info(f"The original comparison dataset is unavailable here: {exc}")

with tabs[2]:
    st.subheader("Relief & recovery")
    st.markdown(
        '<div class="warnbox"><strong>If anyone is injured or in immediate danger, call your local emergency number first</strong> '
        "(such as 112 or 911). Follow instructions from local authorities and official responders.</div>",
        unsafe_allow_html=True,
    )
    st.caption("Use official channels for urgent help. This section stores a checklist only in this browser session and never submits data to an agency.")

    relief_cols = st.columns(3)
    with relief_cols[0]:
        st.markdown("#### Shelter")
        st.markdown(
            "- [American Red Cross shelter locator](https://www.redcross.org/get-help/disaster-relief-and-recovery-services/find-an-open-shelter.html)\n"
            "- [FEMA disaster assistance](https://www.disasterassistance.gov/)\n"
            "- [211 local services](https://www.211.org/)"
        )
    with relief_cols[1]:
        st.markdown("#### Food & water")
        st.markdown(
            "- [Feeding America food bank finder](https://www.feedingamerica.org/find-your-local-foodbank)\n"
            "- [Red Cross relief services](https://www.redcross.org/get-help/disaster-relief-and-recovery-services.html)\n"
            "- [211 food and meals](https://www.211.org/)"
        )
    with relief_cols[2]:
        st.markdown("#### Basic needs")
        st.markdown(
            "- [USGS earthquake safety](https://www.usgs.gov/programs/earthquake-hazards/earthquake-safety)\n"
            "- [CDC emergency supplies](https://www.cdc.gov/emergency/prepare-supplies/index.html)\n"
            "- [Ready.gov earthquakes](https://www.ready.gov/earthquakes)"
        )

    st.markdown("#### Personal needs checklist")
    need_names = [
        "Safe shelter",
        "Water",
        "Food",
        "Medicine",
        "Blankets",
        "Phone charger",
        "First aid",
        "Documents",
    ]
    check_cols = st.columns(4)
    needs_state: dict[str, bool] = {}
    for idx, need in enumerate(need_names):
        with check_cols[idx % 4]:
            needs_state[need] = st.checkbox(need, key=f"need_{idx}")
    notes = st.text_area("Notes for yourself", placeholder="Example: need accessible shelter; water for 3 people", key="needs_notes")
    need_action_cols = st.columns(3)
    if need_action_cols[0].button("Save locally", use_container_width=True):
        st.session_state.needs_saved = {
            "needs": dict(needs_state),
            "notes": notes,
            "at": eng.iso_utc(st.session_state.get("_now_ms", 0) or __import__("time").time() * 1000),
        }
        st.success("Saved for this browser session only.")
    need_action_cols[1].download_button(
        "Export checklist CSV",
        data=eng.export_needs_csv(needs_state, notes).encode("utf-8"),
        file_name="relief-needs.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if need_action_cols[2].button("Clear", use_container_width=True):
        for idx in range(len(need_names)):
            st.session_state[f"need_{idx}"] = False
        st.session_state.needs_notes = ""
        st.session_state.needs_saved = None
        st.rerun()
    if st.session_state.needs_saved:
        st.caption(f"Last saved: {st.session_state.needs_saved['at']}")

    st.markdown("#### Printable family contact card")
    contact_fields = [
        "Meeting point",
        "Emergency contact",
        "Phone / alternate",
        "Out-of-area contact",
        "Medical needs",
    ]
    contact_values = []
    for field in contact_fields:
        contact_values.append(st.text_input(field, key=f"contact_{field.replace(' ', '_').replace('/', '_')}"))
    rows_html = "".join(
        f"<dt>{html.escape(field)}</dt><dd>{html.escape(value) or '&nbsp;'}</dd>"
        for field, value in zip(contact_fields, contact_values)
    )
    card_html = f"""
      <style>
        .card {{ border:1px solid #b7c6cf; padding:16px; border-radius:8px; font-family:Arial,sans-serif; }}
        .card h3 {{ margin:0 0 10px; }} .card dl {{ display:grid; grid-template-columns:150px 1fr; gap:8px; margin:0; }}
        .card dt {{ font-weight:700; }} .card dd {{ margin:0; border-bottom:1px solid #ccd; min-height:18px; }}
        @media print {{ body {{ margin:0; }} button {{ display:none; }} }}
      </style>
      <div class="card"><h3>Family contact card</h3><dl>{rows_html}</dl></div>
      <button onclick="window.print()" style="margin-top:10px;padding:8px 14px">Print contact card</button>
    """
    from streamlit.components.v1 import html as component_html

    component_html(card_html, height=280)

with tabs[3]:
    st.subheader("Method & limits")
    st.markdown(
        """
        **What the app does**

        The page loads an official USGS Earthquake Catalog CSV feed (past day, week, or month), with a bundled USGS CSV snapshot as fallback. You can also import a USGS-format CSV file. The selected earthquake becomes the mainshock. Observed rows after it set the search radius, infer a spatial trend, and scale an aftershock count model.

        **Trained Random Forest v1.0**

        The default engine is a compact regression Random Forest trained offline on a USGS monthly catalog. Its 90 trees are exported to JSON and evaluated locally — no API key, backend, or runtime model call is required. It predicts the expected count of M≥3 events from magnitude, depth, elapsed time, Omori decay, early observed counts, and inferred trend. A spatial kernel distributes that count over 3 km cells.

        `P(i) = 1 − exp(−N̂ × w(i))`

        **Catalog-calibrated baseline v0.2**

        The baseline uses an Omori–Utsu time decay, an anisotropic spatial kernel, and a productivity scale derived from observed M≥3 counts in the CSV. It is shown for comparison, not as an operational forecast.

        **Deliberately absent**

        - The Random Forest is a teaching prototype, not an operationally validated forecast system.
        - The holdout comparison is one time-ordered split of one monthly catalog, not an accuracy claim.
        - Region coordinates and the inferred fault trend are approximate, not authoritative hazard geometry.
        - The relief section does not dispatch help, collect personal data, or send requests to any agency.

        Use this tool to explore relative patterns in a CSV catalog, not to issue alerts or make safety decisions.
        """
    )
    if model and model.get("validation"):
        st.info(model["validation"].get("note", "Prototype validation note unavailable."))
    else:
        st.info("Bundled model validation metadata unavailable; baseline comparison remains selectable.")
