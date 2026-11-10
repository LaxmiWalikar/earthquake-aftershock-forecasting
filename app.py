"""Step 4: dashboard.  Run with:  streamlit run app.py"""
import json
import pandas as pd, streamlit as st
from plots import predicted_map, actual_map

st.set_page_config(page_title="Aftershock Forecast", page_icon="🌍", layout="wide")
st.markdown("""<style>
.hero{background:#0B1F3A;border-bottom:4px solid #FF8A3D;padding:18px 22px;border-radius:12px;margin-bottom:12px}
.hero h1{color:#fff;margin:0;font-size:1.7rem}.hero p{color:#C9D1DE;margin:.3rem 0 0;font-size:1.05rem}
</style><div class="hero"><h1>Where will the aftershocks strike?</h1>
<p>Pick an earthquake. The AI predicts where aftershocks are most likely, and we compare it with what really happened.</p></div>""", unsafe_allow_html=True)

meta = json.load(open("data/meta.json")); S = json.load(open("data/summary.json"))
order = sorted(meta, key=lambda k: -meta[k]["mag"])
key = st.sidebar.selectbox("Choose an earthquake", order, format_func=lambda k: meta[k]["name"])
m = meta[key]; df = pd.read_csv(f"data/processed/{key}.csv")

c = st.columns(3)
c[0].metric("Earthquakes in the dataset", S["n_events"])
c[1].metric("AI score (this earthquake)", f"{m['auc_nn']:.2f}" if "auc_nn" in m else "n/a")
c[2].metric("Traditional method score", f"{m['auc_base']:.2f}" if "auc_base" in m else "n/a")
st.caption("Score: 1.0 = perfect, 0.5 = random guessing.")

left, right = st.columns(2)
left.subheader("AI prediction"); left.pyplot(predicted_map(df, m, "nn"))
right.subheader("What really happened"); right.pyplot(actual_map(df, m))
st.info("Left: brighter areas are where the AI thinks aftershocks are more likely. Right: yellow squares are places where aftershocks really occurred. The closer the two look, the better the prediction. The white line is the mainshock fault.")

st.subheader(f"Overall, across all {S['n_events']} earthquakes")
o = st.columns(3)
o[0].metric("AI average score", f"{S['auc_nn']:.2f}")
o[1].metric("Traditional average score", f"{S['auc_base']:.2f}")
o[2].metric("Earthquakes where AI wins", f"{S['ai_wins']:.0%}")

with st.expander("How it works"):
    st.write("1. Download major earthquakes and their aftershocks.  2. Split the area around each into a grid.  "
             "3. Calculate how the mainshock changed stress in each square.  4. Train a neural network to predict which squares get aftershocks.")
st.caption("Prototype note: stress values are a simplified estimate, and fault direction and length are approximated.")
