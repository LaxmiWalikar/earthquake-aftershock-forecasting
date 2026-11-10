"""Step 3: train the neural network. Every earthquake is predicted by a model that never saw it (5-fold, grouped by earthquake)."""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features import FEATURES

meta = json.load(open("data/meta.json")); ids = list(meta)
data = pd.concat([pd.read_csv(f"data/processed/{i}.csv") for i in ids], ignore_index=True)
data["nn"] = 0.0; data["base"] = data["f1"]
for k, (tr, te) in enumerate(GroupKFold(5).split(data, groups=data["event"])):
    t = data.iloc[tr]; pos, neg = t[t.label == 1], t[t.label == 0]
    t = pd.concat([pos, neg.sample(min(len(neg), 3 * len(pos)), random_state=0)]).sample(frac=1, random_state=0)   # balance classes
    m = make_pipeline(StandardScaler(), MLPClassifier((64, 64, 32), early_stopping=True, max_iter=200, random_state=0))
    m.fit(t[FEATURES], t.label)
    data.loc[data.index[te], "nn"] = m.predict_proba(data.iloc[te][FEATURES])[:, 1]
    print(f"fold {k + 1}/5 done")

a_nn, a_base = [], []
for i in ids:
    d = data[data.event == i]
    if d.label.nunique() == 2:
        meta[i]["auc_nn"], meta[i]["auc_base"] = roc_auc_score(d.label, d.nn), roc_auc_score(d.label, d.base)
        a_nn.append(meta[i]["auc_nn"]); a_base.append(meta[i]["auc_base"])
    d[["x", "y", "label", "nn", "base"]].to_csv(f"data/processed/{i}.csv", index=False)
summary = dict(n_events=len(ids), auc_nn=float(np.mean(a_nn)), auc_base=float(np.mean(a_base)),
               ai_wins=float(np.mean(np.array(a_nn) > np.array(a_base))))
json.dump(meta, open("data/meta.json", "w"), indent=1); json.dump(summary, open("data/summary.json", "w"), indent=1)
print(summary)
