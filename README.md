# Forecasting Earthquake Aftershock Locations with AI-Assisted Science

Prototype for IBM Review 1. A neural network learns where aftershocks are likely to occur around a major earthquake, and a web dashboard compares its predictions with what really happened.

Team: Laxmi C K (1WA25CS115)

     Lakshmikanta V L (1WA25CS116)
     
     Laxmi Danappa Rathod (1WA25CS117)
     
     Laxmi Walikar (1WA25CS118)
     
     Leela R (1WA25CS119)

## 1. The problem in simple words
After a big earthquake (the **mainshock**), smaller earthquakes called **aftershocks** keep happening for days or weeks and can collapse already damaged buildings. Science can already estimate **when** aftershocks occur (Omori law) and **how big** they are (Gutenberg-Richter law). Predicting **where** they occur is still hard.

## 2. Our approach
1. A mainshock changes the stress in the rock around its fault. Some areas are pushed closer to breaking.
2. We split the area around each earthquake into a 40 x 40 grid of squares.
3. For every square we calculate stress-based numbers (features) and record whether an aftershock really happened there (label: 1 = yes, 0 = no).
4. A neural network learns the link between the features and the label.
5. We test it on earthquakes it has never seen and compare it with a simple baseline standing in for the traditional Coulomb stress method.

## 3. Data (real, from USGS)
- **Source:** USGS Earthquake Catalog, public API: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Mainshocks:** the 118 largest earthquakes of magnitude 7.0 or higher between 2000-01-01 and 2024-12-31. This is similar to, but not necessarily identical to, the 118-earthquake set in the Google AI research.
- **Aftershocks:** all earthquakes of magnitude 4.0 or higher within 30 days after each mainshock, inside a search radius around it.
- Mainshocks with fewer than 20 aftershocks are skipped and the next largest is used.
- **No synthetic data is used.** The option `python data.py --synthetic` exists only for offline code testing and must never be used for reviews. Real data can be verified in `data/events.csv`: real IDs and place names (for example "Sumatra - Andaman Islands"); fake data would show IDs like `syn000`.

**What is real and what is estimated**

| Item | Status |
|---|---|
| Earthquake times, locations, magnitudes, aftershock records | Real (USGS) |
| Fault direction (strike) | Estimated from the aftershock cloud |
| Fault length | Estimated from magnitude (Wells and Coppersmith style formula) |
| Stress values | Simplified estimate (not a full Coulomb/Okada calculation) |

## 4. Tech stack

| Layer | Tool | Used for |
|---|---|---|
| Language | Python 3 | Everything |
| Data download | `requests` | Calling the USGS API |
| Data handling | `pandas`, `numpy` | Tables, grids, calculations |
| Machine learning | `scikit-learn` | Neural network, scaling, cross-validation, metrics |
| Maps | `matplotlib` | Prediction and truth maps |
| Dashboard | `streamlit` | Web interface |

## 5. Machine learning details
- **Model:** multi-layer perceptron neural network (`MLPClassifier`), 3 hidden layers of 64, 64 and 32 neurons. scikit-learn defaults apply (ReLU activation, Adam optimizer); early stopping and up to 200 iterations. Random seed is fixed for repeatable results.
- **Inputs (6 per grid square):**
  - `f1`: shear-like stress pattern (butterfly-shaped lobes around the fault, limited to -2 to 2). Also used as the baseline score.
  - `f2`: a second stress-pattern component.
  - `f3`: stress-pattern strength falling with distance.
  - `u`: position along the fault (normalized by fault half-length).
  - `v`: position across the fault (normalized).
  - `logr`: log of distance from the fault center.
- **Output:** one number from 0 to 1, the predicted chance of an aftershock in that square.
- **Scaling:** `StandardScaler` puts all inputs on a similar range.
- **Class imbalance:** squares with aftershocks are rare, so training keeps all of them and a random sample of "no aftershock" squares (up to 3 times as many).
- **Fair testing:** 5-fold cross-validation grouped by earthquake (`GroupKFold`). The model is always tested on earthquakes it never trained on, so it cannot memorize answers.
- **Metric:** ROC-AUC per earthquake (1.0 = perfect, 0.5 = random guessing), then averaged across earthquakes. The dashboard also shows the share of earthquakes where the AI beats the baseline.
- **Baseline:** the `f1` stress pattern used directly as the prediction, a simplified stand-in for Coulomb stress.

## 6. Project files

| File | Purpose |
|---|---|
| `config.py` | All settings (number of earthquakes, magnitude, dates, grid size) |
| `data.py` | Downloads mainshocks and aftershocks from USGS. Can be re-run: it skips files already downloaded |
| `features.py` | Builds grids, stress features and labels; estimates fault strike and length |
| `train.py` | Trains the neural network with grouped cross-validation, saves predictions and scores |
| `plots.py` | Draws the prediction and "what really happened" maps |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Libraries to install |

Folders created when running: `data/raw/` (downloaded aftershocks), `data/processed/` (grids, features, predictions), plus `data/events.csv`, `data/meta.json`, `data/summary.json`. Deleting `data/` frees space and the steps can be re-run.

## 7. How to run
```
pip install -r requirements.txt
python data.py        # downloads the 118 earthquakes (internet needed, about 10-30 minutes)
python features.py    # builds the grids, features and labels
python train.py       # trains and tests the neural network
streamlit run app.py  # opens the dashboard in the browser
```
For a quick trial on a slow laptop, set `N_EVENTS` in `config.py` to about 30.

## 8. How to read the dashboard
- **Sidebar:** choose an earthquake.
- **Top numbers:** earthquakes in the dataset, the AI score and the traditional-method score for the chosen earthquake.
- **Left map (AI prediction):** brighter = more likely to have aftershocks (brightness is relative within that earthquake's area).
- **Right map (what really happened):** yellow squares = aftershocks really occurred there. The white line is the mainshock fault.
- **Bottom section:** average scores across all earthquakes and how often the AI wins.

## 9. Limitations (please read)
- Stress values are a simplified estimate that assumes a strike-slip fault pattern. For other fault types (for example subduction earthquakes) it fits poorly, which can push the baseline score below 0.5.
- Fault direction is estimated from the aftershocks themselves, which gives the model a hint about where aftershocks are, so scores are likely optimistic.
- The model also sees distance from the fault, so it may partly learn "aftershocks are near the fault" rather than the stress physics.
- Each grid square can be tens of kilometers wide for large earthquakes.
- Results are probabilities, not exact predictions.

## 10. Next steps
1. Replace the stress estimate with a real Coulomb stress calculation (Okada slip model) using all 6 stress tensor components.
2. Use independent fault data (published focal mechanisms and slip models) instead of estimated strike and length.
3. Compare more models: random forest, gradient boosting, logistic regression, and check which inputs matter most.
4. Add precision-recall plots and results grouped by earthquake size.
5. Put the dashboard online (for example Streamlit Community Cloud).

## 11. Glossary
- **Mainshock:** the biggest earthquake in a sequence.
- **Aftershock:** a smaller earthquake that follows it.
- **Stress change:** how much the mainshock squeezed or relaxed the surrounding rock.
- **Neural network:** a model made of layers of small calculators that learns patterns from examples.
- **Training / testing:** learning from examples / checking on new examples.
- **Baseline:** a simple method we compare the AI against.
- **ROC-AUC:** a score from 0 to 1 for how well predictions rank true aftershock squares above the others.

## 12. Data credit
Earthquake data: U.S. Geological Survey (USGS) Earthquake Catalog.

