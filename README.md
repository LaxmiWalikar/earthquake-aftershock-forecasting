# Aftershock Location Forecast (prototype)

Uses the 118 largest earthquakes (M7.0+, 2000-2024) from USGS, so it is a similar dataset to the Google study, not the identical list.

## Run (4 steps)
```
pip install -r requirements.txt
python data.py        # downloads the earthquakes (internet needed, may take 10-30 minutes)
python features.py    # builds the grids and features
python train.py       # trains and tests the neural network
streamlit run app.py  # opens the dashboard in your browser
```
Offline test with fake data: `python data.py --synthetic` instead of step 1.

## Settings
Change number of earthquakes, magnitude, dates and grid size in `config.py`.

## Approximations (say these in review)
- Stress change is a simplified estimate, not a full Coulomb/Okada calculation.
- Fault strike comes from the aftershock cloud, and length from magnitude.
- If some events have too few aftershocks they are skipped, and the next largest is used.
