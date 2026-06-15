# Movie Recommender System — Run Instructions

## 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run project phases (in order)

From the project root:

```bash
python3 main.py --phase 2 --data-dir data
python3 main.py --phase 3 --data-dir data
python3 main.py --phase 4 --data-dir data
python3 main.py --phase 5 --data-dir data
python3 main.py --phase 6 --data-dir data
python3 main.py --phase 7 --data-dir data
```

## 3) Run Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

## 4) Main output locations

- Processed data: `data/processed/`
- Charts: `outputs/charts/`
- Tables/metrics: `outputs/tables/`
- Saved model: `outputs/models/svd_model.joblib`
