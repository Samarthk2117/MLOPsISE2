# Student Performance Analysis

## Project overview
This project demonstrates:
- Training a student-performance regression model using a reproducible synthetic dataset.
- Tracking model parameters, evaluation metrics, and artifacts with MLflow.
- Comparing reference and simulated current feature distributions using the two-sample Kolmogorov–Smirnov (KS) test.
- Saving drift results as CSV, JSON, and PNG files.

**Important:** All data is synthetic and intended only for learning MLflow and monitoring workflows. The model is not validated for real student assessment.

## Repository structure
```text
student-performance-mlflow-drift/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── artifacts/                  # generated after running main.py
    ├── reference_data.csv
    ├── simulated_current_data.csv
    ├── model_predictions.csv
    ├── drift_results.csv
    ├── drift_summary.json
    └── drift_plot.png
```

## Requirements
- Python 3.10 or 3.11 recommended.
- Internet access for installing packages the first time.
- No external dataset download is required.

## Run on Windows PowerShell
```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Open the MLflow UI
After `main.py` finishes, open a second terminal in the project folder and run:
```powershell
python -m mlflow ui --backend-store-uri ./mlruns --port 5001
```
Open http://127.0.0.1:5001 in your browser. You should see the `Student_Performance_Data_Drift_Analysis` experiment with baseline and drift runs. Verify this on your own machine.

## What the code does
1. Creates 240 synthetic reference rows with five numeric features and a score target.
2. Trains a `RandomForestRegressor` and calculates MAE, RMSE, and R² on a held-out test set.
3. Logs parameters, metrics, a prediction CSV, and the fitted model to MLflow.
4. Creates a second synthetic dataset and deliberately shifts study hours and attendance to demonstrate drift.
5. Runs a two-sample KS test for each input feature. A p-value below 0.05 is flagged as a statistically significant distribution difference.
6. Logs drift metrics and CSV/JSON/PNG artifacts to MLflow.

A drift flag is not proof that model performance has decreased. Investigate changes and assess model performance before deciding whether retraining is needed.

## GitHub setup
Create an empty GitHub repository, then run these commands from this folder:
```powershell
git init
git add main.py requirements.txt README.md .gitignore
git commit -m "Add MLflow experiment tracking and data drift analysis"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
Replace the example remote URL with your actual repository URL.

Generated local files (`mlruns/`, `.venv/`, and output artifacts) are excluded by `.gitignore` by default. If your instructor requires screenshots or output evidence, run the code and add selected generated files or screenshots after confirming their contents.

## Troubleshooting
- If package installation fails, check `python --version` and try Python 3.10 or 3.11.
- If port 5001 is occupied, use `--port 5002` and open `http://127.0.0.1:5002`.
- If MLflow model logging reports a compatibility issue, update packages using `pip install -r requirements.txt`.
