import argparse, pandas as pd, json, joblib
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--X-test', default='data/processed_data/X_test_scaled.csv')
parser.add_argument('--y-test', default='data/processed_data/y_test.csv')
parser.add_argument('--model', default='models/gbr_model.pkl')
parser.add_argument('--scaler', default='models/data/scaler.pkl')
args = parser.parse_args()

X_test = pd.read_csv(args.X_test)
y_test = pd.read_csv(args.y_test).squeeze()
model = joblib.load(args.model)
scaler = joblib.load(args.scaler)  # Non utilisé ici mais pour cohérence

y_pred = model.predict(X_test)

scores = {
    'mse': float(mean_squared_error(y_test, y_pred)),
    'r2': float(r2_score(y_test, y_pred))
}

Path('metrics').mkdir(exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

pd.DataFrame({'y_test': y_test, 'predictions': y_pred}).to_csv('data/predictions.csv', index=False)
