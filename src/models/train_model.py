import argparse, pandas as pd, joblib
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--X-train', default='data/processed/X_train_scaled.csv')
parser.add_argument('--y-train', default='data/processed/y_train.csv')
parser.add_argument('--params', default='models/best_params.pkl')
args = parser.parse_args()

X_train = pd.read_csv(args.X_train)
y_train = pd.read_csv(args.y_train).squeeze()
params = joblib.load(args.params)

model = GradientBoostingRegressor(**params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'models/gbr_model.pkl')
