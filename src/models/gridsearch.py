import argparse, pandas as pd, joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--X-train', default='data/processed_data/X_train_scaled.csv')
parser.add_argument('--y-train', default='data/processed_data/y_train.csv')
args = parser.parse_args()

X_train = pd.read_csv(args.X_train)
y_train = pd.read_csv(args.y_train).squeeze()

gbr = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, 'models/best_params.pkl')
print(f"Meilleurs params: {grid.best_params_}")
