import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--X-train', default='data/processed_data/X_train.csv')
parser.add_argument('--X-test', default='data/processed_data/X_test.csv')
args = parser.parse_args()

X_train = pd.read_csv(args.X_train)
X_test = pd.read_csv(args.X_test)

print("Colonnes X_train:", X_train.columns.tolist())
print("Types:", X_train.dtypes)

# SUPPRIMEZ la colonne date (non numérique)
if 'date' in X_train.columns:
    print("✅ Suppression de la colonne 'date'")
    X_train_num = X_train.drop('date', axis=1)
    X_test_num = X_test.drop('date', axis=1)
else:
    X_train_num = X_train
    X_test_num = X_test

print("X_train après suppression date:", X_train_num.shape)
print("Types numériques:", X_train_num.dtypes)

# Vérification finale : toutes les colonnes doivent être numériques
assert X_train_num.select_dtypes(include=[np.number]).shape[1] == X_train_num.shape[1], "Colonnes non-numériques restantes !"

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

Path('models/data').mkdir(exist_ok=True, parents=True)
joblib.dump(scaler, 'models/data/scaler.pkl')

pd.DataFrame(X_train_scaled, columns=X_train_num.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test_num.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)

print("✅ Scaling terminé !")
