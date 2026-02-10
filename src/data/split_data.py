import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/raw/raw.csv')
parser.add_argument('--test-size', type=float, default=0.2)
args = parser.parse_args()

print(f"Lecture du fichier: {args.input}")

if not Path(args.input).exists():
    raise FileNotFoundError(f"Fichier non trouvé: {args.input}")

df = pd.read_csv(args.input)
print(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("Colonnes:", df.columns.tolist())

# IDENTIFICATION DES FEATURES : TOUTES les colonnes sauf 'silica_concentrate'
feature_cols = [col for col in df.columns if col != 'silica_concentrate']
print(f"Features sélectionnées: {feature_cols}")

X = df[feature_cols]
y = df['silica_concentrate']

# Vérification que X ne contient que des numériques
print("Types de données X:")
print(X.dtypes)
print("\nÉchantillon X:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42
)

Path('data/processed_data').mkdir(exist_ok=True, parents=True)
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("✅ Split terminé ! Fichiers sauvegardés dans data/processed/")
