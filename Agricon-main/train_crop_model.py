"""Train RandomForest crop recommender from Data/crop_recommendation.csv → models/RandomForest.pkl"""
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    csv_path = os.path.join(BASE_DIR, "Data", "crop_recommendation.csv")
    df = pd.read_csv(csv_path)
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    out_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "RandomForest.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved {out_path} (holdout accuracy: {acc:.4f})")


if __name__ == "__main__":
    main()
