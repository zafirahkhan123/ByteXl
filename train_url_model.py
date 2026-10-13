
from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


data_path = Path("data/url/PhiUSIIL_Phishing_URL_Dataset.csv")
df = pd.read_csv(data_path, low_memory=False)

print("Dataset loaded:", df.shape)


df = df.dropna(subset=["label"]).copy()
y = df["label"].astype(int)


drop_cols = ["label", "FILENAME", "URL", "Domain", "Title"]
X = df.drop(columns=drop_cols, errors="ignore")


numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2
    ))
])


print("Training URL model...")
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))


models_dir = Path("Models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "url_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("\nSaved model to:", model_path.resolve())
