import os
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


# ================================
# CONFIG
# ================================
ZIP_PATH = "icumortal.zip"
EXTRACT_PATH = "icumortal"
DECISION_THRESHOLD = 0.20


# ================================
# FEATURE ENGINEERING
# ================================
def engineer_features(file_path):
    df = pd.read_csv(file_path)

    record_id = int(df[df['Parameter'] == 'RecordID']['Value'].values[0])

    data = df[df['Parameter'] != 'RecordID'].copy()
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')

    data = data[data['Value'] != -1]
    data = data.dropna(subset=['Value'])

    stats = data.groupby('Parameter')['Value'].agg(['mean', 'max', 'min', 'std', 'last', 'count'])

    feature_dict = {'RecordID': record_id}
    for param in stats.index:
        for stat in ['mean', 'max', 'min', 'std', 'last', 'count']:
            feature_dict[f"{param}_{stat}"] = stats.loc[param, stat]

    return feature_dict


# ================================
# MAIN PIPELINE
# ================================
def main():
    print("🚀 Starting ICU Mortality Pipeline...\n")

    # ---------- Extract Data ----------
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(EXTRACT_PATH)
        print("✅ Data extracted")
    else:
        raise FileNotFoundError("icumortal.zip not found")

    outcomes_path = os.path.join(EXTRACT_PATH, "Outcomes-a.txt")

    # Check nested structure first (set-a/set-a), then flat (set-a)
    set_a_path = os.path.join(EXTRACT_PATH, "set-a", "set-a")
    if not os.path.exists(set_a_path):
        set_a_path = os.path.join(EXTRACT_PATH, "set-a")

    # ---------- Feature Engineering ----------
    patient_files = [f for f in os.listdir(set_a_path) if f.endswith('.txt')]
    features_list = []

    print("⚙️ Processing patient files...")
    for file_name in tqdm(patient_files):
        f_path = os.path.join(set_a_path, file_name)
        try:
            features_list.append(engineer_features(f_path))
        except Exception:
            continue

    df_features = pd.DataFrame(features_list)

    # ---------- Merge ----------
    df_outcomes = pd.read_csv(outcomes_path)
    df_master = pd.merge(df_features, df_outcomes[['RecordID', 'In-hospital_death']], on='RecordID')

    print("✅ Feature matrix:", df_master.shape)

    # ---------- EDA ----------
    plt.figure()
    sns.countplot(x='In-hospital_death', data=df_master)
    plt.savefig("class_distribution.png")
    plt.close()

    missing = df_master.isnull().mean().sort_values(ascending=False).head(30)
    plt.figure()
    missing.plot(kind='bar')
    plt.savefig("missing_features.png")
    plt.close()

    # ---------- Preprocessing ----------
    X = df_master.drop(columns=['RecordID', 'In-hospital_death'])
    y = df_master['In-hospital_death']

    X = X.loc[:, X.isnull().mean() < 0.7]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ---------- Models ----------
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)

    model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    model.fit(X_train, y_train)

    # ---------- Evaluation ----------
    def get_metrics(m, X, y):
        probs = m.predict_proba(X)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        return {
            "Accuracy": accuracy_score(y, preds),
            "Precision": precision_score(y, preds, zero_division=0),
            "Recall": recall_score(y, preds, zero_division=0),
            "F1": f1_score(y, preds, zero_division=0),
            "AUC": roc_auc_score(y, probs)
        }

    results = {
        "RF": get_metrics(rf, X_test, y_test),
        "XGB": get_metrics(xgb, X_test, y_test),
        "Ensemble": get_metrics(model, X_test, y_test)
    }

    print("\n📊 MODEL RESULTS")
    print(pd.DataFrame(results).T.round(3))

    # ---------- Plots ----------
    plt.figure()
    for name, m in [("RF", rf), ("XGB", xgb), ("ENS", model)]:
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=name)
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_test, (model.predict_proba(X_test)[:, 1] >= DECISION_THRESHOLD).astype(int))
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ---------- SHAP ----------
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap.png")
    plt.close()

    # ---------- Save Predictions ----------
    probs = model.predict_proba(X_test)[:, 1]

    def risk(p):
        if p < 0.20:
            return "Low"
        elif p <= 0.50:
            return "Medium"
        else:
            return "High"

    output = pd.DataFrame({
        "True": y_test,
        "Pred": (probs >= DECISION_THRESHOLD).astype(int),
        "Prob": probs,
        "Risk": [risk(p) for p in probs]
    })

    output.to_csv("icu_predictions.csv", index=False)

    print("\n✅ Pipeline Complete")
    print("📁 Files generated:")
    print("- icu_predictions.csv")
    print("- roc_curve.png")
    print("- confusion_matrix.png")
    print("- shap.png")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    main()