# finof_local.py — Local adaptation of finof.py (originally for Google Colab)
# Removed: !pip install, /content/ paths, google.colab imports

# Cell 2: Data Extraction and Path Verification
import zipfile
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(base_dir, 'icumortal.zip')
extract_path = os.path.join(base_dir, 'icumortal')

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    print(f"Extracted files to {extract_path}")
else:
    print(f"Zip file not found at {zip_path}")

set_a_path = os.path.join(extract_path, 'set-a', 'set-a')
outcomes_path = os.path.join(extract_path, 'Outcomes-a.txt')

if not os.path.exists(set_a_path):
    set_a_path = os.path.join(extract_path, 'set-a')

print(f"Patient files path: {set_a_path}")
print(f"Outcomes file path: {outcomes_path}")

# Cell 3: Parse patient files + feature engineering
import pandas as pd
import numpy as np
from tqdm import tqdm

def engineer_features(file_path):
    """Extracts statistical features from a single patient record."""
    df = pd.read_csv(file_path)
    record_id = int(df[df['Parameter'] == 'RecordID']['Value'].values[0])
    static_list = ['Age', 'Gender', 'Height', 'ICUtype', 'Weight']
    data = df[df['Parameter'] != 'RecordID'].copy()
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    data = data[data['Value'] != -1]
    data = data.dropna(subset=['Value'])
    stats = data.groupby('Parameter')['Value'].agg(['mean', 'max', 'min', 'std', 'last', 'count'])
    feature_dict = {'RecordID': record_id}
    for param in stats.index:
        for stat_name in ['mean', 'max', 'min', 'std', 'last', 'count']:
            feature_dict[f"{param}_{stat_name}"] = stats.loc[param, stat_name]
    return feature_dict

patient_files = [f for f in os.listdir(set_a_path) if f.endswith('.txt')]
features_list = []

print("Processing patient files in batches to optimize memory...")
for file_name in tqdm(patient_files):
    f_path = os.path.join(set_a_path, file_name)
    try:
        features_list.append(engineer_features(f_path))
    except Exception:
        continue

df_features = pd.DataFrame(features_list)
print(f"\nFeature matrix created: {df_features.shape}")

# Cell 4: Merge labels + EDA charts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for local execution
import matplotlib.pyplot as plt
import seaborn as sns

df_outcomes = pd.read_csv(outcomes_path)
df_master = pd.merge(df_features, df_outcomes[['RecordID', 'In-hospital_death']], on='RecordID', how='inner')

plt.figure(figsize=(6, 4))
sns.countplot(x='In-hospital_death', data=df_master, palette='viridis')
plt.title('Class Distribution (0: Alive, 1: Deceased)')
plt.savefig(os.path.join(base_dir, 'class_distribution.png'))
plt.close()

missing_data = df_master.isnull().mean().sort_values(ascending=False).head(30)
plt.figure(figsize=(10, 6))
missing_data.plot(kind='bar', color='salmon')
plt.title('Top 30 Features by Missingness Percentage')
plt.ylabel('Fraction of Missing Values')
plt.savefig(os.path.join(base_dir, 'missingness.png'))
plt.close()

print(f"Final Master Shape: {df_master.shape}")

# Cell 5: Preprocessing + SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X = df_master.drop(columns=['RecordID', 'In-hospital_death'])
y = df_master['In-hospital_death']

threshold = 0.7
X = X.loc[:, X.isnull().mean() < threshold]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Post-SMOTE Train size: {X_train_res.shape}")
print("\nClass balance in resampled training set:")
print(pd.Series(y_train_res).value_counts())

# Cell 6: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
print("Random Forest Training Complete.")

# Cell 7: Train XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_res, y_train_res)
print("XGBoost Training Complete.")

# Cell 8: Build Voting Ensemble
from sklearn.ensemble import VotingClassifier

vote_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'
)
vote_model.fit(X_train_res, y_train_res)
print("Ensemble (Soft Voting) Training Complete.")

# Cell 9: Evaluate all models + print metrics
DECISION_THRESHOLD = 0.20

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_metrics(model, X, y, threshold=DECISION_THRESHOLD):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        'Acc':  accuracy_score(y, preds),
        'Prec': precision_score(y, preds, zero_division=0),
        'Rec':  recall_score(y, preds, zero_division=0),
        'F1':   f1_score(y, preds, zero_division=0),
        'AUC':  roc_auc_score(y, probs)
    }

results = {
    'Random Forest': get_metrics(rf_model,  X_test_scaled, y_test),
    'XGBoost':       get_metrics(xgb_model, X_test_scaled, y_test),
    'Ensemble':      get_metrics(vote_model, X_test_scaled, y_test)
}

df_results = pd.DataFrame(results).T
print(f"\nMetrics at decision threshold = {DECISION_THRESHOLD}")
print(df_results.round(3))

probs_ensemble = vote_model.predict_proba(X_test_scaled)[:, 1]
preds_050 = (probs_ensemble >= 0.50).astype(int)
preds_020 = (probs_ensemble >= 0.20).astype(int)

print("\n── Before vs After Threshold Change (Ensemble) ──")
comparison = pd.DataFrame({
    'Threshold 0.50 (old)': {
        'Recall':    f"{recall_score(y_test, preds_050):.3f}",
        'Precision': f"{precision_score(y_test, preds_050, zero_division=0):.3f}",
        'F1':        f"{f1_score(y_test, preds_050, zero_division=0):.3f}",
        'Deaths caught': f"{((y_test==1) & (preds_050==1)).sum()} / 111",
        'Deaths missed': f"{((y_test==1) & (preds_050==0)).sum()}",
        'False alarms':  f"{((y_test==0) & (preds_050==1)).sum()}",
    },
    'Threshold 0.20 (new)': {
        'Recall':    f"{recall_score(y_test, preds_020):.3f}",
        'Precision': f"{precision_score(y_test, preds_020, zero_division=0):.3f}",
        'F1':        f"{f1_score(y_test, preds_020, zero_division=0):.3f}",
        'Deaths caught': f"{((y_test==1) & (preds_020==1)).sum()} / 111",
        'Deaths missed': f"{((y_test==1) & (preds_020==0)).sum()}",
        'False alarms':  f"{((y_test==0) & (preds_020==1)).sum()}",
    }
}).T
print(comparison)

# Cell 10: Generate all evaluation charts
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for name, model in [('RF', rf_model), ('XGB', xgb_model), ('Ensemble', vote_model)]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    axes[0,0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.2f})")
axes[0,0].plot([0,1], [0,1], 'k--')
axes[0,0].set_title('ROC Curves')
axes[0,0].legend()

for name, model in [('RF', rf_model), ('XGB', xgb_model), ('Ensemble', vote_model)]:
    p, r, _ = precision_recall_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    axes[0,1].plot(r, p, label=name)
axes[0,1].set_title('Precision-Recall Curves')
axes[0,1].legend()

cm = confusion_matrix(y_test, (vote_model.predict_proba(X_test_scaled)[:, 1] >= DECISION_THRESHOLD).astype(int))
ConfusionMatrixDisplay(cm).plot(ax=axes[1,0], cmap='Blues')
axes[1,0].set_title(f'Confusion Matrix (Ensemble @ threshold={DECISION_THRESHOLD})')

importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
importances.plot(kind='barh', ax=axes[1,1])
axes[1,1].set_title('Top 20 Features (RF)')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'icu_ml_report.png'))
plt.close()
print("Saved icu_ml_report.png")

# Cell 11: SHAP explainability
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, max_display=15, show=False)
plt.title('SHAP Summary Plot (XGBoost)')
plt.savefig(os.path.join(base_dir, 'shap_summary.png'))
plt.close()
print("Saved shap_summary.png")

# Cell 12: Save output files
probs = vote_model.predict_proba(X_test_scaled)[:, 1]

def get_tier(p):
    if p < 0.20:   return 'Low Risk'
    elif p <= 0.50: return 'Medium Risk'
    else:           return 'High Risk'

test_indices = X_test.index
record_ids   = df_master.loc[test_indices, 'RecordID']

icu_predictions = pd.DataFrame({
    'RecordID':          record_ids,
    'True_Label':        y_test,
    'Predicted_Label':   (probs >= DECISION_THRESHOLD).astype(int),
    'Death_Probability': probs,
    'Risk_Tier':         [get_tier(p) for p in probs]
})

icu_predictions.to_csv(os.path.join(base_dir, 'icu_predictions.csv'), index=False)

print("=" * 50)
print("  FINAL RESULTS (threshold = 0.20)")
print("=" * 50)
y_pred_final = (probs >= DECISION_THRESHOLD).astype(int)
print(f"  Accuracy:        {accuracy_score(y_test, y_pred_final):.4f}")
print(f"  Precision:       {precision_score(y_test, y_pred_final, zero_division=0):.4f}")
print(f"  Recall:          {recall_score(y_test, y_pred_final, zero_division=0):.4f}  <- was 0.333 at threshold 0.50")
print(f"  F1-Score:        {f1_score(y_test, y_pred_final, zero_division=0):.4f}")
print(f"  ROC-AUC:         {roc_auc_score(y_test, probs):.4f}  <- unchanged (threshold-independent)")
print(f"  Deaths caught:   {((y_test==1) & (y_pred_final==1)).sum()} / 111")
print(f"  Deaths missed:   {((y_test==1) & (y_pred_final==0)).sum()}")
print(f"  False alarms:    {((y_test==0) & (y_pred_final==1)).sum()}")
print("=" * 50)
print("\nRisk tier distribution:")
print(icu_predictions['Risk_Tier'].value_counts())
print("\nFiles saved: icu_predictions.csv, icu_ml_report.png, shap_summary.png")

# Cell 13: Pipeline Architecture Diagram
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

C = {
    'navy':      '#0D2B4E',
    'blue':      '#1565C0',
    'lightblue': '#1E88E5',
    'teal':      '#00897B',
    'purple':    '#6A1B9A',
    'green':     '#2E7D32',
    'amber':     '#E65100',
    'red':       '#C62828',
    'grey':      '#37474F',
    'text':      '#ECEFF1',
    'subtext':   '#90A4AE',
    'white':     '#FFFFFF',
    'border':    '#2C3E50',
}

def box(ax, x, y, w, h, label, sublabel='', color='#1565C0',
        text_color='#FFFFFF', radius=0.3, fontsize=10, subfontsize=8):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={radius}",
                            linewidth=1.2, edgecolor=color,
                            facecolor=color + '33', zorder=3)
    ax.add_patch(fancy)
    fancy2 = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle=f"round,pad=0.05,rounding_size={radius}",
                             linewidth=1.5, edgecolor=color,
                             facecolor='none', zorder=4)
    ax.add_patch(fancy2)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=5)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha='center', va='center', fontsize=subfontsize,
                color=C['subtext'], zorder=5, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#90A4AE', lw=1.5, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color=C['subtext'],
                ha='left', va='center', zorder=5)

def section_bg(ax, x, y, w, h, color, label=''):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           linewidth=0.8, edgecolor=color+'66',
                           facecolor=color+'11', zorder=1)
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.2, y + h - 0.25, label, fontsize=8,
                color=color, fontweight='bold', zorder=2, alpha=0.9)

ax.text(10, 27.3, 'ICU Mortality Prediction — Pipeline Architecture',
        ha='center', va='center', fontsize=17, color=C['white'],
        fontweight='bold', zorder=5)
ax.text(10, 26.8, 'PhysioNet Set-A  |  4,000 Patients  |  RF + XGBoost Ensemble',
        ha='center', va='center', fontsize=10, color=C['subtext'], zorder=5)
ax.plot([1, 19], [26.5, 26.5], color=C['lightblue'], linewidth=0.8, alpha=0.5)

section_bg(ax, 0.5, 23.5, 19, 2.7, C['teal'], '1 DATA INGESTION')
box(ax, 3.5, 25.2, 4.5, 0.9, 'icumortal.zip',
    '4,000 patient .txt files + Outcomes-a.txt', C['teal'], fontsize=9, subfontsize=7.5)
box(ax, 10, 25.2, 4.5, 0.9, 'Zip Extraction',
    'Cell 2 - Path verification', C['teal'], fontsize=9, subfontsize=7.5)
box(ax, 16.5, 25.2, 3, 0.9, 'File Paths Set',
    'set_a_path / outcomes_path', C['teal'], fontsize=9, subfontsize=7.5)
arrow(ax, 5.75, 25.2, 7.75, 25.2, C['teal'], lw=2)
arrow(ax, 12.25, 25.2, 15, 25.2, C['teal'], lw=2)
box(ax, 4.5, 24.1, 4, 0.8, 'Patient Files (Time, Param, Value)',
    '4,000 x .txt', C['grey'], fontsize=8.5, subfontsize=7.5)
box(ax, 10, 24.1, 4, 0.8, 'Outcomes-a.txt',
    'RecordID | In-hospital_death', C['grey'], fontsize=8.5, subfontsize=7.5)
box(ax, 16, 24.1, 3.5, 0.8, 'Step 1 Complete',
    'Paths verified', C['green'], fontsize=8.5, subfontsize=7.5)

section_bg(ax, 0.5, 20.2, 19, 3.0, C['blue'], '2 FEATURE ENGINEERING  (Cell 3)')
arrow(ax, 4.5, 23.5, 4.5, 23.2, C['blue'], lw=2)
arrow(ax, 10, 23.5, 10, 23.2, C['blue'], lw=2)
box(ax, 4.5, 22.5, 4.5, 1.3,
    'engineer_features()',
    'Filter -1 sentinels / groupby().agg()', C['blue'], fontsize=9, subfontsize=8)
stat_labels = ['_mean', '_max', '_min', '_std', '_last', '_count']
for i, sl in enumerate(stat_labels):
    bx = 1.5 + i * 2.8
    box(ax, bx, 21.15, 2.3, 0.65, sl, '', C['blue'], fontsize=8)
    arrow(ax, 4.5, 21.85, bx, 21.47, C['blue'], lw=1.2)
box(ax, 10, 22.5, 4, 1.3, 'pd.merge()',
    'features_df + outcomes_df on RecordID', C['teal'], fontsize=9, subfontsize=8)
arrow(ax, 6.75, 22.5, 8, 22.5, C['lightblue'], lw=2)
box(ax, 16, 22.5, 3.5, 1.3, 'Feature Matrix',
    '4,000 patients / ~197 features', C['green'], fontsize=9, subfontsize=8)
arrow(ax, 12, 22.5, 14.25, 22.5, C['lightblue'], lw=2)
ax.text(10, 20.55,
        'WARNING: data = data[data["Value"] != -1]  applied',
        ha='center', va='center', fontsize=8, color='#FFB300', style='italic', zorder=5)

section_bg(ax, 0.5, 16.5, 19, 3.5, C['purple'], '3 PREPROCESSING  (Cell 5)')
arrow(ax, 16, 21.85, 16, 20.0, C['purple'], lw=2)
arrow(ax, 16, 20.0, 10, 19.85, C['purple'], lw=2)
box(ax, 4, 19.4, 4, 1.2, 'Drop Sparse Cols',
    'isnull().mean() > 0.70 removed', C['purple'], fontsize=9, subfontsize=8)
box(ax, 10, 19.4, 4, 1.2, 'Stratified Split',
    'train_test_split(stratify=y) / 80:20', C['purple'], fontsize=9, subfontsize=8)
arrow(ax, 6, 19.4, 8, 19.4, C['purple'], lw=2)
box(ax, 4, 17.7, 3.5, 1.1, 'SimpleImputer',
    'strategy="median" / fit on TRAIN only', C['grey'], fontsize=9, subfontsize=8)
box(ax, 10, 17.7, 3.5, 1.1, 'StandardScaler',
    'fit_transform(X_train) / transform(X_test)', C['grey'], fontsize=9, subfontsize=8)
box(ax, 16, 17.7, 3.5, 1.1, 'SMOTE',
    'fit_resample(X_train) / TRAIN ONLY -> 50/50', C['red'], fontsize=9, subfontsize=8)
arrow(ax, 10, 18.8, 4,  18.25, C['purple'], lw=1.5)
arrow(ax, 10, 18.8, 10, 18.25, C['purple'], lw=1.5)
arrow(ax, 10, 18.8, 16, 18.25, C['purple'], lw=1.5)
ax.text(10, 16.75,
        'WARNING: SMOTE applied AFTER split',
        ha='center', va='center', fontsize=8, color='#FF7043', style='italic', zorder=5)

section_bg(ax, 0.5, 11.8, 19, 4.4, C['amber'], '4 ENSEMBLE MODEL TRAINING  (Cells 6 - 8)')
arrow(ax, 4,  17.15, 4,  15.85, C['amber'], lw=2)
arrow(ax, 10, 17.15, 10, 15.85, C['amber'], lw=2)
arrow(ax, 16, 17.15, 16, 15.85, C['amber'], lw=2)
box(ax, 4.5, 15.0, 5, 1.6,
    'Random Forest',
    'n_estimators=200 | max_depth=10\nBagging - 200 parallel trees',
    C['amber'], fontsize=9.5, subfontsize=8)
box(ax, 11.5, 15.0, 5, 1.6,
    'XGBoost',
    'n_estimators=200 | max_depth=6\nlearning_rate=0.05\nBoosting - 200 sequential trees',
    C['red'], fontsize=9.5, subfontsize=8)
ax.text(4.5,  13.68, 'Tree1  Tree2  Tree3 ... Tree200',
        ha='center', fontsize=7.5, color=C['amber'], alpha=0.8, zorder=4)
ax.text(11.5, 13.68, 'Boost1 -> Boost2 -> ... -> Boost200',
        ha='center', fontsize=7.5, color=C['red'], alpha=0.8, zorder=4)
box(ax, 8, 12.4, 8, 1.4,
    'Soft Voting Ensemble  (VotingClassifier)',
    'voting="soft" / Final P(death) = avg( P_RF + P_XGB ) / 2',
    C['green'], fontsize=10, subfontsize=8.5)
arrow(ax, 4.5,  14.2, 5.5,  13.1, C['amber'], lw=2, label='P(death)')
arrow(ax, 11.5, 14.2, 10.5, 13.1, C['red'],   lw=2, label='P(death)')

section_bg(ax, 0.5, 8.0, 19, 3.8, C['lightblue'], '5 THRESHOLD DECISION & RISK STRATIFICATION')
arrow(ax, 8, 11.7, 8, 11.3, C['lightblue'], lw=2.5)
arrow(ax, 8, 11.3, 10, 11.3, C['lightblue'], lw=2.5)
box(ax, 10, 10.6, 5, 1.3,
    'Decision Threshold = 0.20',
    'FIXED from default 0.50 / Recall: 33% -> 85.6%',
    C['red'], fontsize=9.5, subfontsize=8.5)
arrow(ax, 12.5, 10.6, 15.5, 10.6, C['lightblue'], lw=1.5)
arrow(ax, 10, 9.95, 4,  9.55, C['green'],     lw=2)
arrow(ax, 10, 9.95, 10, 9.55, C['amber'],     lw=2)
arrow(ax, 10, 9.95, 16, 9.55, C['red'],       lw=2)
box(ax, 4,  9.0, 4.5, 0.95, '[LOW] Low Risk',
    'prob < 0.20 -> 509 patients',  C['green'], fontsize=9, subfontsize=8)
box(ax, 10, 9.0, 4.5, 0.95, '[MED] Medium Risk',
    '0.20 <= prob <= 0.50 -> 233', C['amber'], fontsize=9, subfontsize=8)
box(ax, 16, 9.0, 3.5, 0.95, '[HIGH] High Risk',
    'prob > 0.50 -> 58 patients',  C['red'],   fontsize=9, subfontsize=8)
box(ax, 16.5, 10.5, 2.5, 0.9, 'Predicted Label',
    '(probs >= 0.20).astype(int)',  C['lightblue'], fontsize=8, subfontsize=7.5)
ax.text(10, 8.2,
        'Test set: 95/111 deaths caught (85.6% recall)  |  16 missed  |  196 false alarms  |  AUC = 0.850',
        ha='center', va='center', fontsize=8.5, color='#80DEEA',
        fontweight='bold', zorder=5)

section_bg(ax, 0.5, 4.6, 19, 3.2, C['teal'], '6 EVALUATION & EXPLAINABILITY')
arrow(ax, 8, 8.0, 8, 7.8, C['teal'], lw=2)
arrow(ax, 8, 7.8, 10, 7.8, C['teal'], lw=2)
metrics = [
    ('Accuracy', '73.5%', C['blue']),
    ('Precision', '32.7%', C['purple']),
    ('Recall',   '85.6%', C['green']),
    ('F1-Score', '47.3%', C['teal']),
    ('ROC-AUC',  '0.850', C['amber']),
]
for i, (lbl, val, col) in enumerate(metrics):
    bx = 2 + i * 3.3
    box(ax, bx, 7.1, 2.9, 1.0, val, lbl, col, fontsize=11, subfontsize=8)
chart_items = [
    ('ROC Curve\nAUC=0.85',    C['blue'],   3.0),
    ('Precision-Recall\nCurve',C['purple'], 7.0),
    ('Confusion\nMatrix',      C['teal'],  11.0),
    ('Feature\nImportances',   C['amber'], 14.5),
    ('SHAP\nSummary Plot',     C['red'],   17.5),
]
for lbl, col, bx in chart_items:
    box(ax, bx, 5.5, 2.7, 0.9, lbl, '', col, fontsize=8)
    arrow(ax, bx, 6.6, bx, 5.95, col, lw=1.2)
ax.text(10, 4.8, 'icu_ml_report.png  |  shap_summary.png  |  icu_predictions.csv',
        ha='center', va='center', fontsize=8.5, color=C['subtext'],
        style='italic', zorder=5)

section_bg(ax, 0.5, 1.3, 19, 3.0, C['green'], '7 OUTPUTS  (Cell 12)')
arrow(ax, 10, 4.6, 10, 4.3, C['green'], lw=2)
box(ax, 4, 3.1, 5, 1.6, 'icu_predictions.csv',
    'RecordID | True_Label\nPredicted_Label | Death_Probability\nRisk_Tier',
    C['green'], fontsize=9, subfontsize=7.5)
box(ax, 10.5, 3.1, 4, 1.6, 'icu_ml_report.png',
    'ROC | PR Curve\nConfusion Matrix\nFeature Importances',
    C['blue'], fontsize=9, subfontsize=7.5)
box(ax, 16, 3.1, 3.5, 1.6, 'shap_summary.png',
    'Top 15 SHAP\nfeature impacts\n(XGBoost)',
    C['purple'], fontsize=9, subfontsize=7.5)
arrow(ax, 10, 4.3, 4,    3.9, C['green'],  lw=1.8)
arrow(ax, 10, 4.3, 10.5, 3.9, C['blue'],   lw=1.8)
arrow(ax, 10, 4.3, 16,   3.9, C['purple'], lw=1.8)

legend_items = [
    (C['teal'],      'Data Ingestion'),
    (C['blue'],      'Feature Engineering'),
    (C['purple'],    'Preprocessing'),
    (C['amber'],     'Model Training'),
    (C['lightblue'], 'Threshold / Risk'),
    (C['green'],     'Outputs'),
]
for i, (col, lbl) in enumerate(legend_items):
    lx = 1.0 + i * 3.2
    ax.add_patch(FancyBboxPatch((lx, 0.5), 0.4, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=col+'44', edgecolor=col, lw=1.2))
    ax.text(lx + 0.55, 0.7, lbl, fontsize=8, color=C['subtext'], va='center')

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(base_dir, 'architecture_diagram.png'), dpi=160, bbox_inches='tight',
            facecolor='#0D1117', edgecolor='none')
plt.close()
print("Architecture diagram saved as architecture_diagram.png")

print("\n✅ All done! Output files saved to:", base_dir)
