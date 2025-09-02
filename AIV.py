import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# ===================== 1) read data =====================
data = pd.read_excel('raw_data.xlsx')


X = data.drop(['NO.', 'AIV', 'ADRESS'], axis=1)
y = data['AIV'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

# ===================== 2) StandardScaler =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===================== 3) SMOTE =====================
smote = SMOTE(sampling_strategy='minority')
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# ===================== 4) StratifiedKFold =====================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


scoring = {
    'f1': make_scorer(f1_score, average='binary'),
    'precision': 'precision',
    'recall': 'recall'
}

# ===================== 5) Parameter  =====================
lgbm_param_grid = {
    'n_estimators': list(range(1, 200, 1)),
    'num_leaves': list(range(1, 100, 1)),
    'max_depth': list(range(1, 10, 1)),
    'subsample': [0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5,0.6,0.7,0.8,0.9,1.0],
}

# ===================== 6) GridSearchCV =====================
lgbm_grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=1),
    param_grid=lgbm_param_grid,
    scoring=scoring,
    refit='f1',
    cv=skf,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

lgbm_grid_search.fit(X_train_res, y_train_res)

print("Best params:", lgbm_grid_search.best_params_)
print("Best CV F1 (mean over folds):", lgbm_grid_search.best_score_)


cvres = lgbm_grid_search.cv_results_
best_idx = lgbm_grid_search.best_index_
n_splits = skf.get_n_splits()


print("Per-fold Val Precision:", [cvres[f"split{k}_test_precision"][best_idx] for k in range(n_splits)])
print("Per-fold Val Recall   :", [cvres[f"split{k}_test_recall"][best_idx]    for k in range(n_splits)])
print("Per-fold Val F1       :", [cvres[f"split{k}_test_f1"][best_idx]        for k in range(n_splits)])
print("Per-fold Train F1     :", [cvres[f"split{k}_train_f1"][best_idx]       for k in range(n_splits)])

val_f1_folds = [cvres[f"split{k}_test_f1"][best_idx] for k in range(n_splits)]
print(f"Val F1 Mean={np.mean(val_f1_folds):.4f} | Std={np.std(val_f1_folds, ddof=1):.4f}")


best_model = lgbm_grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print(f"\n[GridSearch LGBM] Test Accuracy = {acc:.4f} | Test F1 = {f1:.4f}")
print(classification_report(y_test, y_pred))

# 绘制SHAP
explainer = shap.Explainer(best_model, X_train_res,)
shap_values = explainer(X_train_res)
plt.figure()
shap.summary_plot(shap_values, X_train_res, max_display=28, show=False)
plt.savefig('shap_summary_plot_tnr.tif', dpi=300, format='tiff')
plt.show()
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=21, show=False)
plt.tight_layout()
plt.savefig('all importance.tif', dpi=300, format='tiff', bbox_inches='tight')

plt.show()

