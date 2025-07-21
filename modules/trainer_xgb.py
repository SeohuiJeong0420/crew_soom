# modules/trainer_xgb.py

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score

def train_xgboost(csv_path="data/asos_seoul_daily_enriched.csv", model_output="models/xgb_model_daily.pkl", scaler_output="models/xgb_scaler_daily.pkl"):
    df = pd.read_csv(csv_path)

    features = [
        'avgTa', 'minTa', 'maxTa', 'sumRn', 'avgWs', 'avgRhm', 'avgTs', 'avgTd', 'avgPs',
        'month', 'day', 'weekday', 'is_weekend', 'is_rainy', 'rain_hours', 'max_hourly_rn'
    ]
    target = 'flood_risk'

    X = df[features]
    y = df[target]

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

    # 모델 학습
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

    # 저장
    joblib.dump(model, model_output)
    joblib.dump(scaler, scaler_output)
    print(f"모델 저장 완료: {model_output}")
    print(f"스케일러 저장 완료: {scaler_output}")

    # 시각화 저장
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (XGBoost)")
    plt.tight_layout()
    plt.savefig("outputs/xgb_confusion_matrix_daily.png")
    plt.show()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve (XGBoost)")
    plt.savefig("outputs/xgb_roc_curve_daily.png")
    plt.show()

    # Precision-Recall 시각화
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (XGBoost)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/xgb_precision_recall_daily.png")
    plt.show()

    # Feature Importance 시각화
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='gain', max_num_features=15, height=0.5)
    plt.title('XGBoost Feature Importance (Top 15 by Gain)')
    plt.tight_layout()
    plt.savefig("outputs/xgb_feature_importance_daily.png")
    plt.show()
