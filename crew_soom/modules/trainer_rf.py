import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

def train_random_forest(csv_path="data/asos_seoul_daily_enriched.csv",
                        model_path="models/randomforest_enriched_model.pkl",
                        fig_path="outputs/randomforest_eval_plots.png"):
    # 1. 데이터 불러오기
    df = pd.read_csv(csv_path, parse_dates=['tm'])

    # 2. 결측치 제거
    df = df.dropna()

    # 3. 특성 / 타겟 분리
    X = df.drop(columns=['tm', 'flood_risk'])
    y = df['flood_risk']

    # 4. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. 모델 학습
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 6. 예측 및 평가
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

    # 7. 시각화 저장
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=plt.gca())
    plt.title("Confusion Matrix")

    plt.subplot(1, 3, 2)
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=plt.gca())
    plt.title("ROC Curve")

    plt.subplot(1, 3, 3)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=plt.gca())
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

    # 8. 모델 저장
    joblib.dump(model, model_path)
    print(f"모델 저장 완료: {model_path}")
