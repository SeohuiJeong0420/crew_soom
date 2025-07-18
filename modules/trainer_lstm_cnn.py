import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_cnn(csv_path="data/asos_seoul_daily_enriched.csv",
                   model_path="models/lstm_cnn_model.h5",
                   scaler_path="models/lstm_cnn_scaler.pkl"):
    df = pd.read_csv(csv_path)

    features = ['avgTa', 'minTa', 'maxTa', 'sumRn', 'avgWs', 'avgRhm', 'avgTs', 'avgTd', 'avgPs']
    target = 'flood_risk'

    df = df.dropna(subset=features + [target])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df[target].values

    def create_sequences(X, y, window_size=7):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
    )

    # LSTM + CNN 모델 정의
    model = Sequential()
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

    # 저장
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")

    # 시각화 저장
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_pred_prob):.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM+CNN ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("outputs/lstm_cnn_roc_curve.png")
    plt.show()

    prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("LSTM+CNN Precision-Recall Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig("outputs/lstm_cnn_precision_recall_curve.png")
    plt.show()
