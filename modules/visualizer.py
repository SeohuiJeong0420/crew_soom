import matplotlib.pyplot as plt

def plot_model_comparison():
    # 모델 이름
    models = ["RandomForest", "XGBoost", "LSTM+CNN", "Transformer+SMOTE"]

    # 각 성능 지표
    accuracy  = [0.980, 0.964, 0.587, 0.740]
    recall    = [0.850, 0.876, 0.000, 0.910]   # 침수 Recall
    roc_auc   = [0.970, 0.977, 0.769, 0.874]
    precision = [0.960, 0.816, 0.196, 0.290]   # 침수 Precision
    f1_score  = [0.920, 0.845, 0.320, 0.440]   # 침수 F1-score

    metrics = [accuracy, recall, roc_auc, precision, f1_score]
    labels = ["Accuracy", "Recall (Flood)", "ROC AUC", "Precision (Flood)", "F1-score (Flood)"]

    x = range(len(models))
    bar_width = 0.15

    plt.figure(figsize=(14, 6))

    for i, metric in enumerate(metrics):
        plt.bar([p + i * bar_width for p in x], metric, width=bar_width, label=labels[i])

    plt.xticks([p + 2 * bar_width for p in x], models)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("침수 예측 모델 성능 비교 (5가지 지표)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison_metrics.png")
    plt.show()
