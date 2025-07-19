# modules/evaluator.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

class ModelEvaluator:
    """모델 평가"""
    
    def __init__(self):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
    def evaluate_all_models(self, models, X, y):
        """모든 모델 평가"""
        
        # 데이터 분할 (훈련과 동일하게)
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        results = {}
        
        fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        for i, (name, model) in enumerate(models.items()):
            print(f"\n📊 {name} 평가:")
            
            # 예측
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # 성능 지표
            auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'AUC': auc,
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1': report['1']['f1-score']
            }
            
            print(f"   AUC: {auc:.4f}")
            print(f"   정밀도: {report['1']['precision']:.4f}")
            print(f"   재현율: {report['1']['recall']:.4f}")
            
            # ROC 곡선
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[i].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_title(f'{name} ROC 곡선')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 결과 요약
        results_df = pd.DataFrame(results).T
        print(f"\n🏆 모델 성능 비교:")
        print(results_df.round(4))
        
        return results