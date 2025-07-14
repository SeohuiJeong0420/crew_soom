# modules/trainer.py
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# XGBoost (선택사항)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

class ModelTrainer:
    """모델 훈련"""
    
    def __init__(self):
        self.models = {}
        
    def train_models(self, X, y, feature_names):
        """여러 모델 훈련"""
        
        # 데이터 분할 (시계열 고려)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"📊 데이터 분할:")
        print(f"   훈련: {len(X_train)}개, 테스트: {len(X_test)}개")
        
        # Random Forest
        print("🌳 Random Forest 훈련...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        # 성능 확인
        y_pred = rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"   AUC: {report['macro avg']['f1-score']:.3f}")
        
        # XGBoost (가능한 경우)
        if XGB_AVAILABLE:
            print("🚀 XGBoost 훈련...")
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            xgb_model.fit(X_train, y_train)
            self.models['XGBoost'] = xgb_model
            
            y_pred_xgb = xgb_model.predict(X_test)
            report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
            print(f"   AUC: {report_xgb['macro avg']['f1-score']:.3f}")
        
        # 모델 저장
        self.save_models(feature_names)
        
        return self.models
    
    def save_models(self, feature_names):
        """모델 저장"""
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            filename = f'models/{name.lower()}_model.pkl'
            joblib.dump(model, filename)
            print(f"💾 {name} 저장: {filename}")
        
        # 특성명 저장
        joblib.dump(feature_names, 'models/feature_names.pkl')
        print("💾 특성명 저장 완료")