# modules/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """데이터 전처리"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """특성 준비 (기존 ML_COMPLETE_DATASET 활용)"""
        
        # 이미 전처리된 데이터이므로 필요한 특성만 선택
        feature_columns = [
            'precipitation', 'humidity', 'avg_temp', 'wind_speed',
            'month', 'is_peak_rainy', 'is_typhoon_season', 
            'precip_ma3', 'precip_ma7', 'rain_days_cumsum',
            'precip_risk_level', 'actual_flood'
        ]
        
        # 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        
        # 기본 특성들이 없으면 생성
        if 'precipitation' not in df.columns:
            print("기본 특성이 없습니다. 원본 데이터를 확인하세요.")
            return None, None, None
        
        # 특성과 타겟 분리
        if 'is_flood_risk' in df.columns:
            target_col = 'is_flood_risk'
        else:
            # 50mm 이상을 위험으로 분류
            df['is_flood_risk'] = (df['precipitation'] >= 50).astype(int)
            target_col = 'is_flood_risk'
        
        # 필수 특성들 생성 (없는 경우)
        if 'month' not in df.columns and 'obs_date' in df.columns:
            df['month'] = pd.to_datetime(df['obs_date']).dt.month
        
        if 'is_peak_rainy' not in df.columns and 'month' in df.columns:
            df['is_peak_rainy'] = (df['month'].isin([6, 7])).astype(int)
        
        # 최종 특성 선택
        final_features = [col for col in available_features if col != 'actual_flood']
        final_features = [col for col in final_features if col in df.columns]
        
        X = df[final_features]
        y = df[target_col]
        
        # 결측값 처리
        X = X.fillna(X.median())
        
        print(f"특성 준비 완료:")
        print(f"   - 특성 수: {len(final_features)}")
        print(f"   - 샘플 수: {len(X)}")
        print(f"   - 양성 비율: {y.mean()*100:.1f}%")
        
        return X, y, final_features