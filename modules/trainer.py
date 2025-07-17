# modules/advanced_trainer.py - 고급 머신러닝 모델 통합 시스템
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# XGBoost (선택사항)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# TensorFlow/Keras (LSTM + CNN, Transformer용)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class AdvancedModelTrainer:
    """고급 머신러닝 모델 통합 훈련 시스템
    
    지원 모델:
    1. RandomForest (기존)
    2. XGBoost (그래디언트 부스팅)
    3. LSTM + CNN 하이브리드 (시계열 + 공간 특성)
    4. Transformer 기반 (어텐션 메커니즘)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.sequence_length = 14  # 14일 시퀀스
        
        # Focal Loss 구현 (불균형 데이터용)
        if TF_AVAILABLE:
            self.focal_loss = self._create_focal_loss()
    
    def _create_focal_loss(self, alpha=0.25, gamma=2.0):
        """Focal Loss 구현 (불균형 데이터 처리)"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.ones_like(y_true) * alpha
            alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
            cross_entropy = -tf.math.log(p_t)
            weight = alpha_t * tf.pow((1 - p_t), gamma)
            focal_loss = weight * cross_entropy
            return tf.reduce_mean(focal_loss)
        return focal_loss_fixed
    
    def prepare_advanced_features(self, df):
        """고급 특성 엔지니어링"""
        print("고급 특성 엔지니어링 시작...")
        
        # 기본 특성 확인
        df = df.copy()
        df['obs_date'] = pd.to_datetime(df['obs_date'])
        df = df.sort_values('obs_date').reset_index(drop=True)
        
        # 1. 시간적 특성 강화
        df['year'] = df['obs_date'].dt.year
        df['month'] = df['obs_date'].dt.month
        df['day_of_year'] = df['obs_date'].dt.dayofyear
        df['week_of_year'] = df['obs_date'].dt.isocalendar().week
        df['season'] = df['month'].apply(self._get_season)
        df['is_monsoon'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_typhoon_season'] = ((df['month'] >= 7) & (df['month'] <= 9)).astype(int)
        
        # 2. 순환적 시간 특성 (sin/cos 변환)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 3. 이동평균 및 지수이동평균
        for window in [3, 7, 14, 30]:
            df[f'precip_ma_{window}'] = df['precipitation'].rolling(window=window, min_periods=1).mean()
            df[f'temp_ma_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).mean()
            df[f'humidity_ma_{window}'] = df['humidity'].rolling(window=window, min_periods=1).mean()
            
            # 지수이동평균
            df[f'precip_ema_{window}'] = df['precipitation'].ewm(span=window).mean()
        
        # 4. 누적 특성
        df['precip_cumsum_7d'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
        df['precip_cumsum_14d'] = df['precipitation'].rolling(window=14, min_periods=1).sum()
        df['precip_cumsum_30d'] = df['precipitation'].rolling(window=30, min_periods=1).sum()
        
        # 5. 변화율 특성
        df['precip_change_1d'] = df['precipitation'].pct_change(1).fillna(0)
        df['precip_change_3d'] = df['precipitation'].pct_change(3).fillna(0)
        df['temp_change_1d'] = df['avg_temp'].pct_change(1).fillna(0)
        
        # 6. 변동성 특성 (표준편차)
        for window in [7, 14]:
            df[f'precip_std_{window}'] = df['precipitation'].rolling(window=window, min_periods=1).std().fillna(0)
            df[f'temp_std_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).std().fillna(0)
        
        # 7. 극값 특성
        for window in [7, 14]:
            df[f'precip_max_{window}'] = df['precipitation'].rolling(window=window, min_periods=1).max()
            df[f'precip_min_{window}'] = df['precipitation'].rolling(window=window, min_periods=1).min()
            df[f'temp_range_{window}'] = df[f'precip_max_{window}'] - df[f'precip_min_{window}']
        
        # 8. 상호작용 특성
        df['temp_humidity_interaction'] = df['avg_temp'] * df['humidity'] / 100
        df['precip_humidity_interaction'] = df['precipitation'] * df['humidity'] / 100
        df['pressure_temp_interaction'] = df.get('pressure', 1013) * df['avg_temp']
        
        # 9. 위험도 레벨 특성
        df['precip_risk_level'] = pd.cut(
            df['precipitation'], 
            bins=[-np.inf, 0, 10, 30, 50, 100, np.inf], 
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # 10. 연속 날짜 특성
        df['days_since_last_rain'] = self._calculate_days_since_event(df, 'precipitation', threshold=1.0)
        df['days_since_heavy_rain'] = self._calculate_days_since_event(df, 'precipitation', threshold=50.0)
        
        print(f"특성 엔지니어링 완료: {len(df.columns)}개 특성")
        return df
    
    def _get_season(self, month):
        """계절 구분"""
        if month in [3, 4, 5]:
            return 1  # 봄
        elif month in [6, 7, 8]:
            return 2  # 여름 (장마철)
        elif month in [9, 10, 11]:
            return 3  # 가을
        else:
            return 4  # 겨울
    
    def _calculate_days_since_event(self, df, column, threshold):
        """특정 이벤트 이후 경과일 계산"""
        days_since = []
        last_event_idx = -999
        
        for i, value in enumerate(df[column]):
            if value >= threshold:
                last_event_idx = i
                days_since.append(0)
            else:
                if last_event_idx == -999:
                    days_since.append(999)  # 이벤트가 없었음
                else:
                    days_since.append(i - last_event_idx)
        
        return days_since
    
    def prepare_sequence_data(self, df, target_col='is_flood_risk'):
        """시퀀스 데이터 준비 (LSTM/Transformer용)"""
        print(f"시퀀스 데이터 준비 (길이: {self.sequence_length}일)...")
        
        # 수치형 특성만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        # 결측값 처리
        df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 정규화
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric), 
            columns=numeric_cols,
            index=df.index
        )
        
        # 시퀀스 생성
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(df_scaled)):
            # 과거 sequence_length일의 데이터
            X_sequences.append(df_scaled.iloc[i-self.sequence_length:i].values)
            # 현재일의 타겟
            y_sequences.append(df[target_col].iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        self.scalers['sequence_scaler'] = scaler
        self.feature_names_sequence = numeric_cols
        
        print(f"시퀀스 데이터 준비 완료:")
        print(f"   X shape: {X_sequences.shape}")
        print(f"   y shape: {y_sequences.shape}")
        print(f"   특성 수: {len(numeric_cols)}")
        
        return X_sequences, y_sequences
    
    def train_all_models(self, df):
        """모든 고급 모델 훈련"""
        print("고급 머신러닝 모델 통합 훈련 시작!")
        print("=" * 60)
        
        # 고급 특성 엔지니어링
        df_enhanced = self.prepare_advanced_features(df)
        
        # 타겟 생성
        if 'is_flood_risk' not in df_enhanced.columns:
            df_enhanced['is_flood_risk'] = (df_enhanced['precipitation'] >= 50).astype(int)
        
        # 1. 전통적 ML 모델 (RandomForest, XGBoost)
        self._train_traditional_models(df_enhanced)
        
        # 2. 딥러닝 모델 (LSTM+CNN, Transformer)
        if TF_AVAILABLE:
            self._train_deep_models(df_enhanced)
        
        # 3. 모델 성능 비교
        self._compare_model_performance()
        
        # 4. 모델 저장
        self._save_all_models()
        
        return self.models, self.model_performance
    
    def _train_traditional_models(self, df):
        """전통적 ML 모델 훈련"""
        print("\n1️⃣ 전통적 ML 모델 훈련...")
        
        # 특성 선택 (수치형만)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ['is_flood_risk', 'actual_flood']]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['is_flood_risk']
        
        # 데이터 분할 (시계열 고려)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['traditional_scaler'] = scaler
        self.feature_names_traditional = feature_cols
        
        # RandomForest 훈련
        print("RandomForest 훈련 중...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # 성능 평가
        rf_performance = self._evaluate_model(rf_model, X_test_scaled, y_test, 'RandomForest')
        self.models['RandomForest'] = rf_model
        self.model_performance['RandomForest'] = rf_performance
        
        # XGBoost 훈련 (가능한 경우)
        if XGB_AVAILABLE:
            print("XGBoost 훈련 중...")
            
            # 클래스 가중치 계산
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            xgb_performance = self._evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost')
            self.models['XGBoost'] = xgb_model
            self.model_performance['XGBoost'] = xgb_performance
    
    def _train_deep_models(self, df):
        """딥러닝 모델 훈련 (LSTM+CNN, Transformer)"""
        print("\n2️⃣ 딥러닝 모델 훈련...")
        
        # 시퀀스 데이터 준비
        X_seq, y_seq = self.prepare_sequence_data(df)
        
        if len(X_seq) < 100:
            print("시퀀스 데이터가 부족합니다. 딥러닝 모델 스킵.")
            return
        
        # 데이터 분할
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # 클래스 가중치 계산
        class_weight = {
            0: len(y_train) / (2 * (y_train == 0).sum()),
            1: len(y_train) / (2 * (y_train == 1).sum())
        }
        
        # LSTM + CNN 하이브리드 모델 훈련
        print("LSTM + CNN 하이브리드 모델 훈련 중...")
        lstm_cnn_model = self._create_lstm_cnn_model(X_train.shape[1:])
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        # 훈련
        history_lstm = lstm_cnn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        # 성능 평가
        lstm_performance = self._evaluate_deep_model(lstm_cnn_model, X_test, y_test, 'LSTM_CNN')
        self.models['LSTM_CNN'] = lstm_cnn_model
        self.model_performance['LSTM_CNN'] = lstm_performance
        
        # Transformer 모델 훈련
        print("Transformer 모델 훈련 중...")
        transformer_model = self._create_transformer_model(X_train.shape[1:])
        
        history_transformer = transformer_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        transformer_performance = self._evaluate_deep_model(transformer_model, X_test, y_test, 'Transformer')
        self.models['Transformer'] = transformer_model
        self.model_performance['Transformer'] = transformer_performance
    
    def _create_lstm_cnn_model(self, input_shape):
        """LSTM + CNN 하이브리드 모델 생성"""
        model = Sequential([
            # CNN 레이어 (로컬 패턴 추출)
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM 레이어 (시계열 패턴 학습)
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(50, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense 레이어
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.focal_loss,  # Focal Loss 사용
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _create_transformer_model(self, input_shape):
        """Transformer 기반 모델 생성"""
        inputs = Input(shape=input_shape)
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed Forward
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        
        # Add & Norm
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        # Global Average Pooling
        pooling_output = GlobalAveragePooling1D()(ffn_output)
        
        # Classification Head
        outputs = Dense(64, activation='relu')(pooling_output)
        outputs = Dropout(0.3)(outputs)
        outputs = Dense(32, activation='relu')(outputs)
        outputs = Dropout(0.2)(outputs)
        outputs = Dense(1, activation='sigmoid')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # 표준 Binary Crossentropy
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """전통적 ML 모델 성능 평가"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 성능 지표 계산
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        performance = {
            'accuracy': report['accuracy'],
            'precision': report.get('1', {}).get('precision', 0),
            'recall': report.get('1', {}).get('recall', 0),
            'f1_score': report.get('1', {}).get('f1-score', 0),
            'auc': auc,
            'model_type': 'traditional'
        }
        
        print(f"   {model_name}: AUC={auc:.4f}, F1={performance['f1_score']:.4f}")
        return performance
    
    def _evaluate_deep_model(self, model, X_test, y_test, model_name):
        """딥러닝 모델 성능 평가"""
        y_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        
        # 성능 지표 계산
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        performance = {
            'accuracy': report['accuracy'],
            'precision': report.get('1', {}).get('precision', 0),
            'recall': report.get('1', {}).get('recall', 0),
            'f1_score': report.get('1', {}).get('f1-score', 0),
            'auc': auc,
            'model_type': 'deep'
        }
        
        print(f"   {model_name}: AUC={auc:.4f}, F1={performance['f1_score']:.4f}")
        return performance
    
    def _compare_model_performance(self):
        """모델 성능 비교 시각화"""
        print("\n3️⃣ 모델 성능 비교...")
        
        if not self.model_performance:
            return
        
        # 성능 데이터프레임 생성
        perf_df = pd.DataFrame(self.model_performance).T
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 1. 성능 지표 비교 바차트
        plt.subplot(2, 2, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        perf_subset = perf_df[metrics]
        perf_subset.plot(kind='bar', ax=plt.gca(), alpha=0.8)
        plt.title('모델별 성능 지표 비교')
        plt.ylabel('점수')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # 2. AUC 점수 비교
        plt.subplot(2, 2, 2)
        auc_scores = perf_df['auc'].sort_values(ascending=False)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:len(auc_scores)]
        plt.bar(auc_scores.index, auc_scores.values, color=colors)
        plt.title('AUC 점수 순위')
        plt.ylabel('AUC 점수')
        plt.xticks(rotation=45)
        for i, v in enumerate(auc_scores.values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. F1 점수 비교
        plt.subplot(2, 2, 3)
        f1_scores = perf_df['f1_score'].sort_values(ascending=False)
        plt.bar(f1_scores.index, f1_scores.values, color=colors)
        plt.title('F1 점수 순위')
        plt.ylabel('F1 점수')
        plt.xticks(rotation=45)
        for i, v in enumerate(f1_scores.values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. 종합 성능 레이더 차트
        plt.subplot(2, 2, 4)
        if len(perf_df) > 0:
            # 정규화된 성능 지표
            normalized_perf = perf_df[metrics]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 원형 완성
            
            colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(normalized_perf)]
            
            for i, (model_name, scores) in enumerate(normalized_perf.iterrows()):
                values = scores.tolist()
                values += values[:1]  # 원형 완성
                
                plt.plot(angles, values, 'o-', linewidth=2, 
                        label=model_name, color=colors_radar[i % len(colors_radar)])
                plt.fill(angles, values, alpha=0.25, color=colors_radar[i % len(colors_radar)])
            
            plt.xticks(angles[:-1], metrics)
            plt.ylim(0, 1)
            plt.title('종합 성능 레이더 차트')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 최고 성능 모델 출력
        best_auc_model = perf_df['auc'].idxmax()
        best_f1_model = perf_df['f1_score'].idxmax()
        
        print(f"\n 최고 성능 모델:")
        print(f"    AUC 최고: {best_auc_model} ({perf_df.loc[best_auc_model, 'auc']:.4f})")
        print(f"    F1 최고: {best_f1_model} ({perf_df.loc[best_f1_model, 'f1_score']:.4f})")
    
    def _save_all_models(self):
        """모든 모델 저장"""
        print("\n4️⃣ 모델 저장 중...")
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            if name in ['LSTM_CNN', 'Transformer']:
                # TensorFlow 모델 저장
                model.save(f'models/{name.lower()}_model.h5')
            else:
                # Scikit-learn 모델 저장
                joblib.dump(model, f'models/{name.lower()}_model.pkl')
            print(f"    {name} 모델 저장 완료")
        
        # 스케일러 저장
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'models/{name}.pkl')
            print(f"    {name} 저장 완료")
        
        # 특성명 저장
        if hasattr(self, 'feature_names_traditional'):
            joblib.dump(self.feature_names_traditional, 'models/feature_names_traditional.pkl')
        if hasattr(self, 'feature_names_sequence'):
            joblib.dump(self.feature_names_sequence, 'models/feature_names_sequence.pkl')
        
        # 성능 정보 저장
        joblib.dump(self.model_performance, 'models/model_performance.pkl')
        
        print("모든 모델 저장 완료!")
    
    def predict_with_model(self, model_name, input_data):
        """특정 모델로 예측"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 훈련되지 않았습니다.")
        
        model = self.models[model_name]
        
        if model_name in ['LSTM_CNN', 'Transformer']:
            # 딥러닝 모델 예측
            scaler = self.scalers['sequence_scaler']
            
            # 시퀀스 데이터 형태로 변환 (이 부분은 실제 구현에서 조정 필요)
            # 현재는 단순 예시
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled.reshape(1, 1, -1), verbose=0)[0][0]
        else:
            # 전통적 ML 모델 예측
            scaler = self.scalers['traditional_scaler']
            input_scaled = scaler.transform([input_data])
            prediction = model.predict_proba(input_scaled)[0][1]
        
        return prediction
    
    def get_model_summary(self):
        """모델 요약 정보 반환"""
        summary = {
            'total_models': len(self.models),
            'model_list': list(self.models.keys()),
            'performance': self.model_performance,
            'training_time': datetime.now().isoformat(),
            'sequence_length': self.sequence_length,
            'features_traditional': len(getattr(self, 'feature_names_traditional', [])),
            'features_sequence': len(getattr(self, 'feature_names_sequence', []))
        }
        
        if self.model_performance:
            best_auc = max(self.model_performance.values(), key=lambda x: x['auc'])
            best_f1 = max(self.model_performance.values(), key=lambda x: x['f1_score'])
            
            summary['best_auc_model'] = {
                'name': [k for k, v in self.model_performance.items() if v['auc'] == best_auc['auc']][0],
                'score': best_auc['auc']
            }
            summary['best_f1_model'] = {
                'name': [k for k, v in self.model_performance.items() if v['f1_score'] == best_f1['f1_score']][0],
                'score': best_f1['f1_score']
            }
        
        return summary


# 테스트 함수
def test_advanced_trainer():
    """고급 모델 트레이너 테스트"""
    print(" 고급 모델 트레이너 테스트 시작...")
    
    # 샘플 데이터 생성
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'obs_date': dates,
        'precipitation': np.random.exponential(5, len(dates)),
        'avg_temp': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 3, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'wind_speed': np.random.gamma(2, 2, len(dates)),
        'pressure': 1013 + np.random.normal(0, 10, len(dates))
    })
    
    # 인공적인 침수 이벤트 생성
    heavy_rain_days = sample_data['precipitation'] > 50
    sample_data.loc[heavy_rain_days, 'precipitation'] *= 2
    
    print(f"테스트 데이터: {len(sample_data)}일")
    
    # 트레이너 초기화 및 훈련
    trainer = AdvancedModelTrainer()
    models, performance = trainer.train_all_models(sample_data)
    
    # 요약 정보 출력
    summary = trainer.get_model_summary()
    print(f"\n 훈련 완료 요약:")
    print(f"    총 모델 수: {summary['total_models']}")
    print(f"    모델 목록: {', '.join(summary['model_list'])}")
    if 'best_auc_model' in summary:
        print(f"    최고 AUC: {summary['best_auc_model']['name']} ({summary['best_auc_model']['score']:.4f})")
        print(f"    최고 F1: {summary['best_f1_model']['name']} ({summary['best_f1_model']['score']:.4f})")
    
    return trainer


if __name__ == "__main__":
    test_advanced_trainer()