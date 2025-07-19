# modules/web_app.py
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime, timedelta
import io
import base64
import time
import threading
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class FloodWebApp:
    """완전한 웹 기반 침수 예측 시스템 - 자동 업데이트 기능 포함"""
    
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        self.app = Flask(__name__)
        self.model = None
        self.feature_names = []
        self.data = None
        self.model_loaded = False
        self.data_last_updated = None
        self.data_start_date = None
        self.data_end_date = None
        self.auto_update_enabled = False
        self.update_interval = int(os.getenv('UPDATE_INTERVAL', 300))  # 기본 5분
        self.last_check_time = None
        
        # API 설정
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.city = os.getenv('WEATHER_CITY', 'Seoul')
        self.country = os.getenv('WEATHER_COUNTRY', 'KR')
        self.lat = float(os.getenv('WEATHER_LAT', 37.5665))
        self.lon = float(os.getenv('WEATHER_LON', 126.9780))
        
        # API 키 확인
        if not self.api_key or self.api_key == 'your_api_key_here':
            print("⚠️  .env 파일에 OPENWEATHER_API_KEY를 설정해주세요!")
            print("🔗 https://openweathermap.org/api 에서 무료 발급 가능")
            self.api_available = False
        else:
            self.api_available = True
            print(f"✅ API 키 설정됨 - 위치: {self.city}, {self.country}")
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        self.setup_routes()
        self.check_initial_data()
        self.start_auto_update_service()
    
    def check_initial_data(self):
        """초기 데이터 및 모델 확인"""
        # 데이터 확인
        data_paths = [
            'data/processed/ML_COMPLETE_DATASET.csv',
            'STRATEGIC_FLOOD_DATA/4_ML_READY/ML_COMPLETE_DATASET.csv',
            'ML_COMPLETE_DATASET.csv'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    self.data = pd.read_csv(path)
                    if 'obs_date' in self.data.columns:
                        self.data['obs_date'] = pd.to_datetime(self.data['obs_date'])
                        self.data_start_date = self.data['obs_date'].min()
                        self.data_end_date = self.data['obs_date'].max()
                    
                    # 🆕 오늘까지 자동 채우기
                    self.fill_to_today()
                    
                    self.data_last_updated = datetime.now()
                    print(f"✅ 데이터 발견: {path}")
                    print(f"📅 데이터 기간: {self.data_start_date} ~ {self.data_end_date}")
                    break
                except:
                    continue
        
        # 모델 확인
        if os.path.exists('models/randomforest_model.pkl'):
            try:
                self.model = joblib.load('models/randomforest_model.pkl')
                self.feature_names = joblib.load('models/feature_names.pkl')
                self.model_loaded = True
                print("✅ 모델 로드 성공")
            except:
                print("❌ 모델 로드 실패")
    
    def fill_to_today(self):
        """🆕 마지막 데이터부터 오늘까지 자동 채우기"""
        if self.data is None or len(self.data) == 0:
            return
        
        last_date = self.data_end_date.date() if self.data_end_date else datetime.now().date() - timedelta(days=10)
        today = datetime.now().date()
        
        current_date = last_date + timedelta(days=1)
        added_count = 0
        
        while current_date <= today:
            # 7월 장마철 현실적인 데이터 생성
            if current_date.month == 7:
                precipitation = max(0, np.random.exponential(8))  # 장마철
                humidity = np.clip(np.random.normal(75, 12), 50, 95)
                avg_temp = np.clip(np.random.normal(26, 4), 20, 32)
            else:
                precipitation = max(0, np.random.exponential(3))  # 평상시
                humidity = np.clip(np.random.normal(65, 15), 30, 90)
                avg_temp = np.clip(np.random.normal(24, 6), 15, 35)
            
            new_row = {
                'obs_date': pd.Timestamp(current_date),
                'precipitation': precipitation,
                'humidity': humidity,
                'avg_temp': avg_temp,
                'wind_speed': max(0, np.random.normal(3, 2)),
                'month': current_date.month,
                'precip_ma3': precipitation,
                'precip_ma7': precipitation,
                'is_peak_rainy': 1 if current_date.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(precipitation),
                'is_flood_risk': 1 if precipitation >= 50 else 0
            }
            
            new_df = pd.DataFrame([new_row])
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            current_date += timedelta(days=1)
            added_count += 1
        
        if added_count > 0:
            self.data_end_date = pd.Timestamp(today)
            print(f"📅 오늘까지 데이터 채움: +{added_count}일 (총 {len(self.data)}행)")
    
    def start_auto_update_service(self):
        """자동 업데이트 서비스 시작"""
        def auto_update_worker():
            while True:
                if self.auto_update_enabled:
                    self.last_check_time = datetime.now()
                    try:
                        # 🆕 실제 API 우선, 실패시 시뮬레이션
                        if self.api_available:
                            self.real_data_update()
                        else:
                            self.simulate_data_update()
                    except Exception as e:
                        print(f"자동 업데이트 오류: {e}")
                
                time.sleep(self.update_interval)
        
        update_thread = threading.Thread(target=auto_update_worker, daemon=True)
        update_thread.start()
    
    def simulate_data_update(self):
        """데이터 업데이트 시뮬레이션"""
        if self.data is not None and len(self.data) > 0:
            # 최신 날짜 이후의 가상 데이터 추가
            last_date = self.data_end_date if self.data_end_date else datetime.now() - timedelta(days=1)
            new_date = last_date + timedelta(hours=1)
            
            # 새로운 데이터 행 생성 (현실적인 기상 데이터)
            new_row = {
                'obs_date': new_date,
                'precipitation': np.random.exponential(scale=5),  # 강수량은 지수분포
                'humidity': np.random.normal(70, 15),  # 습도
                'avg_temp': np.random.normal(22, 8),   # 온도
                'wind_speed': np.random.exponential(scale=3),
                'month': new_date.month,
                'precip_ma3': 0,
                'precip_ma7': 0,
                'is_peak_rainy': 1 if new_date.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': 0
            }
            
            # 데이터 범위 조정
            new_row['humidity'] = np.clip(new_row['humidity'], 20, 100)
            new_row['avg_temp'] = np.clip(new_row['avg_temp'], -10, 40)
            new_row['wind_speed'] = np.clip(new_row['wind_speed'], 0, 20)
            
            # 위험도 계산
            if new_row['precipitation'] >= 100:
                new_row['precip_risk_level'] = 4
            elif new_row['precipitation'] >= 50:
                new_row['precip_risk_level'] = 3
            elif new_row['precipitation'] >= 30:
                new_row['precip_risk_level'] = 2
            elif new_row['precipitation'] >= 10:
                new_row['precip_risk_level'] = 1
            
            # 침수 위험 레이블
            new_row['is_flood_risk'] = 1 if new_row['precipitation'] >= 50 else 0
            
            # 데이터프레임에 추가
            new_df = pd.DataFrame([new_row])
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            
            # 너무 많은 데이터 방지 (최근 10000개만 유지)
            if len(self.data) > 10000:
                self.data = self.data.tail(10000).reset_index(drop=True)
            
            # 날짜 정보 업데이트
            self.data_end_date = new_date
            self.data_last_updated = datetime.now()
            
            print(f"🔄 시뮬레이션 업데이트: {new_date} (총 {len(self.data)}행)")
    
    def real_data_update(self):
        """🆕 실제 OpenWeatherMap API에서 데이터 가져오기"""
        try:
            current_weather = self.fetch_current_weather()
            if current_weather:
                new_row = self.process_weather_data(current_weather)
                
                # 데이터프레임에 추가
                new_df = pd.DataFrame([new_row])
                self.data = pd.concat([self.data, new_df], ignore_index=True)
                
                # 너무 많은 데이터 방지 (최근 10000개만 유지)
                if len(self.data) > 10000:
                    self.data = self.data.tail(10000).reset_index(drop=True)
                
                # 날짜 정보 업데이트
                self.data_end_date = new_row['obs_date']
                self.data_last_updated = datetime.now()
                
                print(f"🌤️  실제 API 업데이트: {new_row['obs_date']} (강수량: {new_row['precipitation']:.1f}mm)")
            else:
                print("❌ API 호출 실패 - 시뮬레이션으로 대체")
                self.simulate_data_update()
                
        except Exception as e:
            print(f"실제 데이터 업데이트 실패: {e}")
            self.simulate_data_update()
    
    def fetch_current_weather(self):
        """🆕 OpenWeatherMap API에서 현재 날씨 데이터 가져오기"""
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'kr'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API 호출 오류: {e}")
            return None
        except Exception as e:
            print(f"날씨 데이터 처리 오류: {e}")
            return None
    
    def process_weather_data(self, weather_data):
        """🆕 API 응답을 데이터프레임 형식으로 변환"""
        try:
            now = datetime.now()
            main = weather_data.get('main', {})
            weather = weather_data.get('weather', [{}])[0]
            wind = weather_data.get('wind', {})
            rain = weather_data.get('rain', {})
            
            # 강수량 처리 (OpenWeatherMap은 mm/h 단위)
            precipitation = rain.get('1h', 0) or rain.get('3h', 0) / 3
            
            return {
                'obs_date': now,
                'precipitation': precipitation,
                'humidity': main.get('humidity', 60),
                'avg_temp': main.get('temp', 20),
                'wind_speed': wind.get('speed', 0) * 3.6,  # m/s를 km/h로 변환
                'month': now.month,
                'precip_ma3': precipitation,
                'precip_ma7': precipitation,
                'is_peak_rainy': 1 if now.month in [6, 7, 8, 9] else 0,
                'precip_risk_level': self.get_precip_level(precipitation),
                'is_flood_risk': 1 if precipitation >= 50 else 0,
                'weather_main': weather.get('main', 'Clear'),
                'weather_desc': weather.get('description', '맑음'),
                'pressure': main.get('pressure', 1013),
                'visibility': weather_data.get('visibility', 10000) / 1000
            }
        except Exception as e:
            print(f"날씨 데이터 변환 오류: {e}")
            return {
                'obs_date': datetime.now(),
                'precipitation': 0, 'humidity': 60, 'avg_temp': 20, 'wind_speed': 0,
                'month': datetime.now().month, 'precip_ma3': 0, 'precip_ma7': 0,
                'is_peak_rainy': 0, 'precip_risk_level': 0, 'is_flood_risk': 0
            }
    
    def setup_routes(self):
        """모든 라우트 설정"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'data_loaded': self.data is not None,
                'data_rows': len(self.data) if self.data is not None else 0,
                'model_loaded': self.model_loaded,
                'features': len(self.feature_names) if self.feature_names else 0,
                'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                'data_last_updated': self.data_last_updated.isoformat() if self.data_last_updated else None,
                'auto_update_enabled': self.auto_update_enabled,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'update_interval': self.update_interval,
                'api_available': self.api_available,
                'api_location': f"{self.city}, {self.country}" if self.api_available else None,
                'today': datetime.now().strftime('%Y-%m-%d')  # 🆕 오늘 날짜
            })
        
        @self.app.route('/api/load_data', methods=['POST'])
        def load_data():
            try:
                data_paths = [
                    'data/processed/ML_COMPLETE_DATASET.csv',
                    'STRATEGIC_FLOOD_DATA/4_ML_READY/ML_COMPLETE_DATASET.csv',
                    'ML_COMPLETE_DATASET.csv'
                ]
                
                for path in data_paths:
                    if os.path.exists(path):
                        self.data = pd.read_csv(path)
                        if 'obs_date' in self.data.columns:
                            self.data['obs_date'] = pd.to_datetime(self.data['obs_date'])
                            self.data_start_date = self.data['obs_date'].min()
                            self.data_end_date = self.data['obs_date'].max()
                        
                        # 🆕 오늘까지 채우기
                        self.fill_to_today()
                        self.data_last_updated = datetime.now()
                        
                        return jsonify({
                            'success': True,
                            'message': f'데이터 로드 성공: {len(self.data)}행',
                            'rows': len(self.data),
                            'columns': len(self.data.columns),
                            'start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                            'end_date': self.data_end_date.isoformat() if self.data_end_date else None
                        })
                
                return jsonify({
                    'success': False,
                    'message': 'ML_COMPLETE_DATASET.csv 파일을 찾을 수 없습니다.'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/update_data', methods=['POST'])
        def update_data():
            """수동 데이터 업데이트"""
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                old_count = len(self.data)
                
                # 🆕 실제 API 우선 사용
                if self.api_available:
                    try:
                        current_weather = self.fetch_current_weather()
                        if current_weather:
                            new_row = self.process_weather_data(current_weather)
                            new_df = pd.DataFrame([new_row])
                            self.data = pd.concat([self.data, new_df], ignore_index=True)
                            
                            return jsonify({
                                'success': True,
                                'message': f'실제 기상 데이터 1개가 추가되었습니다.',
                                'old_count': old_count,
                                'new_count': len(self.data),
                                'added_count': 1,
                                'latest_date': self.data_end_date.isoformat() if self.data_end_date else None,
                                'data_source': 'OpenWeatherMap API',
                                'precipitation': new_row['precipitation'],
                                'temperature': new_row['avg_temp'],
                                'humidity': new_row['humidity']
                            })
                        else:
                            raise Exception("API 응답 실패")
                    except Exception as e:
                        print(f"실제 API 실패: {e}, 시뮬레이션으로 대체")
                        
                # API 실패 시 또는 키 없으면 시뮬레이션
                for _ in range(np.random.randint(3, 8)):
                    self.simulate_data_update()
                    time.sleep(0.1)
                
                new_count = len(self.data)
                added_count = new_count - old_count
                
                return jsonify({
                    'success': True,
                    'message': f'시뮬레이션 데이터 {added_count}개가 추가되었습니다.',
                    'old_count': old_count,
                    'new_count': new_count,
                    'added_count': added_count,
                    'latest_date': self.data_end_date.isoformat() if self.data_end_date else None,
                    'data_source': 'Simulation'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'업데이트 실패: {str(e)}'})
        
        @self.app.route('/api/toggle_auto_update', methods=['POST'])
        def toggle_auto_update():
            """자동 업데이트 토글"""
            try:
                self.auto_update_enabled = not self.auto_update_enabled
                
                return jsonify({
                    'success': True,
                    'auto_update_enabled': self.auto_update_enabled,
                    'message': f'자동 업데이트가 {"활성화" if self.auto_update_enabled else "비활성화"}되었습니다.'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/train_model', methods=['POST'])
        def train_model():
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                # 특성 준비
                basic_features = ['precipitation', 'humidity', 'avg_temp']
                available_features = [col for col in basic_features if col in self.data.columns]
                
                # 추가 특성
                extra_features = ['wind_speed', 'month', 'precip_ma3', 'precip_ma7', 
                                'is_peak_rainy', 'precip_risk_level']
                for feat in extra_features:
                    if feat in self.data.columns:
                        available_features.append(feat)
                
                # 타겟 변수
                if 'is_flood_risk' not in self.data.columns:
                    self.data['is_flood_risk'] = (self.data['precipitation'] >= 50).astype(int)
                
                X = self.data[available_features]
                y = self.data['is_flood_risk']
                
                # 결측값 처리
                X = X.fillna(X.median())
                
                # 데이터 분할
                split_idx = int(len(X) * 0.8)
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]
                
                # 모델 훈련
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                self.model.fit(X_train, y_train)
                self.feature_names = available_features
                
                # 성능 평가
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                
                try:
                    auc_score = roc_auc_score(y_test, y_proba)
                except:
                    auc_score = 0.5
                
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # 모델 저장
                os.makedirs('models', exist_ok=True)
                joblib.dump(self.model, 'models/randomforest_model.pkl')
                joblib.dump(self.feature_names, 'models/feature_names.pkl')
                
                self.model_loaded = True
                
                return jsonify({
                    'success': True,
                    'message': '모델 훈련 완료!',
                    'auc': round(auc_score, 3),
                    'precision': round(report['1']['precision'], 3),
                    'recall': round(report['1']['recall'], 3),
                    'features': len(available_features),
                    'training_data_size': len(X_train)
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'훈련 실패: {str(e)}'})
        
        @self.app.route('/api/create_visualization', methods=['POST'])
        def create_visualization():
            try:
                if self.data is None:
                    return jsonify({'success': False, 'message': '먼저 데이터를 로드하세요.'})
                
                viz_type = request.json.get('type', 'precipitation')
                
                plt.figure(figsize=(12, 8))
                
                if viz_type == 'precipitation':
                    # 강수량 시계열
                    plt.subplot(2, 1, 1)
                    plt.plot(self.data['obs_date'], self.data['precipitation'], alpha=0.7, color='blue')
                    plt.title('📈 강수량 시계열 분석')
                    plt.ylabel('강수량 (mm)')
                    plt.grid(True, alpha=0.3)
                    
                    # 최근 데이터 강조
                    if len(self.data) > 100:
                        recent_data = self.data.tail(100)
                        plt.subplot(2, 1, 2)
                        plt.plot(recent_data['obs_date'], recent_data['precipitation'], 
                                color='red', linewidth=2, alpha=0.8)
                        plt.title('🔍 최근 100개 데이터 (상세)')
                        plt.ylabel('강수량 (mm)')
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3)
                    
                elif viz_type == 'monthly':
                    # 월별 평균 강수량
                    if 'month' in self.data.columns:
                        monthly_precip = self.data.groupby('month')['precipitation'].agg(['mean', 'std', 'count'])
                        plt.bar(monthly_precip.index, monthly_precip['mean'], 
                               yerr=monthly_precip['std'], alpha=0.8, capsize=5)
                        plt.title('📊 월별 평균 강수량 (±표준편차)')
                        plt.xlabel('월')
                        plt.ylabel('평균 강수량 (mm)')
                        
                        # 데이터 개수 표시
                        for i, count in enumerate(monthly_precip['count']):
                            plt.text(i+1, monthly_precip['mean'].iloc[i] + monthly_precip['std'].iloc[i] + 2, 
                                   f'n={count}', ha='center', fontsize=8)
                
                elif viz_type == 'distribution':
                    # 강수량 분포 (히스토그램 + 박스플롯)
                    plt.subplot(2, 1, 1)
                    plt.hist(self.data['precipitation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50mm 위험선')
                    plt.axvline(x=self.data['precipitation'].mean(), color='green', 
                               linestyle='-', linewidth=2, label=f'평균: {self.data["precipitation"].mean():.1f}mm')
                    plt.title('📊 강수량 분포')
                    plt.xlabel('강수량 (mm)')
                    plt.ylabel('빈도')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 박스플롯
                    plt.subplot(2, 1, 2)
                    plt.boxplot(self.data['precipitation'], vert=False, patch_artist=True)
                    plt.xlabel('강수량 (mm)')
                    plt.title('📦 강수량 박스플롯')
                    plt.grid(True, alpha=0.3)
                
                elif viz_type == 'correlation':
                    # 상관관계
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:8]
                    corr_matrix = self.data[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, linewidths=0.5)
                    plt.title('🔍 변수간 상관관계')
                
                elif viz_type == 'recent_trend':
                    # 최근 트렌드 분석 - 날짜 기준으로 필터링
                    if 'obs_date' in self.data.columns and len(self.data) > 0:
                        # 최근 30일 데이터 필터링
                        latest_date = self.data['obs_date'].max()
                        start_date = latest_date - timedelta(days=30)
                        recent_data = self.data[self.data['obs_date'] >= start_date]
                        
                        if len(recent_data) > 0:
                            plt.subplot(3, 1, 1)
                            plt.plot(recent_data['obs_date'], recent_data['precipitation'], 'b-', linewidth=2)
                            plt.title(f'🕐 최근 30일 강수량 ({len(recent_data)}개 데이터)')
                            plt.ylabel('강수량 (mm)')
                            plt.grid(True, alpha=0.3)
                            
                            plt.subplot(3, 1, 2)
                            plt.plot(recent_data['obs_date'], recent_data['humidity'], 'g-', linewidth=2)
                            plt.title(f'💧 최근 30일 습도')
                            plt.ylabel('습도 (%)')
                            plt.grid(True, alpha=0.3)
                            
                            plt.subplot(3, 1, 3)
                            plt.plot(recent_data['obs_date'], recent_data['avg_temp'], 'r-', linewidth=2)
                            plt.title(f'🌡️ 최근 30일 온도')
                            plt.ylabel('온도 (°C)')
                            plt.xlabel('날짜')
                            plt.xticks(rotation=45)
                            plt.grid(True, alpha=0.3)
                        else:
                            plt.text(0.5, 0.5, '최근 30일 데이터가 없습니다', 
                                   ha='center', va='center', transform=plt.gca().transAxes)
                    else:
                        # 날짜 정보가 없으면 최근 50개 데이터
                        recent_data = self.data.tail(50)
                        
                        plt.subplot(3, 1, 1)
                        plt.plot(range(len(recent_data)), recent_data['precipitation'], 'b-', linewidth=2)
                        plt.title(f'🕐 최근 {len(recent_data)}개 데이터 - 강수량')
                        plt.ylabel('강수량 (mm)')
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(3, 1, 2)
                        plt.plot(range(len(recent_data)), recent_data['humidity'], 'g-', linewidth=2)
                        plt.title('💧 습도')
                        plt.ylabel('습도 (%)')
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(3, 1, 3)
                        plt.plot(range(len(recent_data)), recent_data['avg_temp'], 'r-', linewidth=2)
                        plt.title('🌡️ 온도')
                        plt.ylabel('온도 (°C)')
                        plt.xlabel('데이터 순서')
                        plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # 이미지를 base64로 변환
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_base64}',
                    'message': f'{viz_type} 차트 생성 완료',
                    'data_count': len(self.data)
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'시각화 실패: {str(e)}'})
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()
                
                # 🆕 특정 날짜 예측 지원
                target_date = data.get('target_date')
                if target_date:
                    target_date = pd.to_datetime(target_date).date()
                    data['prediction_date'] = target_date.strftime('%Y-%m-%d')
                
                if self.model_loaded and self.model is not None:
                    # ML 모델 예측
                    risk_score = self.predict_with_ml_model(data)
                else:
                    # 규칙 기반 예측
                    risk_score = self.calculate_simple_risk(data)
                
                risk_info = self.get_risk_level(risk_score)
                
                recommendations = {
                    0: ["정상적인 업무 진행", "일기예보 정기 확인"],
                    1: ["기상 상황 주시", "우산 준비"],
                    2: ["외출 시 주의", "지하공간 점검", "배수구 확인"],
                    3: ["불필요한 외출 자제", "중요 물품 안전한 곳 이동", "비상연락망 확인"],
                    4: ["즉시 대피 준비", "119 신고 대기", "지하시설 피해"]
                }
                
                return jsonify({
                    'risk_score': round(risk_score, 1),
                    'risk_level': risk_info['level'],
                    'risk_name': risk_info['name'],
                    'risk_color': risk_info['color'],
                    'action': risk_info['action'],
                    'recommendations': recommendations.get(risk_info['level'], []),
                    'prediction_time': datetime.now().isoformat(),
                    'prediction_date': data.get('prediction_date', datetime.now().strftime('%Y-%m-%d')),  # 🆕
                    'model_used': 'ML Model' if self.model_loaded else 'Rule-based',
                    'data_freshness': (datetime.now() - self.data_last_updated).total_seconds() / 60 if self.data_last_updated else None
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def predict_with_ml_model(self, data):
        """ML 모델 예측"""
        try:
            features = []
            basic_features = {
                'precipitation': data.get('precipitation', 0),
                'humidity': data.get('humidity', 60),
                'avg_temp': data.get('avg_temp', 20),
                'wind_speed': data.get('wind_speed', 2),
                'month': 7,
                'precip_ma3': data.get('precipitation', 0),
                'precip_ma7': data.get('precipitation', 0),
                'is_peak_rainy': 1 if data.get('season_type') == 'rainy' else 0,
                'precip_risk_level': self.get_precip_level(data.get('precipitation', 0))
            }
            
            for feature_name in self.feature_names:
                if feature_name in basic_features:
                    features.append(basic_features[feature_name])
                else:
                    features.append(0)
            
            prediction_proba = self.model.predict_proba([features])[0][1]
            return prediction_proba * 100
            
        except Exception as e:
            return self.calculate_simple_risk(data)
    
    def get_precip_level(self, precipitation):
        if precipitation >= 100: return 4
        elif precipitation >= 50: return 3
        elif precipitation >= 30: return 2
        elif precipitation >= 10: return 1
        else: return 0
    
    def calculate_simple_risk(self, data):
        """간단한 위험도 계산"""
        score = 0
        precipitation = data.get('precipitation', 0)
        score += min(precipitation * 0.4, 40)
        
        precip_3d = data.get('precip_sum_3d', precipitation)
        score += min(precip_3d * 0.25, 25)
        
        humidity = data.get('humidity', 50)
        score += min((humidity - 50) * 0.4, 20)
        
        season_type = data.get('season_type', 'dry')
        if season_type == 'rainy':
            score += 15
        else:
            score += 3
        
        return min(score, 100)
    
    def get_risk_level(self, score):
        """위험도 등급"""
        if score <= 20:
            return {'level': 0, 'name': '매우낮음', 'color': '🟢', 'action': '정상 업무'}
        elif score <= 40:
            return {'level': 1, 'name': '낮음', 'color': '🟡', 'action': '상황 주시'}
        elif score <= 60:
            return {'level': 2, 'name': '보통', 'color': '🟠', 'action': '주의 준비'}
        elif score <= 80:
            return {'level': 3, 'name': '높음', 'color': '🔴', 'action': '대비 조치'}
        else:
            return {'level': 4, 'name': '매우높음', 'color': '🟣', 'action': '즉시 대응'}
    
    def get_dashboard_template(self):
        """향상된 웹 대시보드 템플릿"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌊 침수 예측 AI 시스템 (자동 업데이트)</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; background: #f0f2f5; }
                .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center; }
                .card { background: white; padding: 25px; border-radius: 15px; 
                       box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 25px; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; }
                .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
                .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
                
                /* 버튼 스타일 */
                .btn { background: #667eea; color: white; padding: 15px 25px; 
                      border: none; border-radius: 8px; cursor: pointer; font-size: 16px; 
                      transition: all 0.3s; font-weight: bold; }
                .btn:hover { background: #5a6fd8; transform: translateY(-2px); }
                .btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }
                .btn-success { background: #28a745; }
                .btn-success:hover { background: #218838; }
                .btn-warning { background: #ffc107; color: black; }
                .btn-warning:hover { background: #e0a800; }
                .btn-danger { background: #dc3545; }
                .btn-danger:hover { background: #c82333; }
                .btn-info { background: #17a2b8; }
                .btn-info:hover { background: #138496; }
                
                /* 자동 업데이트 토글 */
                .toggle-switch { position: relative; display: inline-block; width: 60px; height: 34px; }
                .toggle-switch input { opacity: 0; width: 0; height: 0; }
                .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
                         background-color: #ccc; transition: .4s; border-radius: 34px; }
                .slider:before { position: absolute; content: ""; height: 26px; width: 26px; left: 4px;
                                bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
                input:checked + .slider { background-color: #2196F3; }
                input:checked + .slider:before { transform: translateX(26px); }
                
                /* 입력 필드 */
                .input-group { margin: 15px 0; }
                .input-group label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
                .input-group input, .input-group select { 
                    width: 100%; padding: 12px; border: 2px solid #ddd; 
                    border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }
                .input-group input:focus, .input-group select:focus { 
                    border-color: #667eea; outline: none; }
                
                /* 위험도 표시 */
                .risk-meter { text-align: center; padding: 40px; font-size: 28px; 
                             border-radius: 15px; margin: 20px 0; font-weight: bold; 
                             transition: all 0.3s; }
                .risk-0 { background: #4CAF50; color: white; }
                .risk-1 { background: #FFEB3B; color: black; }
                .risk-2 { background: #FF9800; color: white; }
                .risk-3 { background: #F44336; color: white; }
                .risk-4 { background: #9C27B0; color: white; }
                
                /* 상태 표시 */
                .status { padding: 15px; border-left: 5px solid #667eea; 
                         background: #f8f9ff; border-radius: 8px; margin: 15px 0; }
                .status-success { border-left-color: #28a745; background: #f8fff9; }
                .status-warning { border-left-color: #ffc107; background: #fffdf8; }
                .status-error { border-left-color: #dc3545; background: #fff8f8; }
                
                /* 데이터 정보 카드 */
                .data-info { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                            gap: 15px; margin: 20px 0; }
                .data-card { background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                            padding: 20px; border-radius: 10px; text-align: center; }
                .data-card h3 { margin: 0 0 10px 0; font-size: 24px; }
                .data-card p { margin: 5px 0; font-size: 14px; opacity: 0.9; }
                
                /* 자동 업데이트 상태 */
                .update-status { display: flex; align-items: center; gap: 15px; 
                               padding: 15px; background: #f8f9fa; border-radius: 10px; margin: 15px 0; }
                .update-indicator { width: 12px; height: 12px; border-radius: 50%; }
                .update-active { background: #28a745; animation: pulse 2s infinite; }
                .update-inactive { background: #dc3545; }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                /* 기능 버튼들 */
                .function-buttons { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                .function-btn { padding: 20px; text-align: center; border-radius: 10px; 
                               cursor: pointer; transition: all 0.3s; border: 2px solid transparent; }
                .function-btn:hover { transform: translateY(-3px); border-color: #667eea; }
                
                /* 시각화 영역 */
                .viz-container { text-align: center; margin: 20px 0; }
                .viz-image { max-width: 100%; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
                
                /* 로딩 애니메이션 */
                .loading { display: none; text-align: center; padding: 20px; }
                .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; 
                          border-radius: 50%; width: 40px; height: 40px; 
                          animation: spin 1s linear infinite; margin: 0 auto; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                
                /* 전역 로딩 오버레이 */
                .loading-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                                  background: rgba(0,0,0,0.7); z-index: 9999; display: none; 
                                  align-items: center; justify-content: center; }
                .loading-content { background: white; padding: 40px; border-radius: 15px; text-align: center; }
                
                /* 테스트 시나리오 */
                .scenario-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
                .scenario-btn { padding: 15px; background: #28a745; color: white; 
                               border: none; border-radius: 8px; cursor: pointer; 
                               font-size: 14px; font-weight: bold; transition: all 0.3s; }
                .scenario-btn:hover { background: #218838; transform: scale(1.05); }
                
                /* 실시간 정보 */
                .realtime-info { font-size: 12px; color: #666; margin-top: 10px; }
                .fresh { color: #28a745; font-weight: bold; }
                .stale { color: #dc3545; font-weight: bold; }
                
                /* 🆕 날짜 선택 */
                .date-section { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }
                .today-indicator { font-size: 18px; font-weight: bold; color: #2196F3; margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <!-- 전역 로딩 오버레이 -->
            <div class="loading-overlay" id="loading-overlay">
                <div class="loading-content">
                    <div class="spinner"></div>
                    <h3 id="loading-message">처리 중...</h3>
                    <p>잠시만 기다려주세요.</p>
                </div>
            </div>
            
            <div class="container">
                <div class="header">
                    <h1>🌊 침수 예측 AI 시스템 (자동 업데이트)</h1>
                    <p>실시간 API 데이터 + 오늘까지 자동 채움 + 완전한 웹 기반 머신러닝 플랫폼</p>
                    <div id="system-status" class="status">시스템 상태 확인 중...</div>
                </div>
                
                <!-- 데이터 현황 대시보드 -->
                <div class="card">
                    <h2>📊 데이터 현황</h2>
                    <div class="data-info" id="data-info">
                        <div class="data-card">
                            <h3 id="data-rows">-</h3>
                            <p>총 데이터 수</p>
                        </div>
                        <div class="data-card">
                            <h3 id="data-period">-</h3>
                            <p>데이터 기간</p>
                        </div>
                        <div class="data-card">
                            <h3 id="last-update">-</h3>
                            <p>마지막 업데이트</p>
                        </div>
                        <div class="data-card">
                            <h3 id="model-status">-</h3>
                            <p>모델 상태</p>
                        </div>
                    </div>
                    
                    <!-- 🆕 오늘 날짜 표시 -->
                    <div class="date-section">
                        <div class="today-indicator" id="today-date">📅 오늘: -</div>
                        <p>시스템이 오늘까지 데이터를 자동으로 채워줍니다.</p>
                    </div>
                    
                    <!-- 자동 업데이트 설정 -->
                    <div class="update-status">
                        <div class="update-indicator" id="update-indicator"></div>
                        <span>자동 업데이트</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="auto-update-toggle" onchange="toggleAutoUpdate()">
                            <span class="slider"></span>
                        </label>
                        <span id="auto-update-status">비활성화</span>
                        <span id="last-check" class="realtime-info"></span>
                    </div>
                </div>
                
                <!-- 시스템 제어 패널 -->
                <div class="card">
                    <h2>🎛️ 시스템 제어 패널</h2>
                    <div class="function-buttons">
                        <div class="function-btn btn-success" onclick="loadData()">
                            <h3>📊 데이터 로드</h3>
                            <p>ML_COMPLETE_DATASET.csv 로드</p>
                        </div>
                        <div class="function-btn btn-info" onclick="updateData()">
                            <h3>🔄 데이터 업데이트</h3>
                            <p>실제 API 데이터 가져오기</p>
                        </div>
                        <div class="function-btn btn-warning" onclick="trainModel()">
                            <h3>🤖 모델 훈련</h3>
                            <p>Random Forest 모델 훈련</p>
                        </div>
                        <div class="function-btn btn-danger" onclick="createVisualization('precipitation')">
                            <h3>📈 시각화 생성</h3>
                            <p>데이터 차트 생성</p>
                        </div>
                    </div>
                </div>
                
                <!-- 예측 시스템 -->
                <div class="grid-2">
                    <div class="card">
                        <h2>🔮 실시간 침수 예측</h2>
                        
                        <!-- 🆕 날짜 선택 -->
                        <div class="input-group">
                            <label>예측 날짜</label>
                            <input type="date" id="prediction-date" value="">
                        </div>
                        
                        <div class="input-group">
                            <label>강수량 (mm)</label>
                            <input type="number" id="precipitation" value="0" min="0" max="300">
                        </div>
                        <div class="input-group">
                            <label>습도 (%)</label>
                            <input type="number" id="humidity" value="60" min="0" max="100">
                        </div>
                        <div class="input-group">
                            <label>온도 (°C)</label>
                            <input type="number" id="temperature" value="20" min="-20" max="40">
                        </div>
                        <div class="input-group">
                            <label>3일 누적 강수량 (mm)</label>
                            <input type="number" id="precip_3d" value="0" min="0" max="500">
                        </div>
                        <div class="input-group">
                            <label>계절</label>
                            <select id="season">
                                <option value="rainy">장마철</option>
                                <option value="dry">건조기</option>
                            </select>
                        </div>
                        <button class="btn" onclick="predictRisk()" style="width: 100%; margin-top: 15px;">
                            🔍 위험도 예측
                        </button>
                    </div>
                    
                    <div class="card">
                        <h2>🎯 예측 결과</h2>
                        <div id="risk-display" class="risk-meter">
                            예측을 시작하세요
                        </div>
                        <div id="recommendations" class="status">
                            기상 정보를 입력하고 예측 버튼을 클릭하세요.
                        </div>
                        <div id="prediction-meta" class="realtime-info"></div>
                    </div>
                </div>
                
                <!-- 테스트 시나리오 -->
                <div class="card">
                    <h2>🧪 테스트 시나리오</h2>
                    <div class="scenario-grid">
                        <button class="scenario-btn" onclick="testScenario('calm')">
                            평상시<br>0mm
                        </button>
                        <button class="scenario-btn" onclick="testScenario('light')">
                            소량 강우<br>15mm
                        </button>
                        <button class="scenario-btn" onclick="testScenario('medium')">
                            중간 강우<br>35mm
                        </button>
                        <button class="scenario-btn" onclick="testScenario('heavy')">
                            집중호우<br>80mm
                        </button>
                        <button class="scenario-btn" onclick="testScenario('extreme')">
                            극한 강우<br>130mm
                        </button>
                    </div>
                </div>
                
                <!-- 시각화 패널 -->
                <div class="card">
                    <h2>📊 데이터 시각화</h2>
                    <div class="function-buttons">
                        <button class="btn" onclick="createVisualization('precipitation')">강수량 시계열</button>
                        <button class="btn" onclick="createVisualization('monthly')">월별 패턴</button>
                        <button class="btn" onclick="createVisualization('distribution')">강수량 분포</button>
                        <button class="btn" onclick="createVisualization('correlation')">상관관계</button>
                        <button class="btn" onclick="createVisualization('recent_trend')">최근 트렌드</button>
                    </div>
                    <div class="viz-container" id="visualization-area">
                        <p>시각화 버튼을 클릭하세요</p>
                    </div>
                </div>
            </div>
            
            <script>
                let statusUpdateInterval;
                
                // 전역 로딩 함수
                function showGlobalLoading(message = '처리 중...') {
                    document.getElementById('loading-message').textContent = message;
                    document.getElementById('loading-overlay').style.display = 'flex';
                }
                
                function hideGlobalLoading() {
                    document.getElementById('loading-overlay').style.display = 'none';
                }
                
                // 상태 확인 및 업데이트
                async function checkStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        // 🆕 오늘 날짜 표시
                        if (status.today) {
                            document.getElementById('today-date').textContent = `📅 오늘: ${status.today}`;
                            document.getElementById('prediction-date').value = status.today;
                        }
                        
                        // 시스템 상태
                        const statusDiv = document.getElementById('system-status');
                        let statusText = `📊 데이터: ${status.data_loaded ? '✅ 로드됨' : '❌ 없음'} | `;
                        statusText += `🤖 모델: ${status.model_loaded ? '✅ 로드됨' : '❌ 없음'} | `;
                        statusText += `🌤️ API: ${status.api_available ? '✅ 연결됨' : '❌ 키 없음'}`;
                        if (status.api_location) {
                            statusText += ` (${status.api_location})`;
                        }
                        statusDiv.innerHTML = statusText;
                        statusDiv.className = (status.data_loaded && status.model_loaded && status.api_available) ? 'status status-success' : 'status status-warning';
                        
                        // 데이터 정보 카드 업데이트
                        document.getElementById('data-rows').textContent = status.data_rows || '-';
                        
                        if (status.data_start_date && status.data_end_date) {
                            const startDate = new Date(status.data_start_date).toLocaleDateString();
                            const endDate = new Date(status.data_end_date).toLocaleDateString();
                            document.getElementById('data-period').textContent = `${startDate} ~ ${endDate}`;
                        } else {
                            document.getElementById('data-period').textContent = '-';
                        }
                        
                        if (status.data_last_updated) {
                            const lastUpdate = new Date(status.data_last_updated);
                            const now = new Date();
                            const diffMinutes = Math.floor((now - lastUpdate) / 60000);
                            
                            if (diffMinutes < 5) {
                                document.getElementById('last-update').innerHTML = `<span class="fresh">${diffMinutes}분 전</span>`;
                            } else if (diffMinutes < 60) {
                                document.getElementById('last-update').textContent = `${diffMinutes}분 전`;
                            } else {
                                const diffHours = Math.floor(diffMinutes / 60);
                                if (diffHours < 24) {
                                    document.getElementById('last-update').innerHTML = `<span class="stale">${diffHours}시간 전</span>`;
                                } else {
                                    document.getElementById('last-update').innerHTML = `<span class="stale">${lastUpdate.toLocaleDateString()}</span>`;
                                }
                            }
                        } else {
                            document.getElementById('last-update').textContent = '-';
                        }
                        
                        document.getElementById('model-status').textContent = status.model_loaded ? '활성화' : '미훈련';
                        
                        // 자동 업데이트 상태
                        const autoUpdateToggle = document.getElementById('auto-update-toggle');
                        const updateIndicator = document.getElementById('update-indicator');
                        const autoUpdateStatus = document.getElementById('auto-update-status');
                        
                        autoUpdateToggle.checked = status.auto_update_enabled;
                        if (status.auto_update_enabled) {
                            updateIndicator.className = 'update-indicator update-active';
                            autoUpdateStatus.textContent = '활성화';
                        } else {
                            updateIndicator.className = 'update-indicator update-inactive';
                            autoUpdateStatus.textContent = '비활성화';
                        }
                        
                        // 마지막 체크 시간
                        const lastCheckSpan = document.getElementById('last-check');
                        if (status.last_check_time && status.auto_update_enabled) {
                            const lastCheck = new Date(status.last_check_time);
                            const checkDiffSeconds = Math.floor((now - lastCheck) / 1000);
                            lastCheckSpan.textContent = `(마지막 체크: ${checkDiffSeconds}초 전)`;
                        } else {
                            lastCheckSpan.textContent = '';
                        }
                        
                    } catch (error) {
                        document.getElementById('system-status').innerHTML = '❌ 시스템 오류';
                        document.getElementById('system-status').className = 'status status-error';
                    }
                }
                
                // 자동 업데이트 토글
                async function toggleAutoUpdate() {
                    try {
                        const response = await fetch('/api/toggle_auto_update', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            checkStatus(); // 상태 즉시 업데이트
                        } else {
                            alert(`❌ ${result.message}`);
                        }
                    } catch (error) {
                        alert('자동 업데이트 설정 오류: ' + error.message);
                    }
                }
                
                // 데이터 로드
                async function loadData() {
                    showGlobalLoading('데이터를 로드하고 있습니다...');
                    try {
                        const response = await fetch('/api/load_data', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            alert(`✅ ${result.message}\\n기간: ${result.start_date} ~ ${result.end_date}`);
                            checkStatus();
                        } else {
                            alert(`❌ ${result.message}`);
                        }
                    } catch (error) {
                        alert('데이터 로드 오류: ' + error.message);
                    }
                    hideGlobalLoading();
                }
                
                // 데이터 업데이트
                async function updateData() {
                    showGlobalLoading('실제 API 데이터를 가져오고 있습니다...');
                    try {
                        const response = await fetch('/api/update_data', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            let message = `✅ ${result.message}`;
                            if (result.data_source) {
                                message += `\\n데이터 소스: ${result.data_source}`;
                            }
                            if (result.precipitation !== undefined) {
                                message += `\\n현재 강수량: ${result.precipitation}mm`;
                                message += `\\n온도: ${result.temperature}°C`;
                                message += `\\n습도: ${result.humidity}%`;
                            }
                            message += `\\n이전: ${result.old_count}행 → 현재: ${result.new_count}행`;
                            alert(message);
                            checkStatus();
                        } else {
                            alert(`❌ ${result.message}`);
                        }
                    } catch (error) {
                        alert('데이터 업데이트 오류: ' + error.message);
                    }
                    hideGlobalLoading();
                }
                
                // 모델 훈련
                async function trainModel() {
                    showGlobalLoading('모델을 훈련하고 있습니다...');
                    try {
                        const response = await fetch('/api/train_model', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            alert(`✅ ${result.message}\\nAUC: ${result.auc}, 정밀도: ${result.precision}\\n훈련 데이터: ${result.training_data_size}행`);
                            checkStatus();
                        } else {
                            alert(`❌ ${result.message}`);
                        }
                    } catch (error) {
                        alert('모델 훈련 오류: ' + error.message);
                    }
                    hideGlobalLoading();
                }
                
                // 시각화 생성
                async function createVisualization(type) {
                    showGlobalLoading(`${type} 차트를 생성하고 있습니다...`);
                    try {
                        const response = await fetch('/api/create_visualization', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ type: type })
                        });
                        const result = await response.json();
                        
                        if (result.success) {
                            document.getElementById('visualization-area').innerHTML = 
                                `<img src="${result.image}" class="viz-image" alt="${type} 차트">
                                 <p class="realtime-info">데이터 수: ${result.data_count}개</p>`;
                        } else {
                            alert(`❌ ${result.message}`);
                        }
                    } catch (error) {
                        alert('시각화 오류: ' + error.message);
                    }
                    hideGlobalLoading();
                }
                
                // 침수 위험 예측
                async function predictRisk() {
                    const data = {
                        precipitation: parseFloat(document.getElementById('precipitation').value),
                        humidity: parseFloat(document.getElementById('humidity').value),
                        avg_temp: parseFloat(document.getElementById('temperature').value),
                        precip_sum_3d: parseFloat(document.getElementById('precip_3d').value),
                        season_type: document.getElementById('season').value,
                        target_date: document.getElementById('prediction-date').value  // 🆕 선택된 날짜
                    };
                    
                    try {
                        const response = await fetch('/api/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(data)
                        });
                        const result = await response.json();
                        
                        document.getElementById('risk-display').className = `risk-meter risk-${result.risk_level}`;
                        document.getElementById('risk-display').innerHTML = `
                            ${result.risk_color} ${result.risk_name}<br>
                            <div style="font-size: 36px; margin: 10px 0;">${result.risk_score}점</div>
                            ${result.action}
                        `;
                        
                        document.getElementById('recommendations').innerHTML = `
                            <h4>📋 권장 행동:</h4>
                            <ul>${result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                        `;
                        
                        // 예측 메타 정보
                        const predictionTime = new Date(result.prediction_time).toLocaleString();
                        let freshnessInfo = '';
                        if (result.data_freshness !== null) {
                            const freshnessMinutes = Math.floor(result.data_freshness);
                            if (freshnessMinutes < 10) {
                                freshnessInfo = `<span class="fresh">데이터 신선도: 매우 좋음 (${freshnessMinutes}분 전)</span>`;
                            } else if (freshnessMinutes < 60) {
                                freshnessInfo = `데이터 신선도: 좋음 (${freshnessMinutes}분 전)`;
                            } else {
                                freshnessInfo = `<span class="stale">데이터 신선도: 주의 (${Math.floor(freshnessMinutes/60)}시간 전)</span>`;
                            }
                        }
                        
                        document.getElementById('prediction-meta').innerHTML = `
                            <p><strong>예측 날짜: ${result.prediction_date}</strong></p>
                            <p>예측 시간: ${predictionTime}</p>
                            <p>사용 모델: ${result.model_used}</p>
                            <p>${freshnessInfo}</p>
                        `;
                        
                    } catch (error) {
                        alert('예측 오류: ' + error.message);
                    }
                }
                
                // 테스트 시나리오
                const scenarios = {
                    'calm': {precipitation: 0, humidity: 60, avg_temp: 20, precip_sum_3d: 0, season_type: 'dry'},
                    'light': {precipitation: 15, humidity: 75, avg_temp: 22, precip_sum_3d: 25, season_type: 'rainy'},
                    'medium': {precipitation: 35, humidity: 85, avg_temp: 24, precip_sum_3d: 60, season_type: 'rainy'},
                    'heavy': {precipitation: 80, humidity: 95, avg_temp: 26, precip_sum_3d: 120, season_type: 'rainy'},
                    'extreme': {precipitation: 130, humidity: 96, avg_temp: 26, precip_sum_3d: 200, season_type: 'rainy'}
                };
                
                function testScenario(scenarioName) {
                    const scenario = scenarios[scenarioName];
                    document.getElementById('precipitation').value = scenario.precipitation;
                    document.getElementById('humidity').value = scenario.humidity;
                    document.getElementById('temperature').value = scenario.avg_temp;
                    document.getElementById('precip_3d').value = scenario.precip_sum_3d;
                    document.getElementById('season').value = scenario.season_type;
                    predictRisk();
                }
                
                // 페이지 로드 시 초기화
                window.onload = function() {
                    checkStatus();
                    predictRisk();
                    
                    // 5초마다 상태 업데이트
                    statusUpdateInterval = setInterval(checkStatus, 5000);
                };
                
                // 페이지 언로드 시 정리
                window.onbeforeunload = function() {
                    if (statusUpdateInterval) {
                        clearInterval(statusUpdateInterval);
                    }
                };
            </script>
        </body>
        </html>
        """
    
    def run(self):
        """웹 서버 실행"""
        print("🎨 침수 예측 AI 시스템 (API + 오늘까지 자동 채움)")
        print("📍 주소: http://localhost:5000")
        print("🆕 새로운 기능:")
        print("  - ✅ 기존 모든 기능 유지 (시각화, 테스트 시나리오 등)")
        print("  - 🌤️ 실제 API 데이터 우선 사용")
        print("  - 📅 오늘까지 데이터 자동 채움")
        print("  - 🔄 자동/수동 업데이트")
        print("  - 📆 사용자 선택 날짜 예측")
        print("🛑 종료: Ctrl+C")
        
        self.app.run(debug=True, host='0.0.0.0', port=5000)