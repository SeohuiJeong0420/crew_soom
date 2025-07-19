# modules/data_loader.py
import pandas as pd
import os

class DataLoader:
    """기존 STRATEGIC_FLOOD_DATA 활용 데이터 로더"""
    
    def __init__(self):
        self.data_paths = {
            'ml_ready': 'data/processed/ML_COMPLETE_DATASET.csv',
            'daily_all': 'data/raw/daily/daily_all_3years.csv',
            'flood_events': 'data/flood_events/actual_flood_events_2022_2024.csv'
        }
    
    def load_ml_ready_data(self):
        """ML 준비된 데이터 로드"""
        try:
            df = pd.read_csv(self.data_paths['ml_ready'])
            df['obs_date'] = pd.to_datetime(df['obs_date'])
            print(f"✅ ML 데이터 로드: {len(df)}행")
            return df
        except FileNotFoundError:
            print("❌ ML_COMPLETE_DATASET.csv를 찾을 수 없습니다.")
            print("💡 STRATEGIC_FLOOD_DATA/4_ML_READY/에서 파일을 복사하세요.")
            return None
    
    def load_flood_events(self):
        """실제 침수 사건 데이터 로드"""
        try:
            df = pd.read_csv(self.data_paths['flood_events'])
            df['flood_date'] = pd.to_datetime(df['flood_date'])
            print(f"✅ 침수 사건 데이터 로드: {len(df)}건")
            return df
        except FileNotFoundError:
            print("❌ 침수 사건 데이터를 찾을 수 없습니다.")
            return None
    
    def get_data_info(self):
        """데이터 정보 반환"""
        info = {}
        
        for name, path in self.data_paths.items():
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    info[name] = {
                        'exists': True,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': os.path.getsize(path) / 1024 / 1024
                    }
                except:
                    info[name] = {'exists': False}
            else:
                info[name] = {'exists': False}
        
        return info