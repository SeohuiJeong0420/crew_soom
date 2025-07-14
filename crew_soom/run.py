import os
import sys

# 폴더 생성
for folder in ['data/processed', 'data/raw', 'models', 'outputs']:
    os.makedirs(folder, exist_ok=True)

# 웹앱 실행
from modules.web_app import FloodWebApp

if __name__ == "__main__":
    print(" 침수 예측 AI 시스템 시작!")
    print(" 웹 인터페이스 로딩 중...")
    
    app = FloodWebApp()
    app.run()