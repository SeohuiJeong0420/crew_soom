import requests
from bs4 import BeautifulSoup
import json
import time
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd

class WeatherDataCrolling:
    def __init__(self, city, display=100):
        self.city = city
        self.display = display
        self.weather_data = self.get_today_weather_data()

    def get_today_weather_data(self):
        """네이버에서 서울 오늘 날씨 크롤링"""
        
        url = "https://search.naver.com/search.naver"
        params = {'query': '서울 날씨'}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 날씨 데이터 추출
            weather_data = {
                '날짜': datetime.now().strftime("%Y년 %m월 %d일"),
            }
            
            # 현재 온도
            temp_element = soup.select_one('.temperature_text')
            if temp_element:
                weather_data['현재온도'] = temp_element.get_text().strip()
            else:
                weather_data['현재온도'] = "정보 없음"
            
            # 날씨 상태
            weather_element = soup.select_one('.summary_list .sort:nth-child(1) .desc')
            if weather_element:
                weather_data['날씨상태'] = weather_element.get_text().strip()
            else:
                weather_data['날씨상태'] = "정보 없음"
            
            # 날씨 상세 (비, 눈, 흐림 등)
            weather_before_slash = soup.select_one('.weather.before_slash')
            if weather_before_slash:
                weather_data['날씨상세'] = weather_before_slash.get_text().strip()
            else:
                weather_data['날씨상세'] = "정보 없음"
            
            # 강수량 - 정확한 파싱
            precipitation_elements = soup.select('.summary_list .sort')
            precipitation_found = False
            
            for element in precipitation_elements:
                text = element.get_text()
                if '강수량' in text or 'mm' in text:
                    desc = element.select_one('.desc')
                    if desc:
                        precip_text = desc.get_text().strip()
                        # mm 단위만 추출, % 무시
                        if 'mm' in precip_text and '%' not in precip_text:
                            weather_data['강수량'] = precip_text
                            precipitation_found = True
                            break
                        elif precip_text == '0mm' or '0mm' in precip_text:
                            weather_data['강수량'] = '0mm'
                            precipitation_found = True
                            break
            
            if not precipitation_found:
                weather_data['강수량'] = '0mm'

            # 미세먼지
            dust_elements = soup.select('.today_chart_list .item_today')
            for element in dust_elements:
                text = element.get_text()
                if '미세먼지' in text and '초미세먼지' not in text:
                    weather_data['미세먼지'] = element.select_one('.txt').get_text().strip()
                elif '초미세먼지' in text:
                    weather_data['초미세먼지'] = element.select_one('.txt').get_text().strip()
            
            # 기본값 설정
            if '미세먼지' not in weather_data:
                weather_data['미세먼지'] = "정보 없음"
            if '초미세먼지' not in weather_data:
                weather_data['초미세먼지'] = "정보 없음"
            
            print(" 날씨 크롤링 성공!")
            return weather_data
            
        except Exception as e:
            print(f" 크롤링 실패: {e}")
            return None

    @staticmethod    
    def save_today_weather(weather_data):
        """오늘 날씨를 고정 파일명으로 업데이트"""
        
        filename = "today_data/오늘날씨.xlsx"  # 항상 같은 파일명
        
        if weather_data:
            # 새 데이터를 DataFrame으로 변환
            df = pd.DataFrame([weather_data])
            
            # 기존 파일 덮어쓰기 (매일 새로운 내용으로 교체)
            df.to_excel('today_data/오늘날씨.xlsx', index=False)
            
            print(f"오늘 날씨 업데이트 완료!")
            print(f"파일: {filename}")
            print(f"날짜: {weather_data['날짜']}")
            print(f"온도: {weather_data['현재온도']}")
            
            return filename
        else:
            print("저장할 날씨 데이터가 없습니다.")
            return None

    @staticmethod    
    def save_weather_with_backup(weather_data):
        """백업 포함 업데이트"""
        
        filename = "today_data/오늘날씨.xlsx"
        
        if weather_data:
            # 기존 파일이 있으면 백업
            if os.path.exists(filename):
                backup_time = datetime.now().strftime("%Y%m%d")
                backup_name = f"어제날씨_{backup_time}.xlsx"
                
                # 어제 파일을 백업으로 이름 변경
                try:
                    import shutil
                    shutil.copy(filename, backup_name)
                    print(f"어제 날씨 백업: {backup_name}")
                except:
                    pass
            
            # 오늘 날씨로 덮어쓰기
            df = pd.DataFrame([weather_data])
            df.to_excel(filename, index=False)
            
            print(f"오늘 날씨 업데이트!")
            print(f"{weather_data['날짜']} - {weather_data['현재온도']}")
            
            return filename
        
        return None
    
    @staticmethod
    def update_today_weather():
        """매일 실행하는 함수"""
        
        # 날씨 크롤링 (기존 함수 사용)
        weather_data = WeatherDataCrolling.get_today_weather_data()
        
        if weather_data:
            # 파일 업데이트
            WeatherDataCrolling.save_today_weather(weather_data)
            
            print("\n 업데이트 완료!")
            print("매일 같은 파일이 오늘 날씨로 갱신됩니다.")
        else:
            print("날씨 크롤링 실패")

def main():
    """메인 실행 함수"""
    
    # 1. 날씨 크롤링
    crewler = WeatherDataCrolling("서울")
    weather_data = crewler.get_today_weather_data()
    
    # 2. 업데이트 여부 확인
    if weather_data:
        # 파일 업데이트
        WeatherDataCrolling.save_today_weather(weather_data)
        
        print("\n 업데이트 완료!")
        print("매일 같은 파일이 오늘 날씨로 갱신됩니다.")
    else:
        print("날씨 크롤링 실패")  

if __name__ == "__main__":
    main()