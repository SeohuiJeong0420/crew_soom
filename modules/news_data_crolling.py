import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from dotenv import load_dotenv
import os
import urllib.request
import re
import urllib.parse
from pathlib import Path
from datetime import datetime

class NewsDataCrolling:
    def __init__(self, word, display=100):
        self.word = word
        self.display = display
        self.news_data = self.getnews_data(word, display)

    def getnews_data(self, word, display=100):

        load_dotenv()
        client_id = os.getenv('Client_ID')
        client_secret = os.getenv('Client_Secret')
        encoded_query = urllib.parse.quote(word)  # 한글을 URL 인코딩
        url = 'https://openapi.naver.com/v1/search/news.json?query={}&display={}'.format(encoded_query, display)
        
        headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret
        }
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                items = response.json()['items']
                news_list = []
                
                for idx, item in enumerate(items):
                    # HTML 태그 제거
                    link = item.get('link')
                    title = re.sub('<.*?>', '', item.get('title', ''))
                    title = title.replace('&quot;', '"').replace('&amp;', '&')
                    description = re.sub('<.*?>', '', item.get('description', ''))
                    description = description.replace('&quot;', '"').replace('&amp;', '&')
                    
                    news_data = {
                        'no': idx + 1,
                        'title': title, #뉴스 제목
                        'link': item.get('link', ''), #뉴스 링크
                        'description': description, #뉴스 요약
                        'pubDate': item.get('pubDate', ''), #발행 날짜
                        'collected_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S") #데이터 수집 기간
                    }
                    news_list.append(news_data)
                    
                    # 진행상황 출력
                    #print(f"{idx + 1}. {title}")
                
                print(f"'{word}' 관련 뉴스 {len(news_list)}개 수집 완료")
                news_df = pd.DataFrame(news_list)
                return news_df
                
            else:
                print(f"API 오류: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"오류 발생: {e}")
            return pd.DataFrame()
        
    @staticmethod #정적 메서드
    def filter_weather_news(news_df):
        """날씨 관련 뉴스만 필터링"""
        weather_keywords = ['날씨', '기온', '온도', '비', '눈', '바람', '습도', '기상', '예보', '폭염', '한파', '태풍']
        
        filtered_news = []
        for _, row in news_df.iterrows():
            title = row['title'].lower()
            
            # 제목에 날씨 키워드가 포함된 경우만
            if any(keyword in title for keyword in weather_keywords):
                filtered_news.append(row)
        
        return pd.DataFrame(filtered_news)

    @staticmethod #정적 메서드
    def update_and_save_news(word, display=100, file_path=None, max_rows=100):
        #1. 파일 경로 설정
        if file_path is None:
            file_path = f"news_data/{word}_news.xlsx"

        print(f"=== '{word}' 뉴스 업데이트 시작 ===")
        print(f"수집 개수: {display}개")
        print(f"저장 경로: {file_path}")
        print("-" * 50)

        #2. 기존 데이터 로드
        existing_df = pd.DataFrame()
        if Path(file_path).exists():
            try:
                existing_df = pd.read_excel(file_path)
                print(f"기존 데이터 로드 완료: {len(existing_df)} 개")
            except Exception as e:
                print(f"기존 데이터 로드 오류 : {e}")
        else:
            print(f"기존 데이터 로드 실패 : {file_path} 파일이 존재하지 않음. 새로 생성 요망")

        #3. 새로운 뉴스 데이터 수집
        crawler = NewsDataCrolling(word, display) #객체 생성
        new_news_df = crawler.getnews_data(word, display)  # 메서드 호출
        new_news_df = NewsDataCrolling.filter_weather_news(new_news_df) #뉴스 필터링
        if new_news_df.empty:
            print("새로운 데이터 수집에 실패했습니다.")
            return False
        
        # 4. 중복 제거
        if existing_df.empty:
            new_df_filtered = new_news_df
            duplicates_count = 0
        else:
            new_df_filtered = new_news_df[~new_news_df['link'].isin(existing_df['link'])]
            duplicates_count = len(new_news_df) - len(new_df_filtered)
        
        print(f"중복 제거: {duplicates_count}개")
        print(f"새로운 뉴스: {len(new_df_filtered)}개")
        
        # 5. 새로운 뉴스가 없으면 종료
        if new_df_filtered.empty:
            print("새로운 뉴스가 없습니다.")
            return False
        
        # 6. 데이터 병합 (새로운 뉴스를 맨 위에 추가)
        if existing_df.empty:
            final_df = new_df_filtered
        else:
            final_df = pd.concat([new_df_filtered, existing_df], ignore_index=True)
        
        # 7. no 컬럼 재정렬
        final_df['no'] = range(1, len(final_df) + 1)
        
        # 8. 엑셀 파일로 저장
        try:
            final_df.to_excel(file_path, index=False)
            
            print(f"'{word}' 뉴스 데이터가 '{file_path}'에 저장되었습니다.")
            print(f"새로 추가된 뉴스: {len(new_df_filtered)}개")
            print(f"전체 뉴스 개수: {len(final_df)}개")
            
            return final_df
            
        except Exception as e:
            print(f"파일 저장 오류: {e}")
            return False
    
def main():
    """메인 실행 함수"""
    
    # 1. 간단한 뉴스 수집
    print("날씨 뉴스 수집 시작...")
    crawler = NewsDataCrolling("서울날씨뉴스", 50)
    
    # 2. 뉴스 업데이트 및 저장
    print("\n뉴스 파일 업데이트...")
    result = NewsDataCrolling.update_and_save_news(
        word="서울날씨뉴스", 
        display=100, 
        file_path="news_data/서울날씨뉴스_news.xlsx",
        max_rows=100
    )
    
    if isinstance(result, pd.DataFrame):
        print(f"\n최종 결과: {len(result)}개 뉴스 저장 완료")
        print("최신 뉴스 3개:")
        for i, row in result.head(3).iterrows():
            print(f"  {i+1}. {row['title']}")
    else:
        print("뉴스 업데이트 실패")

if __name__ == "__main__":
    main()
