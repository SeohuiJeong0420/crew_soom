import os
import time
import requests
import pandas as pd
import datetime
from dotenv import load_dotenv

load_dotenv('.env')

# 설정
SERVICE_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_DIR = "data"
os.makedirs(BASE_DIR, exist_ok=True)

# 공통 설정
STN_ID = 108
NUM_OF_ROWS = 800
START_DATE = datetime.date(2000, 1, 10)
END_DATE = datetime.date.today() - datetime.timedelta(days=1)

# 1. 시간별 관측자료 수집 (ASOS)
def get_hourly_data():
    """시간별 데이터 수집"""
    CSV_FILE = os.path.join(BASE_DIR, 'asos_seoul_hourly.csv')
    API_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    def load_existing_times():
        """기존에 수집된 시간 데이터 로드"""
        if not os.path.exists(CSV_FILE):
            return set()
        try:
            df = pd.read_csv(CSV_FILE)
            return set(df['tm'].astype(str))
        except Exception as e:
            print(f"기존 데이터 로드 실패: {e}")
            return set()

    def fetch_day_data(date_obj, existing_times):
        """지정된 하루치(00~23시) 데이터를 수집"""
        all_results = []
        start_dt = datetime.datetime.combine(date_obj, datetime.time(0, 0))
        end_dt = datetime.datetime.combine(date_obj, datetime.time(23, 0))

        page = 1
        while True:
            params = {
                'serviceKey': SERVICE_KEY,
                'numOfRows': NUM_OF_ROWS,
                'pageNo': page,
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': start_dt.strftime("%Y%m%d"),
                'startHh': "00",
                'endDt': end_dt.strftime("%Y%m%d"),
                'endHh': "23",
                'stnIds': STN_ID,
                'dataType': 'JSON'
            }

            try:
                response = requests.get(API_URL, params=params, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"[{date_obj}] 요청 실패: {e}")
                return []

            try:
                data = response.json()
                result_code = data['response']['header']['resultCode']
                result_msg = data['response']['header']['resultMsg']
                if result_code != '00':
                    print(f"[{date_obj}] API 오류: {result_msg}")
                    return []
            except Exception as e:
                print(f"[{date_obj}] JSON 파싱 실패: {e}")
                return []

            items = data['response']['body']['items'].get('item', [])
            if not items:
                break

            # 새로운 데이터만 추가
            for item in items:
                if item['tm'] not in existing_times:
                    all_results.append(item)

            if len(items) < NUM_OF_ROWS:
                break
            page += 1

        print(f"[{date_obj}] {len(all_results)}건 수집 성공")
        return all_results

    def append_to_csv(new_data):
        """새로운 데이터를 CSV에 추가"""
        if not new_data:
            return
        
        df = pd.DataFrame(new_data)
        
        # 기존 파일과 동일한 컬럼 순서로 정렬 (있는 컬럼만)
        target_columns = ['tm', 'ta', 'rn', 'ws', 'wd', 'hm', 'pa', 'ps', 'td', 'pv']
        available_columns = [col for col in target_columns if col in df.columns]
        df = df[available_columns]
        
        # 수치형 컬럼 변환 (tm 제외)
        for col in available_columns:
            if col != 'tm':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        write_header = not os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=write_header, index=False)

    def get_last_collected_date():
        """마지막으로 수집된 날짜 확인"""
        if not os.path.exists(CSV_FILE):
            return START_DATE
        try:
            df = pd.read_csv(CSV_FILE)
            last_time = pd.to_datetime(df['tm']).max()
            return last_time.date() + datetime.timedelta(days=1)
        except Exception as e:
            print(f"마지막 수집일 확인 실패: {e}")
            return START_DATE

    # 데이터 수집 실행
    print("시간별 데이터 수집 시작...")
    existing_times = load_existing_times()
    current_date = get_last_collected_date()
    
    print(f"현재 수집된 데이터: {len(existing_times):,}개 시간대")
    print(f"수집 시작 날짜: {current_date}")
    print(f"수집 종료 날짜: {END_DATE}")

    if current_date > END_DATE:
        print("이미 최신 데이터까지 모두 수집되어 있습니다!")
        return

    collected_count = 0
    while current_date <= END_DATE:
        new_data = fetch_day_data(current_date, existing_times)
        if new_data:
            append_to_csv(new_data)
            collected_count += len(new_data)
            
            # 새로 수집한 데이터의 시간을 기존 set에 추가
            for item in new_data:
                existing_times.add(item['tm'])
            
        current_date += datetime.timedelta(days=1)
        time.sleep(0.5)  # API 과부하 방지

    if collected_count == 0:
        print("새로운 시간별 데이터가 없습니다. 이미 최신 상태입니다!")
    else:
        print(f"시간별 데이터 수집 완료! (새로 추가된 데이터: {collected_count:,}건)")


# 2. 일별 관측자료 수집 (ASOS)
def get_daily_data():
    """일별 데이터 수집"""
    CSV_FILE = os.path.join(BASE_DIR, 'asos_seoul_daily.csv')
    API_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

    def load_existing_dates():
        """기존에 수집된 날짜 데이터 로드"""
        if not os.path.exists(CSV_FILE):
            return set()
        try:
            df = pd.read_csv(CSV_FILE)
            return set(pd.to_datetime(df['tm']).dt.date.astype(str))
        except Exception as e:
            print(f"기존 데이터 로드 실패: {e}")
            return set()

    def fetch_day_data(date_obj):
        """특정 날짜의 일별 데이터 수집"""
        params = {
            'serviceKey': SERVICE_KEY,
            'numOfRows': NUM_OF_ROWS,
            'pageNo': 1,
            'dataCd': 'ASOS',
            'dateCd': 'DAY',
            'startDt': date_obj.strftime("%Y%m%d"),
            'endDt': date_obj.strftime("%Y%m%d"),
            'stnIds': STN_ID,
            'dataType': 'JSON'
        }

        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            result_code = data['response']['header']['resultCode']
            if result_code != '00':
                print(f"[{date_obj}] API 오류: {data['response']['header']['resultMsg']}")
                return []
                
            items = data['response']['body']['items'].get('item', [])
            return items
        except Exception as e:
            print(f"[{date_obj}] 에러 발생: {e}")
            return []

    def append_to_csv(items):
        """새로운 데이터를 CSV에 추가"""
        if not items:
            return

        df = pd.DataFrame(items)

        # 기존 파일과 동일한 컬럼 순서로 정렬 (있는 컬럼만)
        target_columns = [
            'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'avgWs', 'avgRhm',
            'avgTs', 'ddMefs', 'sumGsr', 'maxInsWs', 'sumSmlEv', 'avgTd', 'avgPs'
        ]
        available_columns = [col for col in target_columns if col in df.columns]
        df = df[available_columns]

        # 수치형 컬럼 변환 (tm 제외)
        for col in available_columns:
            if col != 'tm':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # CSV 저장
        write_header = not os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=write_header, index=False)

    def get_last_collected_date():
        """마지막으로 수집된 날짜 확인"""
        if not os.path.exists(CSV_FILE):
            return START_DATE
        try:
            df = pd.read_csv(CSV_FILE)
            last_date = pd.to_datetime(df['tm']).max().date()
            return last_date + datetime.timedelta(days=1)
        except Exception as e:
            print(f"마지막 수집일 확인 실패: {e}")
            return START_DATE

    # 데이터 수집 실행
    print("일별 데이터 수집 시작...")
    existing_dates = load_existing_dates()
    current_date = get_last_collected_date()
    
    print(f"현재 수집된 데이터: {len(existing_dates):,}일")
    print(f"수집 시작 날짜: {current_date}")
    print(f"수집 종료 날짜: {END_DATE}")

    if current_date > END_DATE:
        print("이미 최신 데이터까지 모두 수집되어 있습니다!")
        return

    collected_count = 0
    while current_date <= END_DATE:
        if str(current_date) in existing_dates:
            current_date += datetime.timedelta(days=1)
            continue

        items = fetch_day_data(current_date)
        if items:
            append_to_csv(items)
            collected_count += len(items)
            print(f"[{current_date}] 저장 완료")
        else:
            print(f"[{current_date}] 데이터 없음")

        current_date += datetime.timedelta(days=1)
        time.sleep(0.3)  # API 과부하 방지

    if collected_count == 0:
        print("새로운 일별 데이터가 없습니다. 이미 최신 상태입니다!")
    else:
        print(f"일별 데이터 수집 완료! (새로 추가된 데이터: {collected_count:,}건)")


# 3. 데이터 상태 확인 함수
def check_data_status():
    """현재 수집된 데이터 상태 확인"""
    print("=== 데이터 상태 확인 ===")
    
    # 시간별 데이터 상태
    hourly_file = os.path.join(BASE_DIR, 'asos_seoul_hourly.csv')
    if os.path.exists(hourly_file):
        df_hourly = pd.read_csv(hourly_file)
        hourly_dates = pd.to_datetime(df_hourly['tm'])
        print(f"시간별 데이터 (asos_seoul_hourly.csv):")
        print(f"   - 총 {len(df_hourly):,}건")
        print(f"   - 기간: {hourly_dates.min()} ~ {hourly_dates.max()}")
        print(f"   - 최근 데이터: {hourly_dates.max()}")
        print(f"   - 컬럼: {list(df_hourly.columns)}")
    else:
        print("시간별 데이터: 파일이 없습니다.")
    
    # 일별 데이터 상태  
    daily_file = os.path.join(BASE_DIR, 'asos_seoul_daily.csv')
    if os.path.exists(daily_file):
        df_daily = pd.read_csv(daily_file)
        daily_dates = pd.to_datetime(df_daily['tm'])
        print(f"일별 데이터 (asos_seoul_daily.csv):")
        print(f"   - 총 {len(df_daily):,}건")
        print(f"   - 기간: {daily_dates.min().date()} ~ {daily_dates.max().date()}")
        print(f"   - 최근 데이터: {daily_dates.max().date()}")
        print(f"   - 컬럼: {list(df_daily.columns)}")
    else:
        print("일별 데이터: 파일이 없습니다.")
    
    print()


# 메인 실행 함수
def preprocess_data():
    """전체 날씨 데이터 수집 실행 (기존 함수명 유지)"""
    print("=== 서울 날씨 데이터 수집 시작 ===")
    
    # 0. 현재 데이터 상태 확인
    print("\n[0단계] 현재 데이터 상태 확인...")
    check_data_status()
    
    # 1. 데이터 수집 (증분 업데이트)
    print("\n[1단계] 시간별/일별 날씨 데이터 수집 중...")
    get_hourly_data()
    print()
    get_daily_data()
    
    # 2. 업데이트 후 데이터 상태 재확인
    print("\n[2단계] 업데이트 후 데이터 상태 확인...")
    check_data_status()
    
    print("=== 전체 작업 완료! ===")
    print(f"📁 데이터 저장 위치: {os.path.abspath(BASE_DIR)}/")


def collect_weather_data():
    """전체 날씨 데이터 수집 실행 (별칭)"""
    preprocess_data()


def update_data_only():
    """데이터 수집만 실행"""
    print("=== 데이터 업데이트 모드 ===")
    check_data_status()
    print("새로운 데이터 확인 중...")
    get_hourly_data()
    print()
    get_daily_data()
    print("\n업데이트 완료!")
    check_data_status()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_data_status()
        elif sys.argv[1] == "update":
            update_data_only()
        elif sys.argv[1] == "collect":
            preprocess_data()
        else:
            print("사용법:")
            print("  python preprocessor.py check   - 데이터 상태만 확인")
            print("  python preprocessor.py update  - 새로운 데이터만 수집")
            print("  python preprocessor.py collect - 전체 수집 프로세스 실행")
    else:
        # 기본 실행: 전체 수집 프로세스
        preprocess_data()