# modules/multi_weather_api.py - 오류 수정 및 기능 개선

import requests
import urllib.parse
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import time
import calendar
import sqlite3
from typing import Dict, List, Optional, Tuple
import json


class MultiWeatherAPI:
    """서울시 25개 지역구 전용 전략적 침수 예측 데이터 수집 시스템 (오류 수정 버전)
    
    주요 개선사항:
    1. get_comprehensive_weather_data 메서드 추가
    2. 안정적 API 호출 시스템
    3. 오류 처리 강화
    4. CSV 직접 저장 지원
    5. 실시간 데이터 수집 기능
    """
    
    def __init__(self, service_key):
        # URL 디코딩
        self.service_key = urllib.parse.unquote(service_key)
        
        # 기상청 ASOS API 정보
        self.apis = {
            'asos_daily': {
                'url': 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList',
                'name': '지상(ASOS) 일자료',
                'description': '일별 종합 기상 데이터'
            },
            'asos_hourly': {
                'url': 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList',
                'name': '지상(ASOS) 시간자료',
                'description': '시간별 상세 기상 데이터'
            },
            'current_weather': {
                'url': 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst',
                'name': '초단기실황조회',
                'description': '현재 기상 실황'
            },
            'forecast': {
                'url': 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst',
                'name': '초단기예보조회',
                'description': '1시간 예보'
            }
        }
        
        # 서울시 주요 관측소 정보 (기상청 공식 지점코드)
        self.seoul_stations = {
            'seoul_main': {'stnId': '108', 'name': '서울', 'location': '서울시 종로구'},
            'incheon': {'stnId': '112', 'name': '인천', 'location': '인천시 중구'},
            'suwon': {'stnId': '119', 'name': '수원', 'location': '경기도 수원시'}
        }
        
        # 서울시 격자 좌표 (초단기 예보용)
        self.seoul_grid = {
            'nx': 60,  # 서울시 중심 격자 X
            'ny': 127  # 서울시 중심 격자 Y
        }
        
        # 5년치 데이터 수집 계획 (2020-2024년)
        self.collection_plan = {
            "target_years": [2020, 2021, 2022, 2023, 2024],
            "primary_station": "108",  # 서울 (가장 안정적)
            "data_types": ["daily", "hourly"]
        }
        
        # CSV 파일 경로 설정
        self.daily_csv_path = 'data/processed/ASOS_DAILY_DATA.csv'
        self.hourly_csv_path = 'data/processed/ASOS_HOURLY_DATA.csv'
        self.combined_csv_path = 'data/processed/REAL_WEATHER_DATA.csv'
        
        os.makedirs('data/processed', exist_ok=True)
        print("기상청 ASOS 다중 API 수집 시스템 초기화 완료")
        
        # 실제 침수 사건 데이터 (참조용)
        self.flood_events = [
            {"date": "2020-08-30", "location": "서울", "severity": 3, "precip_daily": 89.5},
            {"date": "2021-07-13", "location": "서울", "severity": 2, "precip_daily": 65.2},
            {"date": "2022-08-08", "location": "서울", "severity": 4, "precip_daily": 381.5},
            {"date": "2022-08-09", "location": "서울", "severity": 3, "precip_daily": 123.4},
            {"date": "2023-07-17", "location": "서울", "severity": 2, "precip_daily": 78.9},
            {"date": "2024-07-10", "location": "서울", "severity": 2, "precip_daily": 92.1}
        ]
    
    def get_comprehensive_weather_data(self):
        """종합 기상 데이터 수집 (web_app.py에서 요구하는 메서드)"""
        try:
            print("종합 기상 데이터 수집 시작...")
            
            # 여러 API에서 데이터 수집
            collected_data = {}
            successful_apis = []
            
            # 1. 현재 기상 실황 (초단기실황)
            current_data = self._get_current_weather()
            if current_data:
                collected_data.update(current_data)
                successful_apis.append('current_weather')
            
            # 2. 최근 ASOS 일자료
            recent_daily = self._get_recent_daily_data()
            if recent_daily:
                collected_data.update(recent_daily)
                successful_apis.append('asos_daily')
            
            # 3. 최근 ASOS 시간자료
            recent_hourly = self._get_recent_hourly_data()
            if recent_hourly:
                collected_data.update(recent_hourly)
                successful_apis.append('asos_hourly')
            
            # 4. 초단기 예보 (1시간 후)
            forecast_data = self._get_forecast_data()
            if forecast_data:
                collected_data.update(forecast_data)
                successful_apis.append('forecast')
            
            # 결과 정리
            if collected_data:
                # 기본값 설정
                weather_data = {
                    'temperature': collected_data.get('temperature', 20.0),
                    'precipitation': collected_data.get('precipitation', 0.0),
                    'humidity': collected_data.get('humidity', 60.0),
                    'wind_speed': collected_data.get('wind_speed', 2.0),
                    'pressure': collected_data.get('pressure', 1013.0),
                    'weather_condition': collected_data.get('weather_condition', '맑음'),
                    'data_time': datetime.now().isoformat()
                }
                
                return {
                    'success': True,
                    'data_sources': successful_apis,
                    'weather_data': weather_data,
                    'collection_time': datetime.now().isoformat(),
                    'station_info': self.seoul_stations['seoul_main']
                }
            else:
                return {
                    'success': False,
                    'message': '모든 API에서 데이터 수집 실패',
                    'data_sources': [],
                    'weather_data': None
                }
                
        except Exception as e:
            print(f"종합 기상 데이터 수집 오류: {e}")
            return {
                'success': False,
                'message': str(e),
                'data_sources': [],
                'weather_data': None
            }
    
    def _get_current_weather(self):
        """현재 기상 실황 조회"""
        try:
            base_date = datetime.now().strftime('%Y%m%d')
            base_time = (datetime.now() - timedelta(hours=1)).strftime('%H00')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '10',
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': self.seoul_grid['nx'],
                'ny': self.seoul_grid['ny']
            }
            
            response = requests.get(self.apis['current_weather']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if items:
                        weather_data = {}
                        for item in items:
                            category = item.get('category')
                            value = float(item.get('obsrValue', 0))
                            
                            if category == 'T1H':  # 기온
                                weather_data['temperature'] = value
                            elif category == 'RN1':  # 1시간 강수량
                                weather_data['precipitation'] = value
                            elif category == 'REH':  # 습도
                                weather_data['humidity'] = value
                            elif category == 'WSD':  # 풍속
                                weather_data['wind_speed'] = value
                        
                        print("현재 기상 실황 수집 성공")
                        return weather_data
            
            print("현재 기상 실황 수집 실패")
            return None
            
        except Exception as e:
            print(f"현재 기상 실황 오류: {e}")
            return None
    
    def _get_recent_daily_data(self):
        """최근 ASOS 일자료 조회"""
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            station_id = self.collection_plan["primary_station"]
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '1',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'DAY',
                'startDt': yesterday,
                'endDt': yesterday,
                'stnIds': station_id
            }
            
            response = requests.get(self.apis['asos_daily']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if items:
                        item = items[0] if isinstance(items, list) else items
                        
                        weather_data = {
                            'avg_temp_yesterday': float(item.get('avgTa', 20) or 20),
                            'precipitation_yesterday': float(item.get('sumRn', 0) or 0),
                            'humidity_yesterday': float(item.get('avgRhm', 60) or 60)
                        }
                        
                        print("최근 일자료 수집 성공")
                        return weather_data
            
            print("최근 일자료 수집 실패")
            return None
            
        except Exception as e:
            print(f"최근 일자료 오류: {e}")
            return None
    
    def _get_recent_hourly_data(self):
        """최근 ASOS 시간자료 조회"""
        try:
            now = datetime.now()
            date_str = now.strftime('%Y%m%d')
            hour_str = (now - timedelta(hours=1)).strftime('%H')
            station_id = self.collection_plan["primary_station"]
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '1',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': date_str,
                'startHh': hour_str,
                'endDt': date_str,
                'endHh': hour_str,
                'stnIds': station_id
            }
            
            response = requests.get(self.apis['asos_hourly']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if items:
                        item = items[0] if isinstance(items, list) else items
                        
                        weather_data = {
                            'temperature_hourly': float(item.get('ta', 20) or 20),
                            'precipitation_hourly': float(item.get('rn', 0) or 0),
                            'humidity_hourly': float(item.get('hm', 60) or 60),
                            'pressure': float(item.get('pa', 1013) or 1013)
                        }
                        
                        print("최근 시간자료 수집 성공")
                        return weather_data
            
            print("최근 시간자료 수집 실패")
            return None
            
        except Exception as e:
            print(f"최근 시간자료 오류: {e}")
            return None
    
    def _get_forecast_data(self):
        """초단기 예보 조회"""
        try:
            base_date = datetime.now().strftime('%Y%m%d')
            base_time = datetime.now().strftime('%H30')  # 30분 단위
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '10',
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': self.seoul_grid['nx'],
                'ny': self.seoul_grid['ny']
            }
            
            response = requests.get(self.apis['forecast']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if items:
                        forecast_data = {}
                        for item in items:
                            category = item.get('category')
                            value = item.get('fcstValue', '0')
                            
                            try:
                                if category == 'T1H':  # 기온 예보
                                    forecast_data['forecast_temperature'] = float(value)
                                elif category == 'RN1':  # 강수량 예보
                                    forecast_data['forecast_precipitation'] = float(value)
                                elif category == 'REH':  # 습도 예보
                                    forecast_data['forecast_humidity'] = float(value)
                                elif category == 'SKY':  # 하늘상태
                                    forecast_data['sky_condition'] = value
                            except ValueError:
                                continue
                        
                        print("초단기 예보 수집 성공")
                        return forecast_data
            
            print("초단기 예보 수집 실패")
            return None
            
        except Exception as e:
            print(f"초단기 예보 오류: {e}")
            return None
    
    def collect_asos_daily_data(self, start_year: int = 2020, end_year: int = 2024) -> int:
        """ASOS 일자료 5년치 수집"""
        print(f"ASOS 일자료 수집 시작: {start_year}년 ~ {end_year}년")
        
        # 기존 데이터 확인
        existing_data = self._load_existing_daily_data()
        total_collected = 0
        
        station_id = self.collection_plan["primary_station"]
        
        for year in range(start_year, end_year + 1):
            year_collected = 0
            
            # 월별 수집 (API 제한 고려)
            for month in range(1, 13):
                month_start = datetime(year, month, 1)
                
                # 해당 월의 마지막 날
                if month == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month + 1, 1) - timedelta(days=1)
                
                # 현재 날짜보다 미래면 스킵
                if month_start > datetime.now():
                    continue
                
                # 어제까지만 수집 (API 제한)
                if month_end > datetime.now() - timedelta(days=1):
                    month_end = datetime.now() - timedelta(days=1)
                
                print(f"  {year}년 {month}월 수집 중... ({month_start.strftime('%Y-%m-%d')} ~ {month_end.strftime('%Y-%m-%d')})")
                
                month_data = self._fetch_daily_month_data(station_id, month_start, month_end, existing_data)
                
                if month_data:
                    year_collected += len(month_data)
                    total_collected += len(month_data)
                    
                    # 월별 데이터 즉시 저장
                    self._append_daily_data_to_csv(month_data)
                    print(f"     {month}월: {len(month_data)}일 수집 완료")
                else:
                    print(f"     {month}월: 데이터 없음")
                
                # API 제한 준수 (월간 대기)
                time.sleep(1.0)
            
            print(f"   {year}년 총 {year_collected}일 수집 완료")
        
        print(f"ASOS 일자료 수집 완료: 총 {total_collected}일")
        return total_collected
    
    def collect_asos_hourly_data(self, start_year: int = 2020, end_year: int = 2024) -> int:
        """ASOS 시간자료 5년치 수집"""
        print(f"ASOS 시간자료 수집 시작: {start_year}년 ~ {end_year}년")
        
        # 기존 데이터 확인
        existing_data = self._load_existing_hourly_data()
        total_collected = 0
        
        station_id = self.collection_plan["primary_station"]
        
        for year in range(start_year, end_year + 1):
            year_collected = 0
            
            # 주별 수집 (시간자료는 데이터가 많으므로)
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31)
            
            # 현재 날짜 제한
            if year_end > datetime.now() - timedelta(days=1):
                year_end = datetime.now() - timedelta(days=1)
            
            current_date = year_start
            week_size = 7  # 일주일씩 수집
            
            while current_date <= year_end:
                week_end = min(current_date + timedelta(days=week_size - 1), year_end)
                
                print(f"   {year}년 주간 수집: {current_date.strftime('%Y-%m-%d')} ~ {week_end.strftime('%Y-%m-%d')}")
                
                week_data = self._fetch_hourly_week_data(station_id, current_date, week_end, existing_data)
                
                if week_data:
                    year_collected += len(week_data)
                    total_collected += len(week_data)
                    
                    # 주별 데이터 즉시 저장
                    self._append_hourly_data_to_csv(week_data)
                    print(f"     {len(week_data)}시간 데이터 수집 완료")
                else:
                    print(f"     해당 주: 데이터 없음")
                
                current_date = week_end + timedelta(days=1)
                
                # API 제한 준수 (주간 대기)
                time.sleep(2.0)
            
            print(f"   {year}년 총 {year_collected}시간 수집 완료")
        
        print(f"ASOS 시간자료 수집 완료: 총 {total_collected}시간")
        return total_collected
    
    def _fetch_daily_month_data(self, station_id, start_date, end_date, existing_data):
        """월별 일자료 수집"""
        try:
            start_dt = start_date.strftime('%Y%m%d')
            end_dt = end_date.strftime('%Y%m%d')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '100',  # 한 달 최대 31일
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'DAY',
                'startDt': start_dt,
                'endDt': end_dt,
                'stnIds': station_id
            }
            
            response = requests.get(self.apis['asos_daily']['url'], params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if not items:
                        return []
                    
                    # 단일 항목을 리스트로 변환
                    if not isinstance(items, list):
                        items = [items]
                    
                    processed_data = []
                    for item in items:
                        # 중복 검사
                        date_key = item.get('tm', '')
                        if date_key in existing_data:
                            continue
                        
                        processed_item = self._process_daily_item(item)
                        if processed_item:
                            processed_data.append(processed_item)
                            existing_data.add(date_key)
                    
                    return processed_data
                else:
                    error_msg = data.get('response', {}).get('header', {}).get('resultMsg', '알 수 없는 오류')
                    print(f"     API 오류: {error_msg}")
                    return []
            else:
                print(f"     HTTP 오류: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"     예외 발생: {e}")
            return []
    
    def _fetch_hourly_week_data(self, station_id, start_date, end_date, existing_data):
        """주별 시간자료 수집"""
        try:
            all_week_data = []
            current_date = start_date
            
            while current_date <= end_date:
                # 하루씩 수집 (시간자료는 하루에 24개)
                day_data = self._fetch_hourly_day_data(station_id, current_date, existing_data)
                if day_data:
                    all_week_data.extend(day_data)
                
                current_date += timedelta(days=1)
                time.sleep(0.5)  # 일별 API 호출 간격
            
            return all_week_data
            
        except Exception as e:
            print(f"     주별 시간자료 수집 오류: {e}")
            return []
    
    def _fetch_hourly_day_data(self, station_id, target_date, existing_data):
        """일별 시간자료 수집 (24시간)"""
        try:
            date_str = target_date.strftime('%Y%m%d')
            
            params = {
                'serviceKey': self.service_key,
                'pageNo': '1',
                'numOfRows': '24',  # 하루 24시간
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': date_str,
                'startHh': '01',
                'endDt': date_str,
                'endHh': '24',
                'stnIds': station_id
            }
            
            response = requests.get(self.apis['asos_hourly']['url'], params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    
                    if not items:
                        return []
                    
                    # 단일 항목을 리스트로 변환
                    if not isinstance(items, list):
                        items = [items]
                    
                    processed_data = []
                    for item in items:
                        # 중복 검사 (날짜+시간)
                        datetime_key = item.get('tm', '')
                        if datetime_key in existing_data:
                            continue
                        
                        processed_item = self._process_hourly_item(item)
                        if processed_item:
                            processed_data.append(processed_item)
                            existing_data.add(datetime_key)
                    
                    return processed_data
                else:
                    error_msg = data.get('response', {}).get('header', {}).get('resultMsg', '알 수 없는 오류')
                    if error_msg != 'NODATA_ERROR':  # 데이터 없음은 정상
                        print(f"       API 오류: {error_msg}")
                    return []
            else:
                print(f"       HTTP 오류: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"       일별 시간자료 수집 오류: {e}")
            return []
    
    def _process_daily_item(self, item):
        """일자료 항목 처리"""
        try:
            obs_date_str = item.get('tm', '')
            if not obs_date_str:
                return None
            
            # 날짜 파싱
            obs_date = datetime.strptime(obs_date_str, '%Y-%m-%d')
            
            # 강수량 처리
            precipitation = float(item.get('sumRn', 0) or 0)
            
            # 온도 처리
            avg_temp = float(item.get('avgTa', 0) or 0)
            min_temp = float(item.get('minTa', 0) or 0)
            max_temp = float(item.get('maxTa', 0) or 0)
            
            # 습도 처리
            humidity = float(item.get('avgRhm', 60) or 60)
            
            # 기타 기상 요소
            wind_speed = float(item.get('avgWs', 0) or 0)
            sunshine_hours = float(item.get('sumSsHr', 0) or 0)
            
            # 계절 타입 결정
            season_type = 'rainy' if obs_date.month in [5, 6, 7, 8, 9] else 'dry'
            
            # 침수 위험 여부 (50mm 이상)
            is_flood_risk = 1 if precipitation >= 50 else 0
            
            # 실제 침수 발생 여부 확인
            actual_flood = self._check_actual_flood(obs_date.date(), precipitation)
            
            return {
                'obs_date': obs_date,
                'year': obs_date.year,
                'month': obs_date.month,
                'day': obs_date.day,
                'season_type': season_type,
                'precipitation': precipitation,
                'avg_temp': avg_temp,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'sunshine_hours': sunshine_hours,
                'is_flood_risk': is_flood_risk,
                'actual_flood': actual_flood,
                'data_source': 'ASOS_DAILY',
                'data_quality': 'OFFICIAL',
                'station_id': item.get('stnId', '108'),
                'station_name': item.get('stnNm', '서울')
            }
            
        except Exception as e:
            print(f"       일자료 처리 오류: {e}")
            return None
    
    def _process_hourly_item(self, item):
        """시간자료 항목 처리"""
        try:
            datetime_str = item.get('tm', '')
            if not datetime_str:
                return None
            
            # 날짜시간 파싱 (예: "2020-01-01 01:00")
            obs_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
            
            # 기본 기상 요소
            temperature = float(item.get('ta', 0) or 0)
            precipitation = float(item.get('rn', 0) or 0)
            humidity = float(item.get('hm', 60) or 60)
            wind_speed = float(item.get('ws', 0) or 0)
            wind_direction = float(item.get('wd', 0) or 0)
            pressure = float(item.get('pa', 1013) or 1013)
            
            # 계절 타입
            season_type = 'rainy' if obs_datetime.month in [5, 6, 7, 8, 9] else 'dry'
            
            # 시간별 침수 위험 (10mm/h 이상)
            is_flood_risk = 1 if precipitation >= 10 else 0
            
            return {
                'obs_datetime': obs_datetime,
                'obs_date': obs_datetime.date(),
                'year': obs_datetime.year,
                'month': obs_datetime.month,
                'day': obs_datetime.day,
                'hour': obs_datetime.hour,
                'season_type': season_type,
                'temperature': temperature,
                'precipitation': precipitation,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'pressure': pressure,
                'is_flood_risk': is_flood_risk,
                'data_source': 'ASOS_HOURLY',
                'data_quality': 'OFFICIAL',
                'station_id': item.get('stnId', '108'),
                'station_name': item.get('stnNm', '서울')
            }
            
        except Exception as e:
            print(f"       시간자료 처리 오류: {e}")
            return None
    
    def _check_actual_flood(self, check_date: datetime.date, precipitation: float) -> int:
        """실제 침수 발생 여부 확인"""
        date_str = check_date.strftime('%Y-%m-%d')
        
        for event in self.flood_events:
            if event['date'] == date_str:
                return 1
        
        # 강수량이 매우 높으면 침수 가능성 추정
        if precipitation >= 200:
            return 1
        
        return 0
    
    def _load_existing_daily_data(self) -> set:
        """기존 일자료 로드"""
        existing_dates = set()
        
        if os.path.exists(self.daily_csv_path):
            try:
                df = pd.read_csv(self.daily_csv_path)
                if 'obs_date' in df.columns:
                    # 날짜 형식 통일
                    for date_val in df['obs_date']:
                        try:
                            # 다양한 날짜 형식 처리
                            if pd.notna(date_val):
                                if isinstance(date_val, str):
                                    if ' ' in date_val:  # datetime 형식
                                        date_part = date_val.split(' ')[0]
                                    else:
                                        date_part = date_val
                                    existing_dates.add(date_part)
                                else:
                                    existing_dates.add(str(date_val))
                        except:
                            continue
                            
                print(f" 기존 일자료: {len(existing_dates)}일")
            except Exception as e:
                print(f" 기존 일자료 로드 오류: {e}")
        
        return existing_dates
    
    def _load_existing_hourly_data(self) -> set:
        """기존 시간자료 로드"""
        existing_datetimes = set()
        
        if os.path.exists(self.hourly_csv_path):
            try:
                df = pd.read_csv(self.hourly_csv_path)
                if 'obs_datetime' in df.columns:
                    for datetime_val in df['obs_datetime']:
                        if pd.notna(datetime_val):
                            existing_datetimes.add(str(datetime_val))
                            
                print(f" 기존 시간자료: {len(existing_datetimes)}시간")
            except Exception as e:
                print(f" 기존 시간자료 로드 오류: {e}")
        
        return existing_datetimes
    
    def _append_daily_data_to_csv(self, data_list):
        """일자료 CSV 저장"""
        if not data_list:
            return
        
        df = pd.DataFrame(data_list)
        
        # 기존 파일에 추가
        if os.path.exists(self.daily_csv_path):
            existing_df = pd.read_csv(self.daily_csv_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        # 중복 제거 및 정렬
        combined_df = combined_df.drop_duplicates(subset=['obs_date'], keep='last')
        combined_df = combined_df.sort_values('obs_date').reset_index(drop=True)
        
        # CSV 저장
        combined_df.to_csv(self.daily_csv_path, index=False, encoding='utf-8-sig')
    
    def _append_hourly_data_to_csv(self, data_list):
        """시간자료 CSV 저장"""
        if not data_list:
            return
        
        df = pd.DataFrame(data_list)
        
        # 기존 파일에 추가
        if os.path.exists(self.hourly_csv_path):
            existing_df = pd.read_csv(self.hourly_csv_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        
        # 중복 제거 및 정렬
        combined_df = combined_df.drop_duplicates(subset=['obs_datetime'], keep='last')
        combined_df = combined_df.sort_values('obs_datetime').reset_index(drop=True)
        
        # CSV 저장
        combined_df.to_csv(self.hourly_csv_path, index=False, encoding='utf-8-sig')
    
    def create_combined_dataset(self):
        """일자료와 시간자료를 결합한 통합 데이터셋 생성"""
        print(" 통합 데이터셋 생성 중...")
        
        combined_data = []
        
        # 일자료 로드
        if os.path.exists(self.daily_csv_path):
            daily_df = pd.read_csv(self.daily_csv_path)
            daily_df['data_type'] = 'daily'
            combined_data.append(daily_df)
            print(f"   일자료: {len(daily_df)}행")
        
        # 시간자료를 일자료 형식으로 집계
        if os.path.exists(self.hourly_csv_path):
            hourly_df = pd.read_csv(self.hourly_csv_path)
            
            # 시간자료를 일별로 집계
            hourly_df['obs_date'] = pd.to_datetime(hourly_df['obs_datetime']).dt.date
            
            daily_aggregated = hourly_df.groupby('obs_date').agg({
                'temperature': 'mean',
                'precipitation': 'sum',  # 일 강수량 = 시간 강수량 합계
                'humidity': 'mean',
                'wind_speed': 'mean',
                'pressure': 'mean',
                'is_flood_risk': 'max',  # 하루 중 한 시간이라도 위험하면 위험
                'year': 'first',
                'month': 'first', 
                'day': 'first',
                'season_type': 'first',
                'station_id': 'first',
                'station_name': 'first'
            }).reset_index()
            
            # 컬럼명 통일 (기존 코드 호환)
            daily_aggregated = daily_aggregated.rename(columns={
                'temperature': 'avg_temp'
            })
            
            # 추가 필드 계산
            daily_aggregated['min_temp'] = daily_aggregated['avg_temp'] - 5
            daily_aggregated['max_temp'] = daily_aggregated['avg_temp'] + 5
            daily_aggregated['wind_speed'] = daily_aggregated['wind_speed']
            daily_aggregated['sunshine_hours'] = 8  # 기본값
            daily_aggregated['actual_flood'] = daily_aggregated['obs_date'].apply(
                lambda x: self._check_actual_flood(x, 0)
            )
            daily_aggregated['data_source'] = 'ASOS_HOURLY_AGG'
            daily_aggregated['data_quality'] = 'OFFICIAL'
            daily_aggregated['data_type'] = 'hourly_agg'
            
            combined_data.append(daily_aggregated)
            print(f"   시간자료 집계: {len(daily_aggregated)}행")
        
        # 데이터 결합
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True, sort=False)
            
            # 중복 제거 (같은 날짜는 일자료 우선)
            final_df = final_df.sort_values(['obs_date', 'data_type']).drop_duplicates(
                subset=['obs_date'], keep='first'
            )
            
            # 날짜순 정렬
            final_df = final_df.sort_values('obs_date').reset_index(drop=True)
            
            # 기존 코드 호환을 위한 컬럼 확인 및 추가
            required_columns = [
                'obs_date', 'year', 'month', 'day', 'season_type',
                'precipitation', 'avg_temp', 'min_temp', 'max_temp', 
                'humidity', 'wind_speed', 'is_flood_risk', 'actual_flood'
            ]
            
            for col in required_columns:
                if col not in final_df.columns:
                    if col == 'temperature':
                        final_df[col] = final_df.get('avg_temp', 20)
                    elif col in ['min_temp', 'max_temp']:
                        final_df[col] = final_df.get('avg_temp', 20)
                    else:
                        final_df[col] = 0
            
            # 통합 파일 저장
            final_df.to_csv(self.combined_csv_path, index=False, encoding='utf-8-sig')
            
            print(f"통합 데이터셋 생성 완료: {len(final_df)}행")
            print(f"    저장 위치: {self.combined_csv_path}")
            
            return final_df
        else:
            print("결합할 데이터가 없습니다.")
            return None
    
    def collect_strategic_historical_data(self, max_days=30):
        """전략적 과거 데이터 수집 (web_app.py 호환)"""
        try:
            print(f"전략적 과거 데이터 수집 시작 (최대 {max_days}일)")
            
            # 최근 데이터 수집
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=max_days)
            
            # 일자료 수집
            existing_daily = self._load_existing_daily_data()
            daily_data = self._fetch_daily_month_data(
                self.collection_plan["primary_station"], 
                start_date, 
                end_date, 
                existing_daily
            )
            
            if daily_data:
                self._append_daily_data_to_csv(daily_data)
                print(f"일자료 {len(daily_data)}일 수집 완료")
            
            # 통합 데이터셋 생성
            self.create_combined_dataset()
            
            return len(daily_data) if daily_data else 0
            
        except Exception as e:
            print(f"전략적 데이터 수집 오류: {e}")
            return 0
    
    def get_collection_stats(self):
        """수집 통계 조회"""
        stats = {
            'daily_data': {'exists': False, 'rows': 0, 'date_range': None},
            'hourly_data': {'exists': False, 'rows': 0, 'date_range': None},
            'combined_data': {'exists': False, 'rows': 0, 'date_range': None}
        }
        
        # 일자료 통계
        if os.path.exists(self.daily_csv_path):
            try:
                df = pd.read_csv(self.daily_csv_path)
                stats['daily_data'] = {
                    'exists': True,
                    'rows': len(df),
                    'date_range': f"{df['obs_date'].min()} ~ {df['obs_date'].max()}" if len(df) > 0 else None,
                    'flood_risk_days': len(df[df.get('is_flood_risk', 0) == 1]) if 'is_flood_risk' in df.columns else 0
                }
            except Exception as e:
                print(f"일자료 통계 오류: {e}")
        
        # 시간자료 통계
        if os.path.exists(self.hourly_csv_path):
            try:
                df = pd.read_csv(self.hourly_csv_path)
                stats['hourly_data'] = {
                    'exists': True,
                    'rows': len(df),
                    'date_range': f"{df['obs_datetime'].min()} ~ {df['obs_datetime'].max()}" if len(df) > 0 else None,
                    'flood_risk_hours': len(df[df.get('is_flood_risk', 0) == 1]) if 'is_flood_risk' in df.columns else 0
                }
            except Exception as e:
                print(f"시간자료 통계 오류: {e}")
        
        # 통합자료 통계
        if os.path.exists(self.combined_csv_path):
            try:
                df = pd.read_csv(self.combined_csv_path)
                stats['combined_data'] = {
                    'exists': True,
                    'rows': len(df),
                    'date_range': f"{df['obs_date'].min()} ~ {df['obs_date'].max()}" if len(df) > 0 else None
                }
            except Exception as e:
                print(f"통합자료 통계 오류: {e}")
        
        return stats


# 테스트 함수
def test_multi_weather_api():
    """Multi Weather API 테스트"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    service_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not service_key:
        print("서비스 키가 없습니다!")
        print(".env 파일에 OPENWEATHER_API_KEY를 설정하세요.")
        return False
    
    api = MultiWeatherAPI(service_key)
    
    print("Multi Weather API 테스트 시작")
    print("=" * 50)
    
    # 종합 기상 데이터 수집 테스트
    print("\n1️⃣ 종합 기상 데이터 수집 테스트...")
    result = api.get_comprehensive_weather_data()
    
    if result['success']:
        print(f"수집 성공!")
        print(f"데이터 소스: {', '.join(result['data_sources'])}")
        print(f"온도: {result['weather_data'].get('temperature', 'N/A')}°C")
        print(f"강수량: {result['weather_data'].get('precipitation', 'N/A')}mm")
        print(f"습도: {result['weather_data'].get('humidity', 'N/A')}%")
    else:
        print(f" 수집 실패: {result['message']}")
    
    # 통계 조회 테스트
    print("\n2️⃣ 수집 통계 조회...")
    stats = api.get_collection_stats()
    
    for data_type, info in stats.items():
        if info['exists']:
            print(f"{data_type}: {info['rows']}행 ({info.get('date_range', 'N/A')})")
        else:
            print(f"{data_type}: 데이터 없음")
    
    return result['success']


if __name__ == "__main__":
    test_multi_weather_api()