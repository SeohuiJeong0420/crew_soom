python3 -c "
import sqlite3
import pandas as pd
import os

# 1. SQLite 확인
db_path = 'data/processed/seoul_flood_prediction.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM strategic_daily ORDER BY obs_date', conn)
    conn.close()
    
    if not df.empty:
        # 2. CSV로 변환
        df['temperature'] = df['avg_temp']
        df['data_source'] = 'CONVERTED_FROM_SQLITE'
        df.to_csv('data/processed/REAL_WEATHER_DATA.csv', index=False)
        print(f'✅ {len(df)}일 데이터를 SQLite → CSV 변환 완료!')
    else:
        print('❌ SQLite 비어있음')
else:
    print('❌ SQLite 파일 없음')
"