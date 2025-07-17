# modules/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class DataVisualizer:
    """데이터 시각화"""
    
    def __init__(self):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        os.makedirs('outputs', exist_ok=True)
        
    def create_all_visualizations(self, df, model_results):
        """모든 시각화 생성"""
        
        print(" 시각화 생성 중...")
        
        # 1. 시계열 분석
        self.plot_time_series(df)
        
        # 2. 계절별 패턴
        self.plot_seasonal_patterns(df)
        
        # 3. 상관관계 분석
        self.plot_correlation_matrix(df)
        
        # 4. 모델 성능 요약
        self.plot_model_summary(model_results)
        
        print("모든 시각화 완료")
        print("저장 위치: outputs/ 폴더")
    
    def plot_time_series(self, df):
        """시계열 분석"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('기상 데이터 시계열 분석', fontsize=16)
        
        # 강수량
        axes[0].plot(df['obs_date'], df['precipitation'], alpha=0.7)
        if 'is_flood_risk' in df.columns:
            flood_dates = df[df['is_flood_risk'] == 1]['obs_date']
            flood_precip = df[df['is_flood_risk'] == 1]['precipitation']
            axes[0].scatter(flood_dates, flood_precip, color='red', s=30, alpha=0.8)
        
        axes[0].set_title('일별 강수량')
        axes[0].set_ylabel('강수량 (mm)')
        axes[0].grid(True, alpha=0.3)
        
        # 온도
        if 'avg_temp' in df.columns:
            axes[1].plot(df['obs_date'], df['avg_temp'], color='orange')
            axes[1].set_title('평균 온도')
            axes[1].set_ylabel('온도 (°C)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_patterns(self, df):
        """계절별 패턴"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('계절별 패턴 분석', fontsize=16)
        
        # 월별 강수량
        if 'month' in df.columns:
            monthly_precip = df.groupby('month')['precipitation'].mean()
            axes[0,0].bar(monthly_precip.index, monthly_precip.values, alpha=0.8)
            axes[0,0].set_title('월별 평균 강수량')
            axes[0,0].set_xlabel('월')
            axes[0,0].set_ylabel('평균 강수량 (mm)')
        
        # 강수량 분포
        axes[0,1].hist(df['precipitation'], bins=50, alpha=0.7, color='skyblue')
        axes[0,1].axvline(x=50, color='red', linestyle='--', label='50mm 위험선')
        axes[0,1].set_title('강수량 분포')
        axes[0,1].legend()
        
        # 온도 vs 강수량
        if 'avg_temp' in df.columns and 'humidity' in df.columns:
            scatter = axes[1,0].scatter(df['avg_temp'], df['precipitation'], 
                                        c=df['humidity'], alpha=0.6, cmap='viridis')
            axes[1,0].set_title('온도 vs 강수량')
            axes[1,0].set_xlabel('온도 (°C)')
            axes[1,0].set_ylabel('강수량 (mm)')
            plt.colorbar(scatter, ax=axes[1,0])
        
        # 위험도 분포
        if 'is_flood_risk' in df.columns:
            risk_counts = df['is_flood_risk'].value_counts()
            axes[1,1].pie(risk_counts.values, labels=['안전', '위험'], autopct='%1.1f%%')
            axes[1,1].set_title('침수 위험도 분포')
        
        plt.tight_layout()
        plt.savefig('outputs/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df):
        """상관관계 매트릭스"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col in df.columns][:10]  # 상위 10개만
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('변수간 상관관계')
            plt.tight_layout()
            plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_model_summary(self, results):
        """모델 성능 요약"""
        if not results:
            return
            
        results_df = pd.DataFrame(results).T
        
        plt.figure(figsize=(10, 6))
        results_df[['AUC', 'Precision', 'Recall', 'F1']].plot(kind='bar', alpha=0.8)
        plt.title('모델 성능 비교')
        plt.ylabel('점수')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()