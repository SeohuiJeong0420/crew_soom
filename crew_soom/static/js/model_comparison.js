// static/js/model_comparison.js - 모델 성능 비교 시스템

/* ==========================================
   모델 성능 비교 시스템 - 새로 추가됨
   ========================================== */

// 모델 성능 비교 글로벌 변수
let currentSlide = 0;
let totalSlides = 0;
let currentModelData = null;

// 모델 성능 데이터
const modelPerformanceDataStatic = {
    'randomforest': {
        name: 'Random Forest',
        metrics: {
            'F1 Score': { value: 0.920, description: '우수한 수준의 정밀도와 재현율 균형' },
            'Accuracy': { value: 0.960, description: '전체 예측 정확도가 매우 높음' },
            'Recall': { value: 0.850, description: '실제 침수 상황을 잘 탐지함' },
            'ROC AUC': { value: 0.970, description: '모델의 분류 성능이 훌륭함' },
            'Precision': { value: 0.960, description: '침수 예측의 정확성이 높음' }
        },
        details: `Random Forest 모델 훈련 결과:

모델 성능:
  - 전체 정확도: 96.0%
  - F1 Score: 0.920
  - 예측 정밀도: 96.0%
  - 재현율: 85.0%
  - ROC AUC: 0.970

모델 특징:
  - 100개의 의사결정트리 사용
  - 랜덤 샘플링으로 과적합 방지
  - 9개 기상 변수와 파생 변수 활용
  - 안정적이고 해석 가능한 모델

학습 데이터:
  - 총 25,420개 데이터 포인트
  - 2022년 실제 침수 사례 포함
  - 기상청 공식 데이터 기반`
    },
    'xgboost': {
        name: 'XGBoost',
        metrics: {
            'F1 Score': { value: 0.845, description: '균형 잡힌 성능으로 안정적 예측' },
            'Accuracy': { value: 0.816, description: '우수한 전체 예측 정확도' },
            'Recall': { value: 0.876, description: '높은 재현율로 침수 대부분 탐지' },
            'ROC AUC': { value: 0.977, description: '가장 높은 분류 성능' },
            'Precision': { value: 0.816, description: '침수 예측에서 오탐 최소화' }
        },
        details: `XGBoost 모델 훈련 결과:

모델 성능:
  - 전체 정확도: 81.6%
  - F1 Score: 0.845
  - 예측 정밀도: 81.6%
  - 재현율: 87.6%
  - ROC AUC: 0.977 (최고점수)

모델 특징:
  - Gradient Boosting 알고리즘 사용
  - 16개 파생 변수로 확장된 특성
  - 지역별 침수 취약성 고려
  - 시간별 기상 패턴 반영

훈련 세부사항:
  - Learning Rate: 0.1
  - Max Depth: 6
  - Estimators: 200
  - StandardScaler로 전처리
  - 5-Fold Cross Validation 수행`
    },
    'lstm-cnn': {
        name: 'LSTM+CNN',
        metrics: {
            'F1 Score': { value: 0.320, description: '시계열 패턴 학습 중이나 낮은 성능' },
            'Accuracy': { value: 0.196, description: '복잡한 모델로 추가 훈련 필요' },
            'Recall': { value: 0.000, description: '실제 침수 상황 탐지 어려움' },
            'ROC AUC': { value: 0.769, description: '중간 수준의 분류 성능' },
            'Precision': { value: 0.196, description: '침수 예측에서 높은 오탐률' }
        },
        details: `LSTM+CNN 모델 훈련 결과:

모델 성능:
  - 전체 정확도: 19.6%
  - F1 Score: 0.320
  - 예측 정밀도: 19.6%
  - 재현율: 0.0%
  - ROC AUC: 0.769

모델 구조:
  - LSTM 레이어: 50 units
  - CNN 필터: 32개, 커널 크기 3
  - 7일 시계열 입력 시퀀스
  - Dropout 0.3 적용

문제점 및 개선방향:
  - 낮은 재현율: 데이터 불균형 문제
  - 복잡한 모델: 더 많은 데이터 필요
  - 추가 훈련과 하이퍼파라미터 튜닝 필요`
    },
    'transformer': {
        name: 'Transformer',
        metrics: {
            'F1 Score': { value: 0.440, description: '어텐션 메커니즘으로 중간 성능' },
            'Accuracy': { value: 0.290, description: '복잡한 모델로 추가 캘리브레이션 필요' },
            'Recall': { value: 0.910, description: '매우 높은 재현율로 침수 대부분 탐지' },
            'ROC AUC': { value: 0.874, description: '양호한 분류 성능' },
            'Precision': { value: 0.290, description: '높은 오탐률로 정밀도 개선 필요' }
        },
        details: `Transformer 모델 훈련 결과:

모델 성능:
  - 전체 정확도: 29.0%
  - F1 Score: 0.440
  - 예측 정밀도: 29.0%
  - 재현율: 91.0% (최고점수)
  - ROC AUC: 0.874

모델 구조:
  - Multi-Head Attention: 2 heads
  - Feed Forward Dim: 16
  - 7일 시계열 입력
  - GlobalAveragePooling1D 사용

특징 및 개선방향:
  - 매우 높은 재현율: 침수 놓치지 않음
  - 낮은 정밀도: 오탐 많이 발생
  - 어텐션 메커니즘 효과적
  - 더 많은 데이터와 정교한 훈련 필요`
    }
};

// 모델 성능 비교 표시
function showModelComparison(modelType) {
    const performanceSlides = document.getElementById('performance-slides');
    const modelDetails = document.getElementById('model-details');
    
    if (modelType === 'comparison') {
        showAllModelsComparison();
        return;
    }
    
    if (modelPerformanceDataStatic[modelType]) {
        currentModelData = modelPerformanceDataStatic[modelType];
        createPerformanceSlides(currentModelData);
        showModelTextDetails(currentModelData);
        
        if (performanceSlides) performanceSlides.style.display = 'block';
        if (modelDetails) modelDetails.style.display = 'block';
        
        // 슬라이드 초기화
        currentSlide = 0;
        showSlide(0);
        
        // 버튼 업데이트
        updateModelComparisonButtons(modelType);
        
        if (typeof showNotification === 'function') {
            showNotification(` ${currentModelData.name} 모델의 성능 지표를 표시합니다.`, 'info', 3000);
        }
    }
}

// 전체 모델 비교 차트
function showAllModelsComparison() {
    if (typeof createModelVisualization === 'function') {
        createModelVisualization();
    } else {
        // API 직접 호출
        showGlobalLoading('4개 모델 성능 비교 차트를 생성하고 있습니다...');
        
        fetch('/api/create_model_comparison', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const vizArea = document.getElementById('visualization-area');
                    if (vizArea) {
                        vizArea.innerHTML = `
                            <div class="viz-result" style="width: 100%;">
                                <img src="${data.image}" class="viz-image" alt="모델 성능 비교 차트" 
                                     style="width: 100%; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 16px; cursor: pointer;" 
                                     onclick="openImageModal('${data.image}')">
                                <div class="viz-info" style="background: linear-gradient(135deg, #2c5ff7, #4a90e2); color: white; padding: 20px; border-radius: 12px; font-size: 14px;">
                                    <h4 style="margin-bottom: 12px; color: white;"> AI 모델 성능 분석 결과</h4>
                                    <p><strong>최고 모델:</strong> ${data.best_model || 'N/A'}</p>
                                    <p><strong>평균 정확도:</strong> ${data.avg_accuracy || 'N/A'}</p>
                                    <p><strong>분석 모델:</strong> ${data.models_count || 4}개</p>
                                    <p><strong>활용 데이터:</strong> ${data.data_used || 'N/A'}</p>
                                </div>
                            </div>
                        `;
                    }
                    
                    if (typeof showNotification === 'function') {
                        showNotification(
                            `AI 모델 성능 비교 분석이 완료되었습니다.\n최고 성능: ${data.best_model}\n 평균 정확도: ${data.avg_accuracy}`, 
                            'success', 
                            6000
                        );
                    }
                } else {
                    throw new Error(data.error || '모델 비교 실패');
                }
            })
            .catch(error => {
                console.error('모델 비교 오류:', error);
                if (typeof showNotification === 'function') {
                    showNotification('모델 비교 오류: ' + error.message, 'error');
                }
            })
            .finally(() => {
                if (typeof hideGlobalLoading === 'function') {
                    hideGlobalLoading();
                }
            });
    }
    
    // 버튼 업데이트
    updateModelComparisonButtons('comparison');
}

// 성능 슬라이드 생성
function createPerformanceSlides(modelData) {
    const container = document.getElementById('slides-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    const metrics = Object.keys(modelData.metrics);
    totalSlides = metrics.length;
    
    metrics.forEach((metric, index) => {
        const slide = document.createElement('div');
        slide.className = 'performance-slide';
        if (index === 0) slide.classList.add('active');
        
        const metricData = modelData.metrics[metric];
        const percentage = (metricData.value * 100).toFixed(1);
        
        slide.innerHTML = `
            <div class="metric-display">
                <div class="metric-title">${metric}</div>
                <div class="metric-value" style="color: ${getMetricColor(metricData.value)}">${percentage}%</div>
                <div class="metric-description">${metricData.description}</div>
                <div style="margin-top: 20px; font-size: 12px; color: #666;">
                    ${modelData.name} 모델의 ${metric} 성능 지표
                </div>
            </div>
        `;
        
        container.appendChild(slide);
    });
    
    updateSlideIndicator();
}

// 성능에 따른 색상 결정
function getMetricColor(value) {
    if (value >= 0.9) return '#00c851'; // 녹색
    if (value >= 0.7) return '#ffbb33'; // 노란색
    if (value >= 0.5) return '#ff8a00'; // 주황색
    return '#ff4444'; // 빨간색
}

// 모델 텍스트 상세 정보 표시
function showModelTextDetails(modelData) {
    const textContainer = document.getElementById('model-text-results');
    if (textContainer) {
        textContainer.innerHTML = `<pre>${modelData.details}</pre>`;
    }
}

// 슬라이드 네비게이션
function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        showSlide(currentSlide + 1);
    } else {
        showSlide(0); // 마지막에서 처음으로
    }
}

function previousSlide() {
    if (currentSlide > 0) {
        showSlide(currentSlide - 1);
    } else {
        showSlide(totalSlides - 1); // 처음에서 마지막으로
    }
}

function showSlide(index) {
    const slides = document.querySelectorAll('.performance-slide');
    
    slides.forEach((slide, i) => {
        slide.classList.remove('active', 'prev');
        if (i === index) {
            slide.classList.add('active');
        } else if (i < index) {
            slide.classList.add('prev');
        }
    });
    
    currentSlide = index;
    updateSlideIndicator();
}

function updateSlideIndicator() {
    const indicator = document.getElementById('slide-indicator');
    if (indicator && totalSlides > 0) {
        indicator.textContent = `${currentSlide + 1} / ${totalSlides}`;
    }
}

// 모델 비교 버튼 상태 업데이트
function updateModelComparisonButtons(activeModel) {
    const buttons = document.querySelectorAll('[onclick^="showModelComparison"]');
    
    buttons.forEach(button => {
        button.classList.remove('btn-primary');
        button.classList.add('btn-outline');
        
        // 활성 버튼 찾기
        const onclick = button.getAttribute('onclick');
        if (onclick) {
            const modelMatch = onclick.match(/showModelComparison\('([^']+)'\)/);
            if (modelMatch && modelMatch[1] === activeModel) {
                button.classList.remove('btn-outline');
                button.classList.add('btn-primary');
            }
        }
    });
}

// 키보드 이벤트로 슬라이드 조작
document.addEventListener('keydown', function(e) {
    const performanceSlides = document.getElementById('performance-slides');
    if (performanceSlides && performanceSlides.style.display === 'block') {
        if (e.key === 'ArrowLeft') {
            previousSlide();
            e.preventDefault();
        } else if (e.key === 'ArrowRight') {
            nextSlide();
            e.preventDefault();
        } else if (e.key === 'Escape') {
            performanceSlides.style.display = 'none';
            document.getElementById('model-details').style.display = 'none';
            e.preventDefault();
        }
    }
});

// 모델 비교 시스템을 위한 추가 유틸리티
function exportModelComparison() {
    if (!currentModelData) {
        if (typeof showNotification === 'function') {
            showNotification('먼저 모델을 선택해주세요.', 'warning');
        }
        return;
    }
    
    const exportData = {
        model: currentModelData.name,
        metrics: currentModelData.metrics,
        exportTime: new Date().toISOString(),
        source: 'CREW_SOOM Dashboard'
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `${currentModelData.name}_performance_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    if (typeof showNotification === 'function') {
        showNotification('모델 성능 데이터가 다운로드되었습니다.', 'success');
    }
}

// 모델 성능 요약 생성
function generateModelSummary() {
    if (!currentModelData) return '';
    
    const metrics = currentModelData.metrics;
    const avgScore = Object.values(metrics).reduce((sum, metric) => sum + metric.value, 0) / Object.keys(metrics).length;
    
    return `${currentModelData.name} 성능 요약:
평균 점수: ${(avgScore * 100).toFixed(1)}%
최고 성능: ${Object.entries(metrics).reduce((best, [key, value]) => value.value > best.value ? {key, value: value.value} : best, {key: '', value: 0}).key}
권장 사용: ${avgScore > 0.8 ? '운영 환경' : '개발/테스트 환경'}`;
}

// 성능 비교 초기화 확인
function initModelComparison() {
    const performanceSlides = document.getElementById('performance-slides');
    const modelDetails = document.getElementById('model-details');
    
    if (performanceSlides) {
        console.log('모델 성능 비교 시스템 초기화됨');
    } else {
        console.warn('모델 성능 비교 UI 요소를 찾을 수 없음');
    }
}

// 이미지 모달 기능 (모델 차트용)
function openImageModal(imageSrc) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageSrc;
    img.style.cssText = `
        max-width: 95%;
        max-height: 95%;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    `;
    
    modal.appendChild(img);
    document.body.appendChild(modal);
    
    modal.onclick = () => modal.remove();
    
    // ESC 키로 닫기
    const handleEsc = (e) => {
        if (e.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', handleEsc);
        }
    };
    document.addEventListener('keydown', handleEsc);
}

// 글로벌 함수들이 없는 경우를 위한 폴백
function showGlobalLoading(message) {
    console.log('Loading:', message);
}

function hideGlobalLoading() {
    console.log('Loading finished');
}

// 페이지 로드 시 모델 비교 시스템 초기화
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(initModelComparison, 1000);
});

console.log('모델 성능 비교 시스템 로드 완료!');
console.log('사용 가능한 모델: Random Forest, XGBoost, LSTM+CNN, Transformer');
console.log('기능: 슬라이드 뷰, 상세 텍스트, 키보드 조작 (←/→/ESC)');
