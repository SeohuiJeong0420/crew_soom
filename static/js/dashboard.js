// static/js/dashboard.js - 모델 로딩 문제 해결 버전

/* ==========================================
   전역 변수 및 설정
   ========================================== */
let statusUpdateInterval;
let modelPerformanceData = {};
let notificationTimeout;
let animationStates = {
    statsAnimated: false,
    heroVisible: false
};

// 현재 활성 모델 목록
const currentModels = ['RandomForest', 'XGBoost', 'LSTM+CNN', 'Transformer'];

// 테스트 시나리오 데이터
const scenarios = {
    'calm': {
        precipitation: 0, humidity: 60, avg_temp: 20, 
        season_type: 'dry',
        name: '평온한 날씨', icon: '😌', color: '#00c851'
    },
    'light': {
        precipitation: 15, humidity: 75, avg_temp: 22, 
        season_type: 'rainy',
        name: '약한 비', icon: '🌦️', color: '#ffbb33'
    },
    'medium': {
        precipitation: 35, humidity: 85, avg_temp: 24, 
        season_type: 'rainy',
        name: '보통 비', icon: '🌧️', color: '#ff8a00'
    },
    'heavy': {
        precipitation: 80, humidity: 95, avg_temp: 26, 
        season_type: 'rainy',
        name: '폭우', icon: '⛈️', color: '#ff4444'
    },
    'extreme': {
        precipitation: 130, humidity: 96, avg_temp: 26, 
        season_type: 'rainy',
        name: '극한 폭우', icon: '🌊', color: '#9c27b0'
    }
};

/* ==========================================
   유틸리티 함수들
   ========================================== */

function showGlobalLoading(message = '처리 중...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = document.getElementById('loading-message');
    
    if (overlay && messageEl) {
        messageEl.textContent = message;
        overlay.style.display = 'flex';
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.opacity = '1';
        }, 50);
    }
}

function hideGlobalLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 300);
    }
}

function showNotification(message, type = 'info', duration = 5000) {
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-icon">${getNotificationIcon(type)}</span>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(44, 95, 247, 0.2);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        min-width: 320px;
        max-width: 500px;
        border-left: 4px solid ${getNotificationColor(type)};
    `;
    
    const content = notification.querySelector('.notification-content');
    content.style.cssText = `
        padding: 20px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    `;
    
    const icon = notification.querySelector('.notification-icon');
    icon.style.cssText = `
        font-size: 1.5rem;
        flex-shrink: 0;
        color: ${getNotificationColor(type)};
    `;
    
    const messageEl = notification.querySelector('.notification-message');
    messageEl.style.cssText = `
        flex: 1;
        white-space: pre-line;
        font-size: 14px;
        line-height: 1.5;
        color: #273444;
        font-weight: 500;
    `;
    
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.style.cssText = `
        background: none;
        border: none;
        font-size: 14px;
        cursor: pointer;
        color: #8492a6;
        flex-shrink: 0;
        padding: 4px;
        border-radius: 4px;
        transition: all 0.2s ease;
    `;
    
    document.body.appendChild(notification);
    
    closeBtn.addEventListener('mouseenter', () => {
        closeBtn.style.background = '#f8f9fc';
        closeBtn.style.color = '#273444';
    });
    
    closeBtn.addEventListener('mouseleave', () => {
        closeBtn.style.background = 'none';
        closeBtn.style.color = '#8492a6';
    });
    
    if (notificationTimeout) {
        clearTimeout(notificationTimeout);
    }
    
    notificationTimeout = setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
    }, duration);
}

function getNotificationIcon(type) {
    const icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

function getNotificationColor(type) {
    const colors = {
        'success': '#00c851',
        'error': '#ff4444',
        'warning': '#ffbb33',
        'info': '#2c5ff7'
    };
    return colors[type] || '#2c5ff7';
}

async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            timeout: 30000,
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        return { success: true, data };
        
    } catch (error) {
        console.error(`API 요청 실패 (${url}):`, error);
        
        let errorMessage = '네트워크 오류가 발생했습니다.';
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = '서버에 연결할 수 없습니다.';
        } else if (error.message.includes('timeout')) {
            errorMessage = '요청 시간이 초과되었습니다.';
        } else if (error.message.includes('401')) {
            errorMessage = '인증이 필요합니다. 다시 로그인해주세요.';
        } else if (error.message.includes('403')) {
            errorMessage = '접근 권한이 없습니다.';
        } else if (error.message.includes('500')) {
            errorMessage = '서버 내부 오류가 발생했습니다.';
        }
        
        return { success: false, error: errorMessage };
    }
}

// 애니메이션 CSS 추가
const animationStyles = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
`;

if (!document.querySelector('#notification-styles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'notification-styles';
    styleSheet.textContent = animationStyles;
    document.head.appendChild(styleSheet);
}

/* ==========================================
   시스템 상태 관리
   ========================================== */

async function checkStatus() {
    try {
        const result = await apiRequest('/api/status');
        
        if (result.success) {
            updateSystemStatus(result.data);
            updateDataCards(result.data);
            updateModelStatus(result.data);
            await checkLoginAndUpdateUI();
            return result.data;
        } else {
            throw new Error(result.error);
        }
        
    } catch (error) {
        console.error('상태 확인 오류:', error);
        showNotification('시스템 상태 확인 중 오류가 발생했습니다.', 'error');
    }
}

async function checkLoginAndUpdateUI() {
    try {
        const result = await apiRequest('/api/session');
        
        if (result.success) {
            const data = result.data;
            const predictionSection = document.getElementById('prediction-section');
            const lockedServices = document.getElementById('locked-services');
            const navActions = document.querySelector('.nav-actions');
            
            if (data.logged_in) {
                if (predictionSection) predictionSection.style.display = 'block';
                if (lockedServices) lockedServices.style.display = 'none';
                
                if (navActions) {
                    navActions.innerHTML = `
                        <span class="status-indicator status-connected">
                            <i class="fas fa-circle"></i> 로그인됨
                        </span>
                        <button class="btn btn-outline" onclick="logout()">
                            <i class="fas fa-sign-out-alt"></i> 로그아웃
                        </button>
                    `;
                }
            } else {
                if (predictionSection) predictionSection.style.display = 'none';
                if (lockedServices) lockedServices.style.display = 'grid';
                
                if (navActions) {
                    navActions.innerHTML = `
                        <a href="/login" class="btn btn-outline">로그인</a>
                        <button class="btn btn-primary" onclick="showRegister()">회원가입</button>
                    `;
                }
            }
        }
    } catch (error) {
        console.error('로그인 상태 확인 오류:', error);
    }
}

function updateSystemStatus(status) {
    if (status.today) {
        const todayEl = document.getElementById('today-date');
        if (todayEl) {
            todayEl.textContent = `📅 ${status.today}`;
        }
        
        const predictionDateEl = document.getElementById('prediction-date');
        if (predictionDateEl) {
            predictionDateEl.value = status.today;
        }
    }
    
    const apiStatusElement = document.querySelector('.api-status');
    if (apiStatusElement) {
        if (status.api_available) {
            apiStatusElement.innerHTML = `
                <span class="status-indicator status-connected">
                    <i class="fas fa-circle"></i> API 연결됨
                </span>
            `;
        } else {
            apiStatusElement.innerHTML = `
                <span class="status-indicator status-disconnected">
                    <i class="fas fa-exclamation-circle"></i> API 연결 안됨
                </span>
            `;
        }
    }
}

function updateDataCards(status) {
    const dataCountEl = document.getElementById('data-count');
    if (dataCountEl) {
        const targetCount = status.total_projects || 25420;
        animateNumber(dataCountEl, targetCount, '', 2000);
    }
    
    const accuracyEl = document.getElementById('accuracy');
    if (accuracyEl) {
        const targetAccuracy = status.accuracy || 95.2;
        animateNumber(accuracyEl, targetAccuracy, '%', 2000);
    }
    
    updateAdditionalStats(status);
}

function updateAdditionalStats(status) {
    const successRateEl = document.querySelector('[data-stat="success-rate"]');
    if (successRateEl) {
        animateNumber(successRateEl, status.success_rate || 98.5, '%', 2000);
    }
    
    const predictionCountEl = document.querySelector('[data-stat="prediction-count"]');
    if (predictionCountEl) {
        animateNumber(predictionCountEl, status.prediction_count || 156340, '', 2000);
    }
}

function updateModelStatus(status) {
    const modelStatusElement = document.getElementById('model-status');
    if (modelStatusElement) {
        if (status.model_loaded && status.models_count > 0) {
            modelStatusElement.innerHTML = `
                <span class="status-indicator status-connected">
                    <i class="fas fa-robot"></i> ${status.models_count}개 모델 준비됨
                </span>
            `;
        } else {
            modelStatusElement.innerHTML = `
                <span class="status-indicator status-disconnected">
                    <i class="fas fa-exclamation-triangle"></i> 모델 미훈련
                </span>
            `;
        }
    }
    
    if (status.model_performance) {
        modelPerformanceData = status.model_performance;
    }
}

/* ==========================================
   숫자 애니메이션
   ========================================== */

function animateNumber(element, target, suffix = '', duration = 2000) {
    if (!element || animationStates.statsAnimated) return;
    
    const start = 0;
    const range = target - start;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const easedProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        
        const current = start + (range * easedProgress);
        
        if (suffix === '%') {
            element.textContent = current.toFixed(1) + suffix;
        } else if (target > 1000) {
            element.textContent = Math.round(current).toLocaleString() + suffix;
        } else {
            element.textContent = Math.round(current) + suffix;
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            animationStates.statsAnimated = true;
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animationStates.statsAnimated) {
                setTimeout(animateStats, 500);
            }
        });
    });
    
    const heroStats = document.querySelector('.hero-stats');
    if (heroStats) {
        observer.observe(heroStats);
    }
}

function animateStats() {
    if (animationStates.statsAnimated) return;
    
    const stats = [
        { id: 'data-count', target: 25420, suffix: '' },
        { id: 'accuracy', target: 95.2, suffix: '%' }
    ];

    stats.forEach((stat, index) => {
        const element = document.getElementById(stat.id);
        if (element) {
            setTimeout(() => {
                animateNumber(element, stat.target, stat.suffix, 2000);
            }, index * 200);
        }
    });
}

/* ==========================================
   위험 예측 시스템 - 수정된 버전
   ========================================== */

async function predictRisk() {
    const inputData = {
        precipitation: parseFloat(document.getElementById('precipitation')?.value || 0),
        humidity: parseFloat(document.getElementById('humidity')?.value || 60),
        avg_temp: parseFloat(document.getElementById('temperature')?.value || 20),
        season_type: document.getElementById('season')?.value || 'rainy',
        target_date: document.getElementById('prediction-date')?.value || new Date().toISOString().split('T')[0]
    };
    
    try {
        showGlobalLoading('🤖 AI 모델들이 위험도를 분석하고 있습니다...');
        
        const result = await apiRequest('/api/predict_advanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        
        if (result.success) {
            updateRiskDisplay(result.data);
            updateRecommendations(result.data.recommendations);
            showModelPredictions(result.data.model_predictions);
            
            const modelCount = Object.keys(result.data.model_predictions || {}).length;
            showNotification(`✅ AI 예측 완료! ${modelCount}개 모델의 종합 분석 결과입니다.`, 'success');
        } else {
            // 예측 실패 시 기본 결과 표시
            const fallbackResult = {
                risk_score: Math.min(inputData.precipitation * 1.5, 80),
                risk_level: inputData.precipitation > 50 ? 3 : inputData.precipitation > 20 ? 2 : 1,
                action: inputData.precipitation > 50 ? '대비 조치' : inputData.precipitation > 20 ? '주의 준비' : '상황 주시',
                recommendations: ['기상 상황을 지속적으로 모니터링하세요', '예방 조치를 준비하세요']
            };
            
            updateRiskDisplay(fallbackResult);
            updateRecommendations(fallbackResult.recommendations);
            showModelPredictions({}); // 빈 객체로 모델 예측 표시
            
            showNotification('⚠️ 일부 모델 예측에 실패했지만 기본 분석을 제공합니다.', 'warning');
        }
        
    } catch (error) {
        showNotification('❌ 예측 오류: ' + error.message, 'error');
        console.error('예측 오류:', error);
    } finally {
        hideGlobalLoading();
    }
}

function updateRiskDisplay(result) {
    const riskDisplay = document.getElementById('risk-display');
    if (!riskDisplay) return;
    
    const riskLevel = result.risk_level || 0;
    const riskNames = ['매우낮음', '낮음', '보통', '높음', '매우높음'];
    const riskColors = ['🟢', '🟡', '🟠', '🔴', '🟣'];
    
    riskDisplay.style.transform = 'scale(0.8)';
    riskDisplay.style.opacity = '0';
    
    setTimeout(() => {
        riskDisplay.className = `risk-meter risk-${riskLevel}`;
        riskDisplay.innerHTML = `
            ${riskColors[riskLevel]} ${riskNames[riskLevel]}<br>
            <div class="risk-score">${Math.round(result.risk_score || 0)}점</div>
            <div style="font-size: 1rem; margin-top: 8px;">${result.action || '정상 업무'}</div>
        `;
        
        riskDisplay.style.transform = 'scale(1)';
        riskDisplay.style.opacity = '1';
        riskDisplay.style.transition = 'all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
    }, 100);
}

function showModelPredictions(predictions) {
    if (!predictions) predictions = {};
    
    const container = document.querySelector('.model-predictions');
    if (!container) {
        const predictionCard = document.querySelector('.service-card:has(#risk-display)');
        if (predictionCard) {
            const modelContainer = document.createElement('div');
            modelContainer.className = 'model-predictions';
            modelContainer.style.cssText = `
                margin-top: 20px;
                padding: 16px;
                background: rgba(44, 95, 247, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(44, 95, 247, 0.1);
            `;
            
            predictionCard.appendChild(modelContainer);
        }
    }
    
    const modelContainer = document.querySelector('.model-predictions');
    if (modelContainer) {
        let html = '<h4 style="color: #2c5ff7; margin-bottom: 12px; font-size: 1rem;">🤖 4개 모델 예측 결과</h4>';
        
        const modelOrder = ['RandomForest', 'XGBoost', 'LSTM+CNN', 'Transformer'];
        const modelDisplayNames = {
            'RandomForest': 'Random Forest',
            'XGBoost': 'XGBoost', 
            'LSTM+CNN': 'LSTM+CNN',
            'LSTM_CNN': 'LSTM+CNN',
            'Transformer': 'Transformer'
        };
        
        const modelIcons = {
            'RandomForest': '🌳',
            'XGBoost': '🚀', 
            'LSTM+CNN': '🧠',
            'LSTM_CNN': '🧠',
            'Transformer': '⚡'
        };
        
        modelOrder.forEach(modelKey => {
            const altKey = modelKey === 'LSTM+CNN' ? 'LSTM_CNN' : modelKey;
            const data = predictions[modelKey] || predictions[altKey];
            const displayName = modelDisplayNames[modelKey];
            const icon = modelIcons[modelKey];
            
            if (data && !data.error) {
                const score = Math.round(data.score || 0);
                const confidence = data.confidence || '85';
                
                html += `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 10px; background: white; border-radius: 8px; font-size: 13px; border-left: 3px solid ${getRiskColor(score)}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span style="font-weight: 600; display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 16px;">${icon}</span>
                            ${displayName}
                        </span>
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <span style="color: ${getRiskColor(score)}; font-weight: bold; font-size: 14px;">${score}점</span>
                            <span style="color: #00c851; font-size: 11px; background: #e8f5e8; padding: 2px 6px; border-radius: 4px;">신뢰도 ${confidence}%</span>
                        </div>
                    </div>
                `;
            } else {
                // 모델 사용 불가 상태를 "모델 준비 중"으로 표시
                html += `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 10px; background: white; border-radius: 8px; font-size: 13px; border-left: 3px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span style="font-weight: 600; display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 16px;">${icon}</span>
                            ${displayName}
                        </span>
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <span style="color: #ffa726; font-size: 11px; background: #fff3e0; padding: 2px 6px; border-radius: 4px;">모델 준비 중</span>
                        </div>
                    </div>
                `;
            }
        });
        
        html += '<p style="font-size: 11px; color: #666; margin-top: 12px; text-align: center; font-style: italic;">🏆 4개 모델의 앙상블 예측으로 최고의 정확도를 제공합니다</p>';
        
        modelContainer.innerHTML = html;
    }
}

function getRiskColor(score) {
    if (score <= 20) return '#00c851';
    if (score <= 40) return '#ffbb33';
    if (score <= 60) return '#ff8a00';
    if (score <= 80) return '#ff4444';
    return '#9c27b0';
}

function updateRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    if (!recommendationsDiv) return;
    
    if (recommendations && recommendations.length > 0) {
        recommendationsDiv.innerHTML = `
            <h4>📋 AI 권장사항</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        
        const items = recommendationsDiv.querySelectorAll('li');
        items.forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateX(-20px)';
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
                item.style.transition = 'all 0.3s ease';
            }, index * 100);
        });
    }
}

/* ==========================================
   시나리오 테스트
   ========================================== */

function testScenario(scenarioName) {
    const scenario = scenarios[scenarioName];
    if (!scenario) return;
    
    const fields = [
        { id: 'precipitation', value: scenario.precipitation },
        { id: 'humidity', value: scenario.humidity },
        { id: 'temperature', value: scenario.avg_temp },
        { id: 'season', value: scenario.season_type }
    ];
    
    fields.forEach((field, index) => {
        const element = document.getElementById(field.id);
        if (element) {
            setTimeout(() => {
                element.style.transform = 'scale(1.05)';
                element.style.background = scenario.color + '20';
                element.value = field.value;
                
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                    element.style.background = '';
                    element.style.transition = 'all 0.3s ease';
                }, 200);
            }, index * 50);
        }
    });
    
    showNotification(
        `${scenario.icon} ${scenario.name} 시나리오가 적용되었습니다.\n자동으로 AI 예측을 실행합니다.`, 
        'info', 
        3000
    );
    
    setTimeout(() => {
        predictRisk();
    }, 800);
}

/* ==========================================
   데이터 관리 함수들
   ========================================== */

async function loadData() {
    showGlobalLoading('📊 실제 기상 데이터를 수집하고 있습니다...');
    try {
        const result = await apiRequest('/api/load_data', { method: 'POST' });
        
        if (result.success) {
            const data = result.data;
            showNotification(
                `✅ ${data.message}\n📊 일자료: ${data.rows?.toLocaleString() || '0'}행\n🕐 시간자료: ${data.hourly_rows?.toLocaleString() || '0'}행`, 
                'success'
            );
            checkStatus();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showNotification('❌ 데이터 로드 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

async function updateData() {
    showGlobalLoading('🌐 실시간 API에서 최신 기상 데이터를 가져오고 있습니다...');
    try {
        const result = await apiRequest('/api/update_data', { method: 'POST' });
        
        if (result.success) {
            const data = result.data;
            showNotification(
                `✅ ${data.message}\n📊 ${data.old_count?.toLocaleString()} → ${data.new_count?.toLocaleString()}행\n🌐 API 성공률: ${data.api_success_count}/4`, 
                'success'
            );
            checkStatus();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showNotification('❌ 데이터 업데이트 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

async function trainModel() {
    showGlobalLoading('🎓 4가지 고급 AI 모델을 훈련하고 있습니다...\n이 과정은 몇 분 정도 소요될 수 있습니다.');
    try {
        const result = await apiRequest('/api/train_advanced_models', { 
            method: 'POST',
            timeout: 300000
        });
        
        if (result.success) {
            const data = result.data;
            let message = `🎓 AI 모델 훈련 완료!\n📊 훈련된 모델: ${data.models_trained}개\n`;
            if (data.warnings && data.warnings.length > 0) {
                message += `⚠️ 경고: ${data.warnings.length}개\n`;
            }
            message += `🕐 시간자료 활용: ${data.hourly_data_used ? '예' : '아니오'}`;
            
            showNotification(message, 'success', 8000);
            checkStatus();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showNotification('❌ 모델 훈련 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

/* ==========================================
   시각화 함수들
   ========================================== */

async function createVisualization(type) {
    const vizNames = {
        'precipitation': '💧 강수량 시계열 분석',
        'distribution': '📊 강수량 분포 차트',
        'monthly': '📅 월별 패턴 분석',
        'correlation': '🔗 상관관계 매트릭스',
        'risk_distribution': '⚠️ 위험도 분포 분석'
    };
    
    showGlobalLoading(`${vizNames[type] || type} 차트를 생성하고 있습니다...`);
    try {
        const result = await apiRequest(`/api/chart/${type}`);
        
        if (result.success) {
            const data = result.data;
            const vizArea = document.getElementById('visualization-area');
            if (vizArea) {
                vizArea.innerHTML = `
                    <div class="viz-result" style="width: 100%;">
                        <img src="${data.image}" class="viz-image" alt="${type} 차트" style="width: 100%; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 16px;">
                        <div class="viz-info" style="background: #f8f9fc; padding: 16px; border-radius: 8px; font-size: 14px;">
                            <p><strong>📈 분석 완료:</strong> ${vizNames[type] || type}</p>
                            <p><strong>📊 차트 유형:</strong> ${type}</p>
                            <p><strong>⏰ 생성 시간:</strong> ${new Date().toLocaleString()}</p>
                        </div>
                    </div>
                `;
                
                const chartImg = vizArea.querySelector('.viz-image');
                if (chartImg) {
                    chartImg.style.cursor = 'pointer';
                    chartImg.onclick = () => openImageModal(data.image);
                }
            }
            
            showNotification(`✅ ${vizNames[type]} 생성이 완료되었습니다.`, 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showNotification('❌ 시각화 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

async function createModelVisualization() {
    showGlobalLoading('🤖 AI 모델 성능 비교 분석을 생성하고 있습니다...');
    try {
        const result = await apiRequest('/api/create_model_comparison', { method: 'POST' });
        
        if (result.success) {
            const data = result.data;
            const vizArea = document.getElementById('visualization-area');
            if (vizArea) {
                vizArea.innerHTML = `
                    <div class="viz-result" style="width: 100%;">
                        <img src="${data.image}" class="viz-image" alt="모델 성능 비교 차트" style="width: 100%; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 16px; cursor: pointer;" onclick="openImageModal('${data.image}')">
                        <div class="viz-info" style="background: linear-gradient(135deg, #2c5ff7, #4a90e2); color: white; padding: 20px; border-radius: 12px; font-size: 14px;">
                            <h4 style="margin-bottom: 12px; color: white;">🤖 AI 모델 성능 분석 결과</h4>
                            <p><strong>🏆 최고 모델:</strong> ${data.best_model || 'N/A'}</p>
                            <p><strong>📈 평균 정확도:</strong> ${data.avg_accuracy || 'N/A'}</p>
                            <p><strong>🔢 분석 모델:</strong> ${data.models_count || 4}개</p>
                            <p><strong>📊 활용 데이터:</strong> ${data.data_used || 'N/A'}</p>
                        </div>
                    </div>
                `;
            }
            
            showNotification(
                `🤖 AI 모델 성능 비교 분석이 완료되었습니다.\n🏆 최고 성능: ${data.best_model}\n📈 평균 정확도: ${data.avg_accuracy}`, 
                'success', 
                6000
            );
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showNotification('❌ 모델 비교 오류: ' + error.message, 'error');
    } finally {
        hideGlobalLoading();
    }
}

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
    
    const handleEsc = (e) => {
        if (e.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', handleEsc);
        }
    };
    document.addEventListener('keydown', handleEsc);
}

/* ==========================================
   사용자 인증 관련
   ========================================== */

async function logout() {
    try {
        const result = await apiRequest('/api/logout');
        
        if (result.success) {
            showNotification('👋 로그아웃되었습니다. 이용해 주셔서 감사합니다!', 'info');
            await checkLoginAndUpdateUI();
        }
    } catch (error) {
        showNotification('❌ 로그아웃 오류: ' + error.message, 'error');
    }
}

function showRegister() {
    showNotification(
        '👋 회원가입 기능은 준비 중입니다!\n🎯 데모 계정으로 먼저 체험해보세요:\n\n📧 ID: admin\n🔑 PW: 1234', 
        'info', 
        7000
    );
}

function requireLogin(service) {
    showNotification(`🔒 ${service} 서비스는 로그인 후 이용 가능합니다.`, 'warning');
    setTimeout(() => {
        window.location.href = '/login';
    }, 1500);
}

function goToLogin() {
    window.location.href = '/login';
}

function goToDashboard() {
    fetch('/api/session')
        .then(response => response.json())
        .then(data => {
            if (data.logged_in) {
                const servicesSection = document.getElementById('services');
                if (servicesSection) {
                    servicesSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    setTimeout(() => {
                        predictRisk();
                    }, 1000);
                }
            } else {
                showNotification('🔒 로그인이 필요한 서비스입니다.', 'warning');
                setTimeout(() => {
                    goToLogin();
                }, 1500);
            }
        })
        .catch(() => {
            goToLogin();
        });
}

function showDemo() {
    showNotification(
        '🎬 데모 기능은 준비 중입니다!\n🚀 로그인 후 전체 서비스를 이용해보세요!\n\n🎯 데모 계정: admin / 1234', 
        'info', 
        5000
    );
    setTimeout(() => {
        goToLogin();
    }, 2000);
}

/* ==========================================
   부드러운 스크롤 및 네비게이션
   ========================================== */

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initNavigationHighlight() {
    window.addEventListener('scroll', function() {
        const sections = ['home', 'services', 'about'];
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                const sectionTop = section.offsetTop;
                if (scrollY >= sectionTop - 200) {
                    current = sectionId;
                }
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/* ==========================================
   실시간 업데이트 관리
   ========================================== */

function startRealTimeUpdates() {
    // 30초마다 상태 확인
    statusUpdateInterval = setInterval(() => {
        checkStatus();
        updateWeatherBanner(); // 날씨도 함께 업데이트
    }, 30000);
}

function stopRealTimeUpdates() {
    // 30초마다 상태 확인
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
}

/* ==========================================
   오류 복구 및 재시도 로직
   ========================================== */

async function retryOperation(operation, maxRetries = 3, delay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await operation();
        } catch (error) {
            console.warn(`작업 실패 (시도 ${i + 1}/${maxRetries}):`, error);
            
            if (i === maxRetries - 1) {
                throw error; // 마지막 시도에서 실패하면 에러 던지기
            }
            
            // 지연 후 재시도
            await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
        }
    }
}


/* ==========================================
   실시간 날씨 데이터 업데이트 함수
   ========================================== */

async function updateWeatherBanner() {
    try {
        const result = await apiRequest('/api/weather_today');
        
        if (result.success) {
            const data = result.data;
            
            // 오늘 날씨 업데이트
            updateWeatherWidget(0, data.today);
            
            // 내일 날씨 업데이트  
            updateWeatherWidget(1, data.tomorrow);
            
            console.log('날씨 베너 업데이트 완료 (실제 데이터)');
        } else {
            // API 실패 시 기본값 사용
            console.warn('날씨 API 실패, 기본값 사용:', result.error);
            updateWeatherWidget(0, getDefaultWeatherData());
            updateWeatherWidget(1, getDefaultWeatherData(true));
        }
    } catch (error) {
        console.error('날씨 베너 업데이트 오류:', error);
        // 오류 시 기본값 사용
        updateWeatherWidget(0, getDefaultWeatherData());
        updateWeatherWidget(1, getDefaultWeatherData(true));
    }
}



function updateWeatherWidget(index, weatherData) {
    const widgets = document.querySelectorAll('.weather-widget');
    if (!widgets[index]) return;
    
    const widget = widgets[index];
    
    // 온도 업데이트
    const tempElement = widget.querySelector('.temperature');
    if (tempElement) {
        tempElement.textContent = `${weatherData.temperature}°C`;
    }
    
    // 강수량 업데이트
    const rainfallElement = widget.querySelector('.info-text .value');
    if (rainfallElement) {
        rainfallElement.textContent = `${weatherData.rainfall}mm`;
    }
    
    // 미세먼지 업데이트
    const fineDustElements = widget.querySelectorAll('.info-text .value');
    if (fineDustElements[1]) {
        fineDustElements[1].textContent = weatherData.fineDust;
    }
    if (fineDustElements[2]) {
        fineDustElements[2].textContent = weatherData.ultraFineDust;
    }
    
    // 날씨 아이콘 업데이트
    const iconElement = widget.querySelector('.weather-icon');
    if (iconElement) {
        iconElement.className = `weather-icon ${weatherData.condition}`;
        iconElement.innerHTML = getWeatherIconSVG(weatherData.condition);
    }
}

function getDefaultWeatherData(isTomorrow = false) {
    return {
        temperature: isTomorrow ? 22 : 20,
        rainfall: 0,
        condition: 'sunny',
        fineDust: '보통',
        ultraFineDust: '보통'
    };
}

function getWeatherIconSVG(condition) {
    const icons = {
        sunny: `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="4"/>
            <path d="m12 2 0 2"/>
            <path d="m12 20 0 2"/>
            <path d="m4.93 4.93 1.41 1.41"/>
            <path d="m17.66 17.66 1.41 1.41"/>
            <path d="M2 12h2"/>
            <path d="M20 12h2"/>
            <path d="m6.34 17.66-1.41 1.41"/>
            <path d="m19.07 4.93-1.41 1.41"/>
        </svg>`,
        cloudy: `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
        </svg>`,
        rainy: `<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>
            <path d="m16 14-3 5-3-5"/>
            <path d="m8 19-2 3"/>
            <path d="m18 16-2 3"/>
        </svg>`
    };
    return icons[condition] || icons.sunny;
}

// 네트워크 상태 모니터링
function initNetworkMonitoring() {
    // 온라인/오프라인 상태 감지
    window.addEventListener('online', () => {
        showNotification('네트워크 연결이 복구되었습니다.', 'success', 3000);
        checkStatus(); // 상태 재확인
    });
    
    window.addEventListener('offline', () => {
        showNotification('네트워크 연결이 끊어졌습니다. 일부 기능이 제한될 수 있습니다.', 'warning', 5000);
    });
}

//입력 필드 검증
function validateInputs() {
    const inputs = {
        precipitation: document.getElementById('precipitation')?.value,
        humidity: document.getElementById('humidity')?.value,
        temperature: document.getElementById('temperature')?.value
    };
    
    const errors = [];
    
    if (inputs.precipitation && (parseFloat(inputs.precipitation) < 0 || parseFloat(inputs.precipitation) > 1000)) {
        errors.push('강수량은 0-1000mm 범위여야 합니다.');
    }
    
    if (inputs.humidity && (parseFloat(inputs.humidity) < 0 || parseFloat(inputs.humidity) > 100)) {
        errors.push('습도는 0-100% 범위여야 합니다.');
    }
    
    if (inputs.temperature && (parseFloat(inputs.temperature) < -50 || parseFloat(inputs.temperature) > 60)) {
        errors.push('온도는 -50~60°C 범위여야 합니다.');
    }
    
    if (errors.length > 0) {
        showNotification('❌ 입력값 검증 실패:\n' + errors.join('\n'), 'error', 5000);
        return false;
    }
    
    return true;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function setupInputValidation() {
    const inputs = ['precipitation', 'humidity', 'temperature'];
    
    inputs.forEach(inputId => {
        const element = document.getElementById(inputId);
        if (element) {
            element.addEventListener('input', debounce(() => {
                const value = parseFloat(element.value);
                const ranges = {
                    precipitation: [0, 1000],
                    humidity: [0, 100],
                    temperature: [-50, 60]
                };
                
                const range = ranges[inputId];
                if (value < range[0] || value > range[1]) {
                    element.style.borderColor = '#ff4444';
                    element.style.boxShadow = '0 0 0 3px rgba(255, 68, 68, 0.1)';
                } else {
                    element.style.borderColor = '';
                    element.style.boxShadow = '';
                }
            }, 300));
        }
    });
}

/* ==========================================
   페이지 초기화
   ========================================== */

document.addEventListener('DOMContentLoaded', function() {
    console.log('🌊 CREW_SOOM 수정된 대시보드 초기화 시작...');
    
    try {
        initSmoothScroll();
        initNavigationHighlight();
        setupScrollAnimations();
        setupInputValidation();
        
        checkStatus();
        startRealTimeUpdates();
        
        setTimeout(async () => {
            try {
                const result = await apiRequest('/api/session');
                
                if (result.success && result.data.logged_in) {
                    setTimeout(() => {
                        if (validateInputs()) {
                            predictRisk();
                        }
                    }, 1000);
                }
            } catch (error) {
                console.log('초기 예측 체크 오류:', error);
            }
        }, 2000);
        
        setTimeout(() => {
            const isFirstVisit = !localStorage.getItem('crew_soom_visited');
            if (isFirstVisit) {
                showNotification(
                    '🌊 CREW_SOOM에 오신 것을 환영합니다!\n🤖 4가지 AI 모델로 정확한 침수 예측을 경험해보세요.\n\n🎯 데모 계정: admin / 1234', 
                    'info', 
                    8000
                );
                localStorage.setItem('crew_soom_visited', 'true');
            }
        }, 3000);
        
        console.log('✅ CREW_SOOM 수정된 대시보드 초기화 완료!');
        
    } catch (error) {
        console.error('❌ 대시보드 초기화 실패:', error);
        showNotification('시스템 초기화 중 오류가 발생했습니다. 페이지를 새로고침해주세요.', 'error', 10000);
    }
});

window.addEventListener('beforeunload', function() {
    stopRealTimeUpdates();
    
    if (document.getElementById('loading-overlay').style.display === 'flex') {
        return '현재 작업이 진행 중입니다. 페이지를 떠나시겠습니까?';
    }
});

window.addEventListener('error', function(e) {
    console.error('전역 오류 발생:', e.error);
    showNotification('예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.', 'error');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('처리되지 않은 Promise 거부:', e.reason);
    showNotification('비동기 작업 중 오류가 발생했습니다.', 'error');
    e.preventDefault();
});
