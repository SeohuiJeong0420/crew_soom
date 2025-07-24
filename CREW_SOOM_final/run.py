from modules import preprocessor, trainer_rf, trainer_xgb, trainer_lstm_cnn, trainer_transformer, trainer, web_app, visualizer

def main():
    print("\nCREW_SOOM 침수 예측 시스템")
    print("=" * 60)
    print("데이터 준비")
    print("1. 시간별/일별 날씨 데이터 수집")
    print("2. 데이터 전처리 및 침수 라벨링")
    print("3. XGBoost용 파생 변수 생성")
    print()
    print("AI 모델 훈련")
    print("4. RandomForest 모델 학습")
    print("5. XGBoost 모델 학습")
    print("6. LSTM+CNN 모델 학습")
    print("7. Transformer 모델 학습")
    print("8. 전체 모델 학습 (4가지 모두)")
    print()
    print("분석 및 비교")
    print("9. 모델 성능 비교 시각화")
    print()
    print("웹 서비스")
    print("10. 웹 애플리케이션 실행")
    print()
    print("11. 종료")
    print("=" * 60)

    while True:
        choice = input("실행할 작업을 선택하세요 (1~11): ")
        
        if choice == "1":
            print("\n기상 데이터 수집을 시작합니다...")
            try:
                preprocessor.preprocess_data()
                print("데이터 수집이 완료되었습니다.\n")
            except Exception as e:
                print(f"데이터 수집 오류: {e}\n")
            
        elif choice == "2":
            print("\n데이터 전처리 및 침수 라벨링을 시작합니다...")
            try:
                trainer.preprocess_hourly_data()
                trainer.preprocess_daily_data()
                print("데이터 전처리가 완료되었습니다.\n")
            except Exception as e:
                print(f"데이터 전처리 오류: {e}\n")
            
        elif choice == "3":
            print("\nXGBoost용 파생 변수를 생성합니다...")
            try:
                trainer.preprocess_xgboost_features()
                print("파생 변수 생성이 완료되었습니다.\n")
            except Exception as e:
                print(f"파생 변수 생성 오류: {e}\n")
            
        elif choice == "4":
            print("\nRandomForest 모델 학습을 시작합니다...")
            try:
                trainer_rf.train_random_forest()
                print("RandomForest 모델 학습이 완료되었습니다.\n")
            except Exception as e:
                print(f"RandomForest 모델 학습 오류: {e}\n")
            
        elif choice == "5":
            print("\nXGBoost 모델 학습을 시작합니다...")
            try:
                trainer_xgb.train_xgboost()
                print("XGBoost 모델 학습이 완료되었습니다.\n")
            except Exception as e:
                print(f"XGBoost 모델 학습 오류: {e}\n")
                
        elif choice == "6":
            print("\nLSTM+CNN 모델 학습을 시작합니다...")
            try:
                trainer_lstm_cnn.train_lstm_cnn()
                print("LSTM+CNN 모델 학습이 완료되었습니다.\n")
            except Exception as e:
                print(f"LSTM+CNN 모델 학습 오류: {e}")
                print("TensorFlow가 설치되어 있는지 확인해주세요.\n")
                
        elif choice == "7":
            print("\nTransformer 모델 학습을 시작합니다...")
            try:
                trainer_transformer.train_transformer()
                print("Transformer 모델 학습이 완료되었습니다.\n")
            except Exception as e:
                print(f"Transformer 모델 학습 오류: {e}")
                print("TensorFlow가 설치되어 있는지 확인해주세요.\n")
                
        elif choice == "8":
            print("\n전체 AI 모델 학습을 시작합니다...")
            print("이 작업은 몇 분 정도 소요될 수 있습니다.")
            print("-" * 50)
            
            models_completed = 0
            total_models = 4
            
            # RandomForest 학습
            print("\n[1/4] RandomForest 모델 학습 중...")
            try:
                trainer_rf.train_random_forest()
                models_completed += 1
                print("RandomForest 완료")
            except Exception as e:
                print(f"RandomForest 실패: {e}")
            
            # XGBoost 학습
            print("\n[2/4] XGBoost 모델 학습 중...")
            try:
                trainer_xgb.train_xgboost()
                models_completed += 1
                print("XGBoost 완료")
            except Exception as e:
                print(f"XGBoost 실패: {e}")
            
            # LSTM+CNN 학습
            print("\n[3/4] LSTM+CNN 모델 학습 중...")
            try:
                trainer_lstm_cnn.train_lstm_cnn()
                models_completed += 1
                print("LSTM+CNN 완료")
            except Exception as e:
                print(f"LSTM+CNN 실패: {e}")
                print("TensorFlow 설치 필요")
            
            # Transformer 학습
            print("\n[4/4] Transformer 모델 학습 중...")
            try:
                trainer_transformer.train_transformer()
                models_completed += 1
                print("Transformer 완료")
            except Exception as e:
                print(f"Transformer 실패: {e}")
                print("TensorFlow 설치 필요")
            
            print("\n" + "=" * 50)
            print(f"전체 모델 학습 완료: {models_completed}/{total_models}개 성공")
            if models_completed == total_models:
                print("모든 AI 모델이 성공적으로 훈련되었습니다!")
            elif models_completed > 0:
                print("일부 모델만 훈련되었습니다. 오류를 확인해주세요.")
            else:
                print("모든 모델 훈련에 실패했습니다. 데이터를 먼저 준비해주세요.")
            print("=" * 50 + "\n")
            
        elif choice == "9":
            print("\n모델 성능 비교 시각화를 생성합니다...")
            try:
                visualizer.plot_model_comparison()
                print("모델 성능 비교 차트가 'outputs/model_comparison_metrics.png'에 저장되었습니다.\n")
            except Exception as e:
                print(f"시각화 생성 오류: {e}\n")
            
        elif choice == "10":
            print("\n웹 애플리케이션을 시작합니다...")
            print("브라우저에서 http://localhost:5000 에 접속하세요")
            print("로그인 정보: ID=admin, PW=1234")
            print("전체 기능을 사용하려면 먼저 모델을 훈련해주세요!")
            print("-" * 50)
            
            # 웹 앱 실행
            try:
                web_app.app.run(
                    host='0.0.0.0',
                    port=5000,
                    debug=False,  # 프로덕션 환경에서는 False
                    threaded=True
                )
            except KeyboardInterrupt:
                print("\n웹 애플리케이션이 종료되었습니다.")
            except Exception as e:
                print(f"\n웹 애플리케이션 실행 오류: {e}")
            
        elif choice == "11":
            print("\nCREW_SOOM을 이용해 주셔서 감사합니다!")
            print("침수 예측으로 안전한 세상을 만들어가요!")
            break
            
        else:
            print("잘못된 선택입니다. 1~11 중에서 입력하세요.\n")

def show_quick_start_guide():
    """빠른 시작 가이드"""
    print("\n빠른 시작 가이드:")
    print("-" * 40)
    print("처음 사용하시는 경우:")
    print("1 → 2 → 3 → 8 → 10 순서로 실행")
    print()
    print("개별 모델만 훈련하는 경우:")
    print("1 → 2 → 3 → 4,5,6,7 중 선택")
    print()
    print("웹앱만 사용하는 경우:")
    print("10 선택 (단, 모델이 미리 훈련되어 있어야 함)")
    print("-" * 40)

if __name__ == "__main__":
    show_quick_start_guide()
    main()