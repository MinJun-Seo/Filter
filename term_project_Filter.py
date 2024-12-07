#opencv를 이용하여 실시간 필터기능 지원, 간단히 색상만 나옴. 
#R,G,B버튼을 누르면 각각의 색으로 필터가 전환, X키를 누를 시 필터 해제, spacebar누를시 스크린샷
import cv2
import numpy as np
from datetime import datetime  # 날짜와 시간을 다루기 위해 추가

# Haar Cascade 파일 경로
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 얼굴 감지 모델 로드
face_cascade = cv2.CascadeClassifier(cascade_path)

# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

print("ESC 키를 눌러 종료하세요. R/G/B 키로 필터 색상 변경. X 키로 필터 끄기. 스페이스바로 스크린샷 저장.")

# 초기 필터 색상 (녹색)
filter_color = (0, 255, 0)
filter_enabled = True  # 필터 활성화 여부

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    # 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if filter_enabled:  # 필터가 활성화된 경우만 적용
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), filter_color, -1)
            alpha = 0.5  # 투명도
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 얼굴 경계선 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 결과 창 표시
    cv2.imshow('Face Detection with Filter', frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF

    # ESC 키로 종료
    if key == 27:
        break
    # R 키로 필터를 빨간색으로 변경
    elif key == ord('r'):
        filter_color = (0, 0, 255)
        filter_enabled = True  # 필터 활성화
    # G 키로 필터를 녹색으로 변경
    elif key == ord('g'):
        filter_color = (0, 255, 0)
        filter_enabled = True  # 필터 활성화
    # B 키로 필터를 파란색으로 변경
    elif key == ord('b'):
        filter_color = (255, 0, 0)
        filter_enabled = True  # 필터 활성화
    # X 키로 필터 끄기
    elif key == ord('x'):
        filter_enabled = False  # 필터 비활성화
    # 스페이스바로 스크린샷 저장
    elif key == 32:  # 스페이스바 키
        # 현재 날짜와 시간을 파일명에 추가
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{current_time}.png"
        cv2.imwrite(filename, frame)  # 프레임 저장
        print(f"스크린샷 저장됨: {filename}")

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()