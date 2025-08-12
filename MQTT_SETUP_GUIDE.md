# MQTT 설정 및 사용 가이드

## 🏗️ **전체 시스템 구조**

```
라즈베리파이 MQTT 브로커 (192.168.137.82)
    ↕️
├── Jetson (192.168.43.10) → 운전자 모니터링 데이터 발행
├── 백엔드 서버 → 데이터 수신 및 저장
└── Wear OS → 긴급 상황 감지 및 알림
```

---

## 📡 **MQTT 토픽 구조**

### **발행 토픽 (Jetson → 브로커)**
```
car/{user_id}/alert      # 알림 메시지
car/{user_id}/status     # 상태 정보
```

### **구독 토픽 (브로커 → 클라이언트)**
```
car/+/alert             # 모든 사용자 알림
car/+/status            # 모든 사용자 상태
car/emergency/alert     # 긴급 상황
```

---

## 🚀 **사용 방법**

### **1. Jetson에서 MQTT 활성화**

```bash
# MQTT 기능을 활성화하여 실행
python3 v1ai_model_modified.py --enable_mqtt --user_id "DriverA"
```

**키보드 컨트롤:**
- `R`: 운전자 얼굴 등록
- `I`: 운전자 식별
- `B`: Baseline 재측정
- `ESC`: 종료

### **2. 백엔드 서버 실행**

```bash
# Spring Boot 서버 실행
./gradlew bootRun
```

**로그 확인:**
- MQTT 연결 상태
- 수신된 알림 메시지
- 데이터베이스 저장 상태

### **3. Wear OS 앱 실행**

**MainActivity에서 MQTT 연결:**
```kotlin
// MQTT 매니저 초기화 및 연결
mqttManager = MqttManager(this)
mqttManager.connect()
```

---

## 🧪 **테스트 방법**

### **1. MQTT 연결 테스트**

```bash
# Jetson에서 테스트 스크립트 실행
python3 test_mqtt.py
```

**예상 결과:**
```
🚀 MQTT 연결 테스트 시작
   브로커: 192.168.137.82:1883
   사용자: moring
✅ MQTT 브로커에 연결되었습니다!
📤 테스트 알림 발행 중...
📨 메시지 수신 대기 중...
```

### **2. 라즈베리파이에서 브로커 상태 확인**

```bash
# 연결된 클라이언트 확인
mosquitto_sub -h localhost -t '$SYS/broker/clients/connected' -u moring -P change_me_123

# 메시지 통계 확인
mosquitto_sub -h localhost -t '$SYS/broker/messages/stored' -u moring -P change_me_123
```

---

## 📊 **메시지 형식**

### **졸음 감지 알림**
```json
{
  "type": "drowsiness",
  "timestamp": 1640995200.123,
  "data": {
    "ear": 0.15,
    "pitch": 12.5,
    "eye_threshold": 0.25,
    "pitch_threshold": 15.0
  }
}
```

### **시선 이탈 알림**
```json
{
  "type": "distraction",
  "timestamp": 1640995200.123,
  "data": {
    "yaw": 25.0,
    "roll": 18.0,
    "yaw_threshold": 20.0,
    "roll_threshold": 15.0
  }
}
```

### **휴대폰 사용 알림**
```json
{
  "type": "phone_usage",
  "timestamp": 1640995200.123,
  "data": {
    "detected_objects": 1,
    "confidence": 0.85
  }
}
```

---

## 🔧 **문제 해결**

### **1. MQTT 연결 실패**

**확인 사항:**
- 라즈베리파이 브로커 실행 상태
- 네트워크 연결 상태
- IP 주소 및 포트 번호
- 사용자명/비밀번호

**해결 방법:**
```bash
# 브로커 상태 확인
sudo systemctl status mosquitto

# 연결 테스트
mosquitto_pub -h 192.168.137.82 -t "test" -m "hello" -u moring -P change_me_123
```

### **2. 메시지 수신 안됨**

**확인 사항:**
- 토픽 구독 상태
- 메시지 발행 상태
- 네트워크 방화벽 설정

**해결 방법:**
```bash
# 구독 테스트
mosquitto_sub -h 192.168.137.82 -t "car/+/alert" -u moring -P change_me_123
```

---

## 📝 **설정 파일**

### **라즈베리파이 MQTT 브로커 설정**
```conf
# /etc/mosquitto/mosquitto.conf
port 1883
log_type all
log_timestamp true
allow_anonymous true
password_file /etc/mosquitto/passwd
persistence true
persistence_location /var/lib/mosquitto/
max_connections 100
```

### **백엔드 서버 설정**
```properties
# application.properties
mqtt.broker.url=tcp://192.168.137.82:1883
mqtt.client.id=backend_client
mqtt.username=moring
mqtt.password=kimoring
```

---

## ✅ **완료 확인 체크리스트**

- [ ] 라즈베리파이 MQTT 브로커 설치 및 실행
- [ ] Jetson에서 MQTT 클라이언트 연결
- [ ] 백엔드 서버에서 MQTT 메시지 수신
- [ ] Wear OS에서 MQTT 연결
- [ ] 테스트 메시지 발행/수신 확인
- [ ] 실제 운전자 모니터링 데이터 전송 확인

---

## 🎯 **다음 단계**

1. **데이터베이스 연동**: 수신된 데이터를 DB에 저장
2. **FCM 푸시 알림**: 긴급 상황 시 푸시 알림 발송
3. **실시간 대시보드**: 웹 대시보드로 실시간 모니터링
4. **알림 설정**: 사용자별 알림 설정 관리
