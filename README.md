import serial
import time
import socket

# ====================================================================
# [설정] Jetson Nano의 IP 주소와 포트 번호를 여기에 입력하세요
# ====================================================================
NANOJETSON_IP = "192.168.137.141"
PORT = 12345
# ====================================================================

# MH-Z19B가 연결된 시리얼 포트
ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)

def read_co2():
    """
    MH-Z19B 센서로부터 CO2 농도를 읽어오는 함수
    """
    try:
        ser.write(b'\xFF\x01\x86\x00\x00\x00\x00\x00\x79')
        time.sleep(0.1)
        response = ser.read(9)
        
        if len(response) != 9:
            return None

        # 체크섬 검사
        checksum = 0xFF - (sum(response[1:8]) % 256) + 1
        if response[0] == 0xFF and response[1] == 0x86 and response[8] == (checksum & 0xFF):
            co2 = response[2] * 256 + response[3]
            return co2
        else:
            return None
            
    except serial.SerialException as e:
        print(f"시리얼 통신 오류: {e}")
        return None
    except Exception as e:
        print(f"CO2 읽기 오류: {e}")
        return None

def send_co2_to_jetson(co2_value):
    """
    CO2 값을 Jetson Nano 서버로 전송하는 함수
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((NANOJETSON_IP, PORT))
            message = str(co2_value).encode('utf-8')
            s.sendall(message)
            print(f"CO2 값 {co2_value} ppm 전송 완료.")
    except Exception as e:
        print(f"데이터 전송 실패: {e}")

# ====================================================================
# 메인 실행 루프
# ====================================================================
while True:
    co2 = read_co2()
    if co2 is not None:
        print(f"CO2 농도: {co2} ppm")
        send_co2_to_jetson(co2)
    else:
        print("센서 응답 없음 또는 체크섬 오류")
    time.sleep(2)


#================================================================
#================================================================
import socket

# ====================================================================
# [설정] 라즈베리 파이와 동일한 포트 번호를 사용해야 합니다
# ====================================================================
HOST = '0.0.0.0' # 모든 네트워크 인터페이스로부터의 연결을 받음
PORT = 12345
# ====================================================================

def run_server():
    """
    라즈베리 파이로부터 CO2 값을 수신하는 서버
    """
    print(f"Jetson Nano 서버 시작. 포트 {PORT}에서 데이터를 기다리는 중...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        conn, addr = s.accept()
        with conn:
            print(f"라즈베리 파이에서 연결됨: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("클라이언트 연결 종료.")
                    break
                try:
                    co2_value = data.decode('utf-8')
                    print(f"수신된 CO2 값: {co2_value} ppm")
                except Exception as e:
                    print(f"데이터 디코딩 오류: {e}")

if __name__ == "__main__":
    run_server()
