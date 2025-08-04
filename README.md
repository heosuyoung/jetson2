# jetson_server.py (젯슨에서 실행)
import socket
import json
import threading # 클라이언트 연결을 스레드로 처리
import sys # sys.exit() 사용

# 젯슨의 IP 주소는 0.0.0.0으로 설정하여 모든 인터페이스에서 연결을 받도록 합니다.
# 클라이언트가 이 IP로 접속하는 것이 아니라, 클라이언트는 젯슨의 실제 할당된 IP (예: 192.168.137.141)로 접속합니다.
HOST = '0.0.0.0'
PORT = 65432       # 라즈베리 파이 클라이언트와 동일한 포트 번호

# --- 데이터 수신 버퍼 및 JSON 파싱 함수 ---
# 클라이언트로부터 단일 JSON 메시지를 '\n'으로 구분하여 수신하는 함수
def receive_json_message(conn, buffer):
    while True:
        try:
            # 데이터 수신. 한번에 많은 데이터가 올 수 있으므로 충분히 크게 설정
            chunk = conn.recv(4096).decode('utf-8')
            if not chunk: # 클라이언트가 연결을 끊었을 때
                return None, buffer
            
            buffer += chunk # 버퍼에 수신된 데이터 추가
            
            # 버퍼에서 첫 번째 완전한 메시지를 찾습니다.
            while '\n' in buffer:
                message, buffer = buffer.split('\n', 1) # 첫 번째 '\n' 기준으로 분리
                try:
                    # 분리된 메시지를 JSON으로 파싱 시도
                    json_data = json.loads(message.strip()) # strip()으로 공백 제거
                    return json_data, buffer # 성공적으로 파싱된 JSON 데이터 반환
                except json.JSONDecodeError as e: # 오타 수정: SONDecodeError -> JSONDecodeError
                    print(f"JSON 디코딩 오류: {e}. 수신된 메시지: '{message}'")
                    # 잘못된 메시지는 버퍼에서 제거하고 다음 메시지 시도
                    continue # 다음 루프에서 버퍼의 나머지 부분을 계속 처리

        except socket.error as e:
            print(f"소켓 오류 발생: {e}")
            return None, buffer # 오류 발생 시 None 반환
        except Exception as e:
            print(f"수신 중 알 수 없는 오류 발생: {e}")
            return None, buffer # 알 수 없는 오류 발생 시 None 반환

# --- 클라이언트 연결 처리 함수 ---
def handle_client_connection(conn, addr):
    print(f"[연결됨] {addr} 에서 연결되었습니다.")
    buffer = "" # 클라이언트별 수신 버퍼 초기화
    
    try:
        while True:
            # 수정된 receive_json_message 함수 호출
            sensor_data, buffer = receive_json_message(conn, buffer)
            
            if sensor_data is None: # 연결 종료 또는 오류 발생 시
                print(f"[연결 종료 또는 오류] {addr} 와의 연결이 끊어졌습니다.")
                break # 루프 종료

            # 수신된 데이터 처리 및 출력
            timestamp = sensor_data.get("timestamp", "N/A")
            touch_detected = sensor_data.get("touch_detected", "N/A")
            
            print(f"\n--- 새로운 센서 데이터 수신 ({addr}) ---")
            print(f"시간: {timestamp}")
            print(f"터치 감지: {touch_detected}")
            # 다른 센서 데이터가 있다면 여기에 추가하여 출력
            print("---------------------------------------")

    except Exception as e:
        print(f"[오류] {addr} 클라이언트 처리 중 예외 발생: {e}")
    finally:
        conn.close()
        print(f"[연결 닫힘] {addr} 와의 연결이 닫혔습니다.")

# --- 서버 시작 함수 ---
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 소켓 재사용 옵션

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5) # 최대 5개의 동시 연결 허용
        print(f"서버가 {HOST}:{PORT} 에서 대기 중입니다...")

        while True:
            conn, addr = server_socket.accept() # 클라이언트 연결 대기
            # 각 클라이언트 연결을 새로운 스레드에서 처리
            client_thread = threading.Thread(target=handle_client_connection, args=(conn, addr))
            client_thread.daemon = True # 메인 스레드 종료 시 서브 스레드도 종료
            client_thread.start()

    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    except Exception as e:
        print(f"서버 시작 또는 실행 중 오류 발생: {e}")
    finally:
        server_socket.close()
        print("서버 소켓이 닫혔습니다.")
        sys.exit(0) # 정상 종료

if __name__ == '__main__':
    start_server()
