# jetson_server.py (젯슨에서 실행)
import socket
import json
import threading # 클라이언트 연결을 스레드로 처리
import sys # sys.exit() 사용

# 젯슨의 IP 주소는 0.0.0.0으로 설정하여 모든 인터페이스에서 연결을 받도록 합니다.
# 클라이언트가 이 IP로 접속하는 것이 아니라, 클라이언트는 젯슨의 실제 할당된 IP (예: 192.168.137.141)로 접속합니다.
HOST = '0.0.0.0'
PORT = 65432        # 라즈베리 파이 클라이언트와 동일한 포트 번호

# --- 데이터 수신 버퍼 및 JSON 파싱 함수 (디버그용 단순화) ---
# 기존 receive_json_message 함수를 이 함수로 완전히 교체해주세요!
def receive_json_message(conn, buffer): # 함수 이름은 그대로 둡니다.
    try:
        # 이 메시지가 뜨는지 확인합니다.
        print(f"[DEBUG_ULTRA] conn.recv(4096) 호출 직전. 클라이언트 ({conn.getpeername()})에서 데이터 대기 중...")
        
        chunk = conn.recv(4096) # .decode() 하지 않고 raw 바이트로 받기
        
        if not chunk:
            # 이 메시지가 뜨는지 확인합니다.
            print(f"[DEBUG_ULTRA] 클라이언트 ({conn.getpeername()}) 연결 종료 감지 (chunk is empty).")
            return None, buffer # 연결 종료 시 None 반환
        
        # chunk에 단 한 바이트라도 들어왔다면 아래 메시지가 무조건 출력됩니다.
        print(f"[DEBUG_ULTRA] RAW 데이터 수신! 길이: {len(chunk)}, 내용: {chunk!r}") # !r은 raw 바이트 출력
        
        # 디코딩 시도 (오류 확인용)
        try:
            decoded_chunk = chunk.decode('utf-8')
            print(f"[DEBUG_ULTRA] 디코딩된 청크: '{decoded_chunk.strip()}'")
            print(f"[DEBUG_ULTRA] 디코딩된 청크 길이: {len(decoded_chunk)}")
        except UnicodeDecodeError as e:
            print(f"[DEBUG_ULTRA] 디코딩 오류: {e}. 수신된 데이터가 UTF-8이 아님.")
            
        # 이 디버그 버전에서는 JSON 파싱을 건너뛰고 빈 딕셔너리를 반환하여 메인 루프가 계속 돌게 합니다.
        # 우리의 목표는 'recv'가 작동하는지 확인하는 것뿐입니다.
        return {}, buffer # 더미 데이터 반환
        
    except socket.timeout:
        print(f"[DEBUG_ULTRA] 소켓 타임아웃 발생. 데이터 대기 중...")
        return None, buffer
    except socket.error as e:
        print(f"[DEBUG_ULTRA] 소켓 오류 발생: {e}")
        return None, buffer
    except Exception as e:
        print(f"[DEBUG_ULTRA] 수신 중 알 수 없는 오류 발생: {e}")
        return None, buffer

# --- 클라이언트 연결 처리 함수 --- (이 부분은 그대로)
def handle_client_connection(conn, addr):
    print(f"[연결됨] {addr} 에서 연결되었습니다.")
    buffer = "" # 클라이언트별 수신 버퍼 초기화
    
    try:
        while True:
            # 수정된 receive_json_message 함수 호출
            sensor_data, buffer = receive_json_message(conn, buffer) # 이 라인은 건드리지 마세요.
            
            if sensor_data is None: # 연결 종료 또는 오류 발생 시
                print(f"[연결 종료 또는 오류] {addr} 와의 연결이 끊어졌습니다.")
                break # 루프 종료

            # 디버그 버전에서는 이 아래 print 문들이 바로 뜨지 않을 수 있습니다.
            # print(f"\n--- 새로운 센서 데이터 수신 ({addr}) ---")
            # print(f"시간: {timestamp}")
            # print(f"터치 감지: {touch_detected}")
            # print("---------------------------------------")

    except Exception as e:
        print(f"[오류] {addr} 클라이언트 처리 중 예외 발생: {e}")
    finally:
        conn.close()
        print(f"[연결 닫힘] {addr} 와의 연결이 닫혔습니다.")

# --- 서버 시작 함수 --- (이 부분은 그대로)
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
