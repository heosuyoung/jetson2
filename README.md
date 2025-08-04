# jetson_server.py (젯슨에서 실행)
import socket
import json
import threading
import sys
import time # 디버깅용으로 추가: import time

HOST = '0.0.0.0'
PORT = 65432

# --- 데이터 수신 버퍼 및 JSON 파싱 함수 (최종 디버그 버전) ---
def receive_json_message(conn, buffer):
    while True:
        try:
            print(f"[{time.time()}] [DEBUG_RECEIVE] conn.recv(4096) 호출 직전. 데이터 대기 중...")
            chunk = conn.recv(4096)
            
            if not chunk:
                print(f"[{time.time()}] [DEBUG_RECEIVE] 클라이언트 ({conn.getpeername()}) 연결 종료 감지 (chunk is empty).")
                return None, buffer
            
            decoded_chunk = chunk.decode('utf-8')
            print(f"[{time.time()}] [DEBUG_RECEIVE] 수신된 청크: '{decoded_chunk.strip()}' (길이: {len(decoded_chunk)})")
            
            buffer += decoded_chunk
            print(f"[{time.time()}] [DEBUG_RECEIVE] 현재 버퍼: '{buffer.strip()}' (총 길이: {len(buffer)})")
            print(f"[{time.time()}] [DEBUG_RECEIVE] 버퍼에 줄바꿈('\\n') 존재 여부: {'\\n' in buffer}")
            
            while '\n' in buffer:
                message, buffer = buffer.split('\n', 1)
                print(f"[{time.time()}] [DEBUG_RECEIVE] 분리된 메시지: '{message.strip()}' (길이: {len(message.strip())})")
                
                try:
                    json_data = json.loads(message.strip())
                    print(f"[{time.time()}] [DEBUG_RECEIVE] JSON 파싱 성공!")
                    return json_data, buffer # JSON 파싱 성공 시 데이터 반환
                except json.JSONDecodeError as e:
                    print(f"[{time.time()}] JSON 디코딩 오류: {e}. 수신된 메시지: '{message.strip()}'")
                    # JSON 디코딩 실패 시, 버퍼에 남은 다른 메시지를 계속 시도
                    continue 

        except socket.timeout: # recv()에 타임아웃이 설정된 경우
            print(f"[{time.time()}] [DEBUG_RECEIVE] 소켓 타임아웃 발생. 데이터 대기 중...")
            return None, buffer 
        except socket.error as e:
            print(f"[{time.time()}] [DEBUG_RECEIVE] 소켓 오류 발생: {e}")
            return None, buffer
        except Exception as e:
            print(f"[{time.time()}] [DEBUG_RECEIVE] 수신 중 알 수 없는 오류 발생: {e}")
            return None, buffer

# --- 클라이언트 연결 처리 함수 --- (이 부분은 그대로)
def handle_client_connection(conn, addr):
    print(f"[연결됨] {addr} 에서 연결되었습니다.")
    buffer = ""
    try:
        while True:
            sensor_data, buffer = receive_json_message(conn, buffer)
            
            if sensor_data is None:
                print(f"[연결 종료 또는 오류] {addr} 와의 연결이 끊어졌습니다.")
                break

            # --- 이 부분이 이제 정상적으로 출력될 것으로 예상됩니다 ---
            timestamp = sensor_data.get("timestamp", "N/A")
            touch_detected = sensor_data.get("touch_detected", "N/A")
            
            print(f"\n--- 새로운 센서 데이터 수신 ({addr}) ---")
            print(f"시간: {timestamp}")
            print(f"터치 감지: {touch_detected}")
            print("---------------------------------------")

    except Exception as e:
        print(f"[오류] {addr} 클라이언트 처리 중 예외 발생: {e}")
    finally:
        conn.close()
        print(f"[연결 닫힘] {addr} 와의 연결이 닫혔습니다.")

# --- 서버 시작 함수 --- (이 부분은 그대로)
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"서버가 {HOST}:{PORT} 에서 대기 중입니다...")

        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client_connection, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()

    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    except Exception as e:
        print(f"서버 시작 또는 실행 중 오류 발생: {e}")
    finally:
        server_socket.close()
        print("서버 소켓이 닫혔습니다.")
        sys.exit(0)

if __name__ == '__main__':
    start_server()
