# run_server.py
from imu.api.http_api import serve_http, GLOBAL_BROKER
from imu.api.stream_http import serve_stream

if __name__ == "__main__":
    serve_http(8080)
    serve_stream(GLOBAL_BROKER, 8090)
    import time
    print("IMU HTTP: 8080; STREAM: 8090")
    while True: time.sleep(3600)