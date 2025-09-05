# server/boot.py
from http.sse_api import serve_async
def boot_http():
    return serve_async()
