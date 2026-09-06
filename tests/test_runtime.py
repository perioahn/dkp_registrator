import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from webapp import server as app_server


def running_server(identity=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/app' and identity is not None:
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(identity).encode())
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_old_server_cannot_reopen_old_ui():
    old = running_server()
    try:
        port, sock = app_server.choose_listener(old.server_port)
        try:
            assert port != old.server_port
            assert sock is not None
        finally:
            if sock: sock.close()
    finally:
        old.shutdown(); old.server_close()


def test_same_build_reuses_existing_session():
    same = running_server(app_server.app_identity())
    try:
        port, sock = app_server.choose_listener(same.server_port)
        assert port == same.server_port and sock is None
    finally:
        same.shutdown(); same.server_close()


def test_different_build_is_not_reused():
    identity = {**app_server.app_identity(), 'build': 'old-build'}
    old = running_server(identity)
    try:
        port, sock = app_server.choose_listener(old.server_port)
        try:
            assert port != old.server_port and sock is not None
        finally:
            if sock: sock.close()
    finally:
        old.shutdown(); old.server_close()
