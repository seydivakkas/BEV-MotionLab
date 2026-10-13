"""Authenticated loopback-only JSON HTTP transport for the in-process v1 API.

This transport does not resolve file paths or serve a browser frontend.
"""
from __future__ import annotations
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_REQUEST_BYTES=262144


def create_server(dispatch, *, token: str, host='127.0.0.1', port=0):
    if host != '127.0.0.1':
        raise ValueError('LOOPBACK_ONLY: cannot bind on a non-loopback interface')
    if not isinstance(token,str) or len(token)<24:
        raise ValueError('strong bearer token (>=24 characters) required')
    if not isinstance(port,int) or not 0 <= port <= 65535:
        raise ValueError('invalid TCP port')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Do not log bearer tokens or user data.

        def _respond(self,status,body):
            b=json.dumps(body,allow_nan=False,sort_keys=True).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authorized(self):
            # Disallow browser-origin requests even if a token is somehow exposed.
            return ('Origin' not in self.headers and
                    secrets.compare_digest(self.headers.get('Authorization',''), 'Bearer '+token))

        def do_GET(self):
            if not self._authorized():
                return self._respond(403,{'error':'FORBIDDEN'})
            if self.path!='/v1/health':
                return self._respond(404,{'error':'NOT_FOUND'})
            return self._respond(200,{'schema_version':'1.0','status':'HEALTHY','network':'loopback-only'})

        def do_POST(self):
            if not self._authorized():
                return self._respond(403,{'error':'FORBIDDEN'})
            if self.path!='/v1/execute':
                return self._respond(404,{'error':'NOT_FOUND'})
            if self.headers.get('Content-Type','').split(';')[0].strip().lower()!='application/json':
                return self._respond(415,{'error':'JSON_REQUIRED'})
            raw_length=self.headers.get('Content-Length','')
            if (len(raw_length)>9 or not raw_length.isdigit()
                    or not 0 < int(raw_length) <= MAX_REQUEST_BYTES):
                return self._respond(413,{'error':'REQUEST_TOO_LARGE_OR_MISSING_LENGTH'})
            try:
                payload=json.loads(self.rfile.read(int(raw_length)).decode('utf-8'))
                result=dispatch(payload)
                return self._respond(200,{'schema_version':'1.0','result':result})
            except (ValueError,TypeError,KeyError,UnicodeDecodeError,json.JSONDecodeError) as exc:
                # No source file paths, stack traces or secrets are returned.
                return self._respond(422,{'error':'INPUT_REJECTED','detail':type(exc).__name__})

    return HTTPServer((host,port),Handler)