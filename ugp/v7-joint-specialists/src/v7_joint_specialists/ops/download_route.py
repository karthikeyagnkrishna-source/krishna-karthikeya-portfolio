"""Private CONNECT relay for this run's public dataset downloads. TLS stays end to end."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import select
import socket

def allowed_target(value):
    host,port=value.rsplit(':',1);host=host.lower();port=int(port)
    if port!=443 or not (host=='huggingface.co' or host.endswith('.huggingface.co') or host=='hf.co' or host.endswith('.hf.co')):
        raise ValueError('Only public dataset TLS endpoints are allowed')
    return host,port

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def do_CONNECT(self):
        try:host,port=allowed_target(self.path)
        except (ValueError,TypeError):self.send_error(403);return
        try:upstream=socket.create_connection((host,port),timeout=30)
        except OSError:self.send_error(502);return
        with upstream:
            upstream.settimeout(None)
            self.send_response(200,'Connection established');self.end_headers()
            try:
                while True:
                    readable,_,_=select.select([self.connection,upstream],[],[],120)
                    if not readable:return
                    for source in readable:
                        data=source.recv(65536)
                        if not data:return
                        destination=upstream if source is self.connection else self.connection
                        destination.sendall(data)
            except OSError:return
    def log_message(self,format,*args):
        # Log only status, never credential-bearing URLs or arbitrary client headers.
        print('dataset TLS connection',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18882);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True
    print('private recording relay ready',flush=True);server.serve_forever()

if __name__=='__main__':main()
