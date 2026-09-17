"""Serve only the dashboard folder, on the port forwarded by Codespaces."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "dist"
    print("Dashboard: http://localhost:8000  |  Ctrl+C stops the server", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), partial(SimpleHTTPRequestHandler, directory=str(folder))).serve_forever()
