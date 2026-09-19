"""Temporary Phase 3 network diagnostic (deleted after use)."""

import socket
import time

HOST = "archive-api.open-meteo.com"

print("--- DNS ---")
try:
    infos = socket.getaddrinfo(HOST, 443, proto=socket.IPPROTO_TCP)
    for family, _, _, _, addr in infos[:6]:
        print("  ", family.name, addr[0])
except Exception as exc:
    print("   DNS FAILED:", type(exc).__name__, exc)

print("--- TCP connect (5 attempts, 20s each) ---")
ip = None
try:
    ip = socket.getaddrinfo(HOST, 443, proto=socket.IPPROTO_TCP)[0][4][0]
except Exception:
    pass
for attempt in range(1, 6):
    t0 = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(20)
    try:
        sock.connect((ip or HOST, 443))
        print(f"   attempt {attempt}: CONNECTED in {time.time() - t0:.1f}s to {ip}")
        sock.close()
        break
    except Exception as exc:
        print(f"   attempt {attempt}: FAILED after {time.time() - t0:.1f}s - {type(exc).__name__}: {exc}")
    finally:
        try:
            sock.close()
        except Exception:
            pass
    time.sleep(3)

print("--- HTTPS GET (60s timeout) ---")
import urllib.request

url = ("https://archive-api.open-meteo.com/v1/archive"
       "?latitude=8.482&longitude=76.920&start_date=2014-01-01&end_date=2014-01-02"
       "&hourly=temperature_2m&timezone=GMT")
t0 = time.time()
try:
    with urllib.request.urlopen(url, timeout=60) as resp:
        print("   OK", resp.status, len(resp.read(200)), "bytes", f"{time.time() - t0:.1f}s")
except Exception as exc:
    print(f"   FAILED after {time.time() - t0:.1f}s - {type(exc).__name__}: {exc}")
