"""Who is asking: the presenter's own machine, or a guest on the network.

In LAN demo mode the API listens on 127.0.0.1 only and is reached through
the Vite proxy, which runs with ``xfwd`` and so appends the real peer
address to ``X-Forwarded-For``. The last entry is the one the proxy wrote;
anything before it came from the client and cannot be trusted.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import quote

from fastapi import HTTPException, Request


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else ""


def _is_own_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    if addr.is_loopback:
        return True
    # Binding only succeeds for an address assigned to one of this machine's
    # interfaces, so the presenter opening the LAN URL on the Mac still counts.
    family = socket.AF_INET6 if addr.version == 6 else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.bind((str(addr), 0))
        return True
    except OSError:
        return False


def is_host(request: Request) -> bool:
    return _is_own_address(client_ip(request))


def require_host(request: Request) -> None:
    if not is_host(request):
        raise HTTPException(
            status_code=403,
            detail="Only the presenter's computer can remove files from the session.",
        )


def attachment_headers(filename: str, size: int) -> dict[str, str]:
    """Content-Disposition that survives non-ASCII names (headers are latin-1)."""
    fallback = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in filename)
    return {
        "Content-Disposition": f"attachment; filename=\"{fallback}\"; "
                               f"filename*=UTF-8''{quote(filename, safe='')}",
        "Content-Length": str(size),
    }
