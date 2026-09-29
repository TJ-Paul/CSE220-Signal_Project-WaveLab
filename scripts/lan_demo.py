"""LAN demo helper: find this Mac's address on the network, check that
everything works offline, and print/open QR codes for joining.

Run by `./run.sh --demo` once both servers are up. Wi-Fi details for the
join QR come from WIFI_SSID / WIFI_PASSWORD (run.sh loads them from
.env.demo).
"""
from __future__ import annotations

import argparse
import html
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import segno
from segno import helpers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".demo" / "join.html"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def lan_addresses() -> list[tuple[str, str]]:
    """(interface, IPv4) for every private address, VPN tunnels excluded."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    found, iface = [], ""
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":", 1)[0]
        match = re.match(r"\s+inet (\d+\.\d+\.\d+\.\d+)", line)
        if match and not iface.startswith(("lo", "utun")):
            found.append((iface, match.group(1)))
    return found


def self_assigned(ip: str) -> bool:
    return ipaddress.ip_address(ip).is_link_local  # 169.254/16: DHCP never answered


def primary_address(candidates: list[tuple[str, str]]) -> str | None:
    """The address on the interface the default route uses (no packet is sent)."""
    usable = [ip for _, ip in candidates if not self_assigned(ip)]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.254.254.254", 1))
            routed = probe.getsockname()[0]
        if routed in usable:
            return routed
    except OSError:
        pass
    return usable[0] if usable else None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def model_warnings() -> list[str]:
    """The venue network has no internet, so every model must already be cached."""
    warnings = []
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    if not list((cache / "whisper").glob("*turbo*.pt")):
        warnings.append("Whisper model not downloaded — run Lyrics & Transcription once "
                        "with internet before the demo.")
    hub = Path(os.getenv("HF_HUB_CACHE") or Path(os.getenv("HF_HOME", cache / "huggingface")) / "hub")
    if not list((hub / "models--adefossez--HTDemucs").glob("snapshots/*/*.safetensors")):
        warnings.append("Demucs model not downloaded — run Karaoke & Vocals (ML engine) once "
                        "with internet before the demo.")
    return warnings


def firewall_warnings() -> list[str]:
    tool = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    if not os.path.exists(tool):
        return []
    try:
        state = subprocess.run([tool, "--getglobalstate"], capture_output=True, text=True).stdout.lower()
        block_all = subprocess.run([tool, "--getblockall"], capture_output=True, text=True).stdout.lower()
    except OSError:
        return []
    if "block all state set to enabled" in block_all:
        return ["macOS firewall is set to BLOCK ALL incoming connections — phones cannot connect. "
                "Turn that off in System Settings → Network → Firewall → Options."]
    if "state = 1" in state or "state = 2" in state:
        # The firewall tracks the real binary, not a Homebrew/nvm symlink.
        node = os.path.realpath(shutil.which("node") or "node")
        return ["macOS firewall is on: if a dialog asks whether “node” may accept incoming "
                "connections, click Allow. To approve it ahead of time, run once:\n"
                f"      sudo {tool} --add {node} --unblockapp {node}"]
    return []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def svg(qr: segno.QRCode) -> str:
    # omitsize swaps fixed width/height for a viewBox, so CSS can scale it.
    return qr.svg_inline(border=2, dark="#0a0e14", light="#ffffff", omitsize=True)


def join_page(url: str, ssid: str | None, password: str | None) -> str:
    e = html.escape
    steps = []
    if ssid:
        wifi = helpers.make_wifi(ssid=ssid, password=password or None,
                                 security="WPA" if password else "nopass")
        steps.append(f"""
      <section class="step">
        <div class="num">1</div>
        <h2>Join the Wi-Fi</h2>
        <div class="qr">{svg(wifi)}</div>
        <dl>
          <dt>Network</dt><dd>{e(ssid)}</dd>
          {f'<dt>Password</dt><dd>{e(password)}</dd>' if password else ''}
        </dl>
        <p class="note">“No Internet Connection” is expected — stay connected.</p>
      </section>""")
    steps.append(f"""
      <section class="step">
        <div class="num">{len(steps) + 1}</div>
        <h2>Open Signal Lab</h2>
        <div class="qr">{svg(segno.make_qr(url, error="m"))}</div>
        <p class="url">{e(url)}</p>
        <p class="note">Menu → Security → <b>Shared Files</b> to download the encrypted files.</p>
      </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Join Signal Lab</title>
<style>
  :root {{ --bg:#0a0e14; --surface:#111823; --border:#223044; --text:#e8eef6;
           --muted:#94a7bd; --primary:#4d94ff; --accent:#f0a12e; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; min-height:100vh; background:var(--bg); color:var(--text);
          font:16px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          padding:32px 16px; }}
  h1 {{ margin:0 0 4px; font-size:clamp(28px,4vw,44px); letter-spacing:-0.02em; }}
  .lead {{ margin:0 0 32px; color:var(--muted); font-size:clamp(15px,1.6vw,20px); }}
  .steps {{ display:flex; flex-wrap:wrap; gap:28px; justify-content:center; }}
  .step {{ position:relative; background:var(--surface); border:1px solid var(--border);
           border-radius:20px; padding:28px 28px 22px; width:min(400px,100%); text-align:center; }}
  .num {{ position:absolute; top:-18px; left:50%; transform:translateX(-50%);
          width:36px; height:36px; border-radius:50%; background:var(--primary); color:#fff;
          display:grid; place-items:center; font-weight:700; }}
  h2 {{ margin:6px 0 16px; font-size:22px; }}
  .qr svg {{ display:block; width:260px; max-width:100%; height:auto; margin:0 auto;
            border-radius:12px; }}
  dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:16px 0 0;
        text-align:left; font-size:18px; }}
  dt {{ color:var(--muted); }}
  dd {{ margin:0; font-family:ui-monospace, monospace; font-weight:600; word-break:break-all; }}
  .url {{ margin:16px 0 0; font:600 clamp(15px,1.6vw,20px) ui-monospace, monospace;
          color:var(--primary); white-space:nowrap; }}
  .note {{ margin:12px 0 0; color:var(--muted); font-size:14px; }}
  .tips {{ margin-top:32px; max-width:840px; color:var(--muted); font-size:14px; text-align:center; }}
  .tips b {{ color:var(--accent); font-weight:600; }}
</style>
</head>
<body>
  <h1>Signal Lab — live demo</h1>
  <p class="lead">Scan with your phone camera. Everything runs on this laptop; nothing leaves the room.</p>
  <div class="steps">{''.join(steps)}
  </div>
  <p class="tips"><b>Page won’t load?</b> Your phone may be sending traffic over mobile data because
  this Wi-Fi has no internet. On Android, choose “Stay connected” / “Keep Wi-Fi” when asked, or turn
  mobile data off for a moment.</p>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--no-open", action="store_true", help="don't open the join page")
    args = parser.parse_args()

    candidates = lan_addresses()
    ip = primary_address(candidates)
    ssid = os.getenv("WIFI_SSID") or None
    password = os.getenv("WIFI_PASSWORD") or None

    warnings = model_warnings() + firewall_warnings()
    if not ip:
        if any(self_assigned(a) for _, a in candidates):
            warnings.insert(0, "This Mac has only a self-assigned 169.254.x.x address: the router "
                               "did not hand out an IP. Check that DHCP is on in the router.")
        else:
            warnings.insert(0, "This Mac is not connected to any network. Join the router's Wi-Fi.")
    if not ssid:
        warnings.append("No WIFI_SSID in .env.demo — the join page shows only the app QR code.")

    print()
    print("  Signal Lab — LAN demo")
    print("  ──────────────────────────────────────")
    if ip:
        url = f"http://{ip}:{args.port}/"
        print(f"  Phones open:  {url}")
        print(f"  This Mac:     http://localhost:{args.port}/")
        others = [f"{a} ({i})" for i, a in candidates if a != ip and not self_assigned(a)]
        if others:
            print(f"  Other addresses: {', '.join(others)}  — use one of these if phones can't connect")
        print()
        segno.make_qr(url, error="m").terminal(compact=True)

        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(join_page(url, ssid, password), encoding="utf-8")
        print(f"  Join page (show it on the projector): {OUT}")
        if not args.no_open and sys.platform == "darwin":
            subprocess.run(["open", str(OUT)], check=False)

    for warning in warnings:
        print(f"\n  ! {warning}")
    print()
    return 0 if ip else 1


if __name__ == "__main__":
    sys.exit(main())
