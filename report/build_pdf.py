"""Render report_content.py to HTML and print it to PDF with headless Edge (DevTools Protocol)."""
import base64
import html
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websocket

from report_content import BLOCKS, META

HERE = Path(__file__).resolve().parent
HTML_OUT = HERE / "report.html"
PDF_OUT = HERE / "Smart_Solar_Monitoring_Report.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1d2330; margin: 0; }
.cover { height: 257mm; display: flex; flex-direction: column; justify-content: center;
         page-break-after: always; border-left: 6px solid #f0a202; padding-left: 14mm; }
.cover .tag { color: #b36b00; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; font-size: 10pt; }
.cover h1 { font-size: 30pt; line-height: 1.15; margin: 6mm 0 4mm; color: #11305c; border: none; }
.cover .sub { font-size: 14pt; color: #4a5568; margin-bottom: 16mm; }
.cover table { border: none; width: auto; font-size: 11pt; }
.cover td { border: none; padding: 1.5mm 8mm 1.5mm 0; }
.cover td:first-child { color: #6b7280; }
.cover .stack { margin-top: 18mm; font-size: 9.5pt; color: #4a5568; }
.toc { page-break-after: always; }
.toc h1 { border: none; }
.toc ol { padding-left: 6mm; font-size: 11pt; line-height: 2; }
h1 { font-size: 16pt; color: #11305c; margin: 9mm 0 3mm; padding-bottom: 1.5mm;
     border-bottom: 2px solid #f0a202; page-break-after: avoid; }
h2 { font-size: 12.5pt; color: #1f5fa8; margin: 6mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3mm; text-align: justify; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.5mm; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 1mm; font-size: 9.3pt; page-break-inside: avoid; }
th { background: #11305c; color: white; text-align: left; padding: 2mm 2.5mm; font-weight: 600; }
td { border-bottom: 1px solid #d9dee7; padding: 1.8mm 2.5mm; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7fb; }
.caption { font-size: 8.8pt; color: #5b6472; font-style: italic; text-align: center; margin: 1.5mm 0 5mm; }
figure { margin: 3mm 0 2mm; text-align: center; page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e3e7ee; }
figcaption.caption { margin-bottom: 3mm; }
.pagebreak { page-break-before: always; }
"""


def inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    return t


CONTENT_W_MM, MAX_FIG_H_MM = 176, 222


def fig_width_mm(rel: str, pct: float) -> float:
    """Requested width, shrunk if needed so the figure (plus caption) fits on one A4 page."""
    from PIL import Image

    w, h = Image.open(HERE / rel).size
    return min(CONTENT_W_MM * pct / 100, MAX_FIG_H_MM * w / h)


def img_uri(rel: str) -> str:
    p = HERE / rel
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def render() -> str:
    out = []
    out.append(f"""<section class="cover">
      <div class="tag">{inline(META['subtitle'])}</div>
      <h1>{inline(META['title'])}</h1>
      <div class="sub">Finding which solar inverter is losing energy, why, and what to do about it</div>
      <table>
        <tr><td>Student</td><td>{inline(META['student'])}</td></tr>
        <tr><td>Course</td><td>{inline(META['course'])}</td></tr>
        <tr><td>Instructor</td><td>{inline(META['instructor'])}</td></tr>
        <tr><td>Date</td><td>{inline(META['date'])}</td></tr>
      </table>
      <div class="stack">Random Forest · ResNet-18 (CNN) · Gemma 2 Transformer · FastAPI · Streamlit</div>
    </section>""")
    toc = [b[1] for b in BLOCKS if b[0] == "h1"]
    out.append('<section class="toc"><h1>Contents</h1><ol>'
               + "".join(f"<li>{inline(re.sub(r'^\d+\.\s*', '', t))}</li>" for t in toc) + "</ol></section>")
    for b in BLOCKS:
        kind = b[0]
        if kind == "h1":
            out.append(f"<h1>{inline(b[1])}</h1>")
        elif kind == "h2":
            out.append(f"<h2>{inline(b[1])}</h2>")
        elif kind == "p":
            out.append(f"<p>{inline(b[1])}</p>")
        elif kind in ("bullets", "numbered"):
            tag = "ul" if kind == "bullets" else "ol"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in b[1]) + f"</{tag}>")
        elif kind == "table":
            head = "".join(f"<th>{inline(h)}</th>" for h in b[1])
            rows = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in b[2])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
                       f'<div class="caption">{inline(b[3])}</div>')
        elif kind == "figure":
            out.append(f'<figure><img src="{img_uri(b[1])}" style="width:{fig_width_mm(b[1], b[3]):.0f}mm">'
                       f'<figcaption class="caption">{inline(b[2])}</figcaption></figure>')
        elif kind == "pagebreak":
            out.append('<div class="pagebreak"></div>')
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(META['title'])}</title>"
            f"<style>{CSS}</style></head><body>{''.join(out)}</body></html>")


def print_pdf(html_path: Path, pdf_path: Path):
    port = 9334
    proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"])
    try:
        for _ in range(40):
            try:
                page = next(t for t in requests.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
                            if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=180, suppress_origin=True)
        n = 0

        def cmd(method, **params):
            nonlocal n
            n += 1
            ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == n:
                    return m.get("result", {})

        cmd("Page.navigate", url=html_path.as_uri())
        time.sleep(4)
        footer = ("<div style='font-size:8px;width:100%;padding:0 17mm;color:#8a8f98;display:flex;"
                  "justify-content:space-between;font-family:Segoe UI'>"
                  f"<span>{html.escape(META['title'])}</span>"
                  "<span>Page <span class='pageNumber'></span> of <span class='totalPages'></span></span></div>")
        res = cmd("Page.printToPDF", printBackground=True, preferCSSPageSize=True,
                  displayHeaderFooter=True, headerTemplate="<span></span>", footerTemplate=footer)
        pdf_path.write_bytes(base64.b64decode(res["data"]))
        ws.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    HTML_OUT.write_text(render(), encoding="utf-8")
    try:
        print_pdf(HTML_OUT, PDF_OUT)
    finally:
        HTML_OUT.unlink(missing_ok=True)  # temporary; the PDF is the deliverable
    from pypdf import PdfReader
    print(f"saved {PDF_OUT.name}: {len(PdfReader(PDF_OUT).pages)} pages, {PDF_OUT.stat().st_size / 1e6:.1f} MB")
