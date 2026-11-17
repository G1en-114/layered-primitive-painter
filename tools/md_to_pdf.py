"""Render a project Markdown document (Chinese text, tables, mermaid diagrams) to an A4 PDF.

    python tools/md_to_pdf.py docs/architecture.md --mermaid /path/to/mermaid.min.js --out docs/architecture.pdf

Markdown -> HTML (python-markdown: tables, fenced code), ```mermaid blocks -> <pre class="mermaid">,
print CSS with CJK fonts, then headless Chromium (Playwright, system chromium) waits until every
mermaid diagram has rendered to SVG and prints the page with page numbers in the footer.
The mermaid script is inlined from a local file, so rendering needs no network.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
:root { --fg: #1d2330; --muted: #5d6576; --rule: #d9dde4; --accent: #1f6f8b; --code: #f3f5f8; }
body { font-family: "Noto Sans CJK SC", "Source Han Sans CN", "Noto Sans CJK TC", sans-serif; color: var(--fg);
       font-size: 10.5pt; line-height: 1.72; margin: 0; }
h1, h2, h3 { font-family: "Noto Serif CJK SC", "Source Han Serif CN", serif; line-height: 1.35; break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 6mm; padding-bottom: 3mm; border-bottom: 2px solid var(--accent); }
h2 { font-size: 15pt; margin: 9mm 0 3mm; color: var(--accent); }
h3 { font-size: 12pt; margin: 6mm 0 2mm; }
p { margin: 0 0 2.6mm; text-align: justify; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin: 0.6mm 0; }
li > ul, li > ol { margin: 1mm 0; }
strong { font-weight: 700; color: #111827; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 7mm 0; }
code { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 9pt; background: var(--code);
       padding: 0.2mm 1.2mm; border-radius: 2px; }
pre { background: var(--code); padding: 3mm; border-radius: 3px; font-size: 8.8pt; overflow-wrap: anywhere;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
pre.mermaid { background: none; text-align: center; padding: 2mm 0; margin: 3mm 0 5mm; }
pre.mermaid svg { max-width: 100%; max-height: 150mm; height: auto; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 5mm; font-size: 9.6pt; break-inside: avoid; }
th, td { border: 1px solid var(--rule); padding: 1.6mm 2.4mm; vertical-align: top; }
th { background: #eef2f6; font-weight: 700; text-align: left; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background: #fafbfc; }
"""


def build_html(md_text: str, mermaid_js: str, title: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    # fenced_code renders ```mermaid as <pre><code class="language-mermaid">…</code></pre>
    body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: f'<pre class="mermaid">{m.group(1)}</pre>', body, flags=re.S)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style><script>{mermaid_js}</script></head><body>{body}
<script>
mermaid.initialize({{startOnLoad: false, theme: "neutral", flowchart: {{htmlLabels: true, curve: "basis"}},
  themeVariables: {{fontFamily: "Noto Sans CJK SC, sans-serif", fontSize: "14px"}}}});
mermaid.run({{querySelector: "pre.mermaid"}}).then(() => {{ document.body.dataset.rendered = "1"; }})
  .catch(e => {{ document.body.dataset.rendered = "error: " + e; }});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--mermaid", type=Path, required=True, help="local mermaid.min.js (pinned version)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chromium", default="/usr/bin/chromium")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    md_text = args.markdown.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in md_text.splitlines() if line.startswith("# ")), args.markdown.stem)
    page_html = build_html(md_text, args.mermaid.read_text(encoding="utf-8"), title)
    html_path = args.out.with_suffix(".html")
    html_path.write_text(page_html, encoding="utf-8")
    footer = ('<div style="width:100%;font-size:8px;color:#8a92a3;padding:0 16mm;display:flex;justify-content:space-between;'
              f'font-family:Noto Sans CJK SC,sans-serif"><span>{html.escape(title)}</span>'
              '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.wait_for_function("document.body.dataset.rendered !== undefined", timeout=60_000)
        status = page.evaluate("document.body.dataset.rendered")
        if status != "1":
            raise SystemExit(f"mermaid failed: {status}")
        page.pdf(path=str(args.out), format="A4", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=footer,
                 margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"})
        browser.close()
    html_path.unlink()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
