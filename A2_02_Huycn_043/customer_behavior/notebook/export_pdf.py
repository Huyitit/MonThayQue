"""
Automated PDF Export Utility for E-Commerce Customer Behavior Report.
Converts ecommerce_customer_behavior.ipynb to a publication-grade PDF report
with MathJax equations, embedded plots, visual formatting, and
print-optimized CSS to prevent code truncation and page splitting.
"""
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path


def export_notebook_to_pdf():
    notebook_dir = Path(__file__).parent.resolve()
    notebook_path = notebook_dir / "ecommerce_customer_behavior.ipynb"
    report_dir = notebook_dir.parents[1] / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    target_pdf_notebook = notebook_dir / "ecommerce_customer_behavior.pdf"
    target_pdf_report = report_dir / "ecommerce_customer_behavior_report.pdf"

    temp_html = Path("/tmp/customer_behavior_report_temp.html")
    temp_pdf = Path("/tmp/customer_behavior_report_temp.pdf")

    print("=" * 68)
    print("   E-COMMERCE CUSTOMER BEHAVIOR NOTEBOOK -> PDF EXPORTER")
    print("=" * 68)

    # 1. Resolve preferred jupyter / python binary (prefer anaconda jupyter for MathJax CDN templates)
    conda_jupyter = Path("/home/huycao/anaconda3/bin/jupyter")
    workspace_dir = notebook_dir.parents[2]
    venv_python = workspace_dir / ".venv" / "bin" / "python"

    if conda_jupyter.exists():
        cmd_html = [
            str(conda_jupyter), "nbconvert",
            "--to", "html",
            "--embed-images",
            str(notebook_path),
            "--output", str(temp_html)
        ]
    elif venv_python.exists():
        cmd_html = [
            str(venv_python), "-m", "nbconvert",
            "--to", "html",
            "--embed-images",
            str(notebook_path),
            "--output", str(temp_html)
        ]
    else:
        cmd_html = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--embed-images",
            str(notebook_path),
            "--output", str(temp_html)
        ]

    print(f"\n[1/4] Compiling notebook to self-contained HTML via {' '.join(cmd_html[:3])}...")
    subprocess.run(cmd_html, check=True)
    print("      ✓ HTML generated successfully.")

    # 2. Post-process HTML for MathJax CDN & Print CSS
    print("\n[2/4] Injecting MathJax CDN and print-optimized pagination CSS...")
    with open(temp_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Convert MathJax to SVG renderer (TeX-AMS_SVG) so headless Chrome embeds vector glyph paths without relying on external webfonts
    html_content = html_content.replace("config=TeX-AMS_CHTML-full,Safe", "config=TeX-AMS_SVG,Safe")
    html_content = html_content.replace("config=TeX-AMS_CHTML", "config=TeX-AMS_SVG")
    html_content = re.sub(
        r'src="file:///usr/share/javascript/mathjax/[^"]+"',
        'src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_SVG,Safe"',
        html_content
    )

    mathjax_extra = ""
    if "mathjax" not in html_content.lower():
        mathjax_extra = '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_SVG,Safe"></script>'

    custom_css = f"""
{mathjax_extra}
<style type="text/css">
  @media print, screen {{
    /* Code block styling & wrapping */
    pre, code, .highlight pre, .jp-CodeCell pre, .input_area pre, div.input_area {{
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      font-size: 8.5pt !important;
      line-height: 1.35 !important;
      max-width: 100% !important;
    }}
    .jp-Cell, .cell {{
      padding: 4px 6px !important;
    }}
    /* Page layout and margins */
    @page {{
      size: A4 portrait;
      margin: 1.4cm 1.2cm !important;
    }}
    /* Prevent orphaned headings at page bottom */
    h1, h2, h3, h4, h5, h6 {{
      page-break-after: avoid !important;
      break-after: avoid !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }}
    .cell:has(h1), .cell:has(h2), .cell:has(h3), .cell:has(h4),
    .jp-Cell:has(h1), .jp-Cell:has(h2), .jp-Cell:has(h3), .jp-Cell:has(h4) {{
      page-break-after: avoid !important;
      break-after: avoid !important;
    }}
    /* Prevent figure and caption splitting across pages */
    .figure-card, .figure-container, figure {{
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      margin: 12px 0 !important;
    }}
    img {{
      max-width: 86% !important;
      max-height: 440px !important;
      height: auto !important;
      display: block !important;
      margin: 8px auto !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }}
    /* Table layout and pagination */
    table, .dataframe {{
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      font-size: 8.5pt !important;
      margin: 10px 0 !important;
    }}
    .output_subarea {{
      max-width: 100% !important;
      overflow-x: hidden !important;
    }}
  }}
</style>
"""

    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{custom_css}\n</head>")
    else:
        html_content = f"{custom_css}\n{html_content}"

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("      ✓ MathJax CDN & custom print styles injected.")

    # 3. Headless Chrome print-to-pdf with MathJax rendering time budget
    chrome_bin = shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome"
    print(f"\n[3/4] Rendering PDF via headless browser ({chrome_bin})...")
    cmd_pdf = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={temp_pdf}",
        str(temp_html)
    ]
    subprocess.run(cmd_pdf, check=True)
    print("      ✓ PDF rendered successfully.")

    # 4. Save to destination paths
    print("\n[4/4] Saving final PDF deliverables...")
    shutil.copyfile(temp_pdf, target_pdf_notebook)
    shutil.copyfile(temp_pdf, target_pdf_report)

    # Clean up temp files
    if temp_html.exists():
        temp_html.unlink()
    if temp_pdf.exists():
        temp_pdf.unlink()

    # Audit outputs
    pdf_size_mb = os.path.getsize(target_pdf_notebook) / (1024 * 1024)
    print("-" * 68)
    print("                   PDF GENERATION COMPLETE")
    print("-" * 68)
    print(f"  • Notebook PDF Location : {target_pdf_notebook}")
    print(f"  • Report PDF Location   : {target_pdf_report}")
    print(f"  • Deliverable Size      : {pdf_size_mb:.2f} MB")
    print("=" * 68)


if __name__ == "__main__":
    export_notebook_to_pdf()
