"""
Automated PDF Export Utility for Diabetes Prediction Report.
Converts diabetes_prediction.ipynb to a publication-grade PDF report
with MathJax equations, embedded plots, visual screenshots, and
print-optimized CSS to prevent code truncation.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path


def export_notebook_to_pdf():
    notebook_dir = Path(__file__).parent.resolve()
    notebook_path = notebook_dir / "diabetes_prediction.ipynb"
    report_dir = notebook_dir.parents[1] / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    target_pdf_notebook = notebook_dir / "diabetes_prediction.pdf"
    target_pdf_report = report_dir / "diabetes_prediction_report.pdf"

    temp_html = Path("/tmp/diabetes_report_temp.html")
    temp_pdf = Path("/tmp/diabetes_report_temp.pdf")

    print("=" * 68)
    print("           DIABETES PREDICTION NOTEBOOK -> PDF EXPORTER")
    print("=" * 68)

    # 1. Jupyter nbconvert to HTML with embedded images
    jupyter_bin = "/home/huycao/anaconda3/bin/jupyter"
    if not os.path.exists(jupyter_bin):
        jupyter_bin = shutil.which("jupyter") or "jupyter"

    print(f"\n[1/4] Compiling notebook to self-contained HTML via {jupyter_bin}...")
    cmd_html = [
        jupyter_bin,
        "nbconvert",
        "--to", "html",
        "--embed-images",
        str(notebook_path),
        "--output", str(temp_html)
    ]
    subprocess.run(cmd_html, check=True)
    print("      ✓ HTML generated successfully.")

    # 2. Inject print CSS for complete code visibility
    print("\n[2/4] Injecting print-optimized CSS for line-wrapping & typography...")
    with open(temp_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    custom_css = """
<style type="text/css">
  @media print, screen {
    /* Enforce line wrapping on code blocks to prevent right-margin clipping */
    pre, code, .highlight pre, .jp-CodeCell pre, .input_area pre, div.input_area {
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      font-size: 8.5pt !important;
      line-height: 1.35 !important;
      max-width: 100% !important;
    }
    .jp-Cell, .cell {
      padding: 4px 6px !important;
    }
    /* Page layout and margins */
    @page {
      size: A4 portrait;
      margin: 1.5cm 1.2cm !important;
    }
    img {
      max-width: 95% !important;
      height: auto !important;
      display: block !important;
      margin: 10px auto !important;
      page-break-inside: avoid !important;
    }
    .output_subarea {
      max-width: 100% !important;
      overflow-x: hidden !important;
    }
  }
</style>
"""
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{custom_css}\n</head>")
    else:
        html_content = f"{custom_css}\n{html_content}"

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("      ✓ Custom print styles injected.")

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
