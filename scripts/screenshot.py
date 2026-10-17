"""Screenshot the running dashboard for the website.

    python scripts/screenshot.py [url] [out.png] [width] [light|dark] [max height]

Needs `pip install playwright` and a Chromium (`playwright install chromium`, or point
CHROMIUM_PATH at an existing Chromium binary).
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
out = sys.argv[2] if len(sys.argv) > 2 else "site/assets/dashboard.png"
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
scheme = sys.argv[4] if len(sys.argv) > 4 else "light"
max_height = int(sys.argv[5]) if len(sys.argv) > 5 else 4200

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
    page = browser.new_page(viewport={"width": width, "height": 4200}, device_scale_factor=2, color_scheme=scheme)
    page.goto(url)
    page.wait_for_selector("text=Order list", timeout=60_000)
    time.sleep(4)  # let the charts finish drawing
    # Streamlit scrolls inside its own container, so use a tall viewport and crop to the content.
    # Crop to the content column (left/right minus the container's own padding), plus a margin.
    box = page.evaluate("""(() => {
        const el = document.querySelector('[data-testid=stMainBlockContainer]');
        const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
        return {left: r.left + parseFloat(cs.paddingLeft), right: r.right - parseFloat(cs.paddingRight),
                top: r.top + parseFloat(cs.paddingTop), bottom: r.bottom};
    })()""")
    margin = 24
    x, y = max(0, box["left"] - margin), max(0, box["top"] - margin)
    page.screenshot(path=out, clip={"x": x, "y": y, "width": box["right"] + margin - x,
                                    "height": min(box["bottom"], max_height) - y})
    browser.close()

# Fewer colours, same crisp text: roughly halves the file size for the website.
from PIL import Image  # noqa: E402  (Pillow comes with matplotlib)

Image.open(out).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
print(f"Saved {out}")
