"""Browser runtime for rendering D3 programs and extracting SVG object IR."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from playwright.sync_api import Browser, Error as PlaywrightError, sync_playwright


_CHROME_CANDIDATES = (
    Path(os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/chromium"),
)

# The dataset ships standalone D3 `.js` programs; render them by wrapping in a
# minimal HTML document. Prefer the local d3 bundle (npm install), fall back to CDN.
_D3_DIST_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "node_modules" / "d3" / "dist" / "d3.min.js",
)
_D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
_WRAPPER_DIR = Path(tempfile.mkdtemp(prefix="chartanno_d3_html_"))


def _d3_script_uri() -> str:
    for candidate in _D3_DIST_CANDIDATES:
        if candidate.is_file():
            return candidate.as_uri()
    return _D3_CDN


def _materialize_html(js_file: Path) -> Path:
    """Wrap a standalone D3 program into a temporary HTML document.

    The program runs inside an IIFE so top-level `const`/`let` names (e.g. a
    chart-local `top`) never collide with read-only window properties.
    """
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<script src="{_d3_script_uri()}"></script>'
        "</head><body><script>\n(function(){\n"
        + js_file.read_text(encoding="utf-8")
        + "\n})();\n</script></body></html>"
    )
    wrapper = _WRAPPER_DIR / (js_file.stem + ".html")
    wrapper.write_text(html, encoding="utf-8")
    return wrapper


def _find_browser_executable() -> Optional[str]:
    for candidate in _CHROME_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


class D3BrowserRuntime:
    """Own one headless browser and render one or more D3 HTML files."""

    def __init__(self, *, timeout_ms: int = 15_000):
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._extractor_path = Path(__file__).with_name("browser_extractor.js")

    def __enter__(self) -> "D3BrowserRuntime":
        self._playwright = sync_playwright().start()
        executable = _find_browser_executable()
        launch_kwargs = {
            "headless": True,
            "args": ["--allow-file-access-from-files", "--disable-web-security"],
        }
        if executable:
            launch_kwargs["executable_path"] = executable
        self._browser = self._playwright.chromium.launch(**launch_kwargs)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def extract(self, html_file: str | Path, *, screenshot_path: str | Path | None = None) -> dict:
        if self._browser is None:
            raise RuntimeError("D3BrowserRuntime must be used as a context manager")

        html_path = Path(html_file).resolve()
        if not html_path.is_file():
            raise FileNotFoundError(html_path)
        if html_path.suffix == ".js":
            html_path = _materialize_html(html_path)

        page = self._browser.new_page(viewport={"width": 1280, "height": 960}, device_scale_factor=1)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        try:
            page.goto(html_path.as_uri(), wait_until="load", timeout=self.timeout_ms)
            page.wait_for_function(
                "window.__chartReady === true || document.querySelector('svg[data-chart-root], svg') !== null",
                timeout=self.timeout_ms,
            )
            # Evaluate the extractor directly so the same runtime works for
            # both HTML documents and standalone XML/SVG documents.
            page.evaluate(self._extractor_path.read_text(encoding="utf-8"))
            result = page.evaluate("window.__extractD3IR()")
            if screenshot_path:
                output = Path(screenshot_path).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                chart = page.locator("svg[data-chart-root], svg").first
                chart.screenshot(path=str(output))
            if errors:
                result["browser_warnings"] = errors
            result["source_file"] = str(html_path)
            return result
        except PlaywrightError as exc:
            detail = f"; browser errors: {' | '.join(errors)}" if errors else ""
            raise RuntimeError(f"Failed to render {html_path}: {exc}{detail}") from exc
        finally:
            page.close()

    def render_bbox_overlay(
        self,
        html_file: str | Path,
        records: list[dict],
        output_path: str | Path,
        *,
        title: str = "Extracted bboxes",
    ) -> None:
        """Render a chart with normalized IR bounding boxes overlaid on its SVG."""
        if self._browser is None:
            raise RuntimeError("D3BrowserRuntime must be used as a context manager")

        html_path = Path(html_file).resolve()
        if html_path.suffix == ".js":
            html_path = _materialize_html(html_path)
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        page = self._browser.new_page(viewport={"width": 1280, "height": 960}, device_scale_factor=1)
        try:
            page.goto(html_path.as_uri(), wait_until="load", timeout=self.timeout_ms)
            page.wait_for_function(
                "window.__chartReady === true || document.querySelector('svg[data-chart-root], svg') !== null",
                timeout=self.timeout_ms,
            )
            page.evaluate(
                """
                ({records, title}) => {
                  const svg = document.querySelector('svg[data-chart-root], svg');
                  if (!svg) throw new Error('No SVG chart root found');
                  const NS = 'http://www.w3.org/2000/svg';
                  const viewBox = svg.viewBox && svg.viewBox.baseVal;
                  const width = viewBox && viewBox.width ? viewBox.width : svg.getBoundingClientRect().width;
                  const height = viewBox && viewBox.height ? viewBox.height : svg.getBoundingClientRect().height;
                  const offsetX = viewBox ? viewBox.x : 0;
                  const offsetY = viewBox ? viewBox.y : 0;
                  const palette = {
                    axes: '#7c3aed', text: '#dc2626', line: '#0284c7', figure_line: '#0284c7',
                    collection: '#059669', patch: '#ea580c', annotation_bbox: '#db2777',
                    annotation_arrow: '#2563eb', table_cell: '#ca8a04'
                  };
                  const overlay = document.createElementNS(NS, 'g');
                  overlay.setAttribute('data-bbox-overlay', 'true');
                  overlay.setAttribute('pointer-events', 'none');
                  svg.appendChild(overlay);

                  for (const record of records) {
                    const box = record && record.bbox;
                    if (!Array.isArray(box) || box.length !== 4) continue;
                    const x = offsetX + Number(box[0]) * width;
                    const y = offsetY + (1 - Number(box[1]) - Number(box[3])) * height;
                    const w = Math.max(Number(box[2]) * width, 2);
                    const h = Math.max(Number(box[3]) * height, 2);
                    if (![x, y, w, h].every(Number.isFinite)) continue;
                    const color = palette[record.kind] || '#dc2626';
                    const rect = document.createElementNS(NS, 'rect');
                    rect.setAttribute('x', x);
                    rect.setAttribute('y', y);
                    rect.setAttribute('width', w);
                    rect.setAttribute('height', h);
                    rect.setAttribute('fill', 'none');
                    rect.setAttribute('stroke', color);
                    rect.setAttribute('stroke-width', '1.5');
                    rect.setAttribute('vector-effect', 'non-scaling-stroke');
                    overlay.appendChild(rect);

                    const label = document.createElementNS(NS, 'text');
                    label.setAttribute('x', x + 2);
                    label.setAttribute('y', Math.max(9, y - 2));
                    label.setAttribute('fill', color);
                    label.setAttribute('stroke', 'white');
                    label.setAttribute('stroke-width', '2.5');
                    label.setAttribute('paint-order', 'stroke');
                    label.setAttribute('font-size', '8');
                    label.setAttribute('font-family', 'Arial, sans-serif');
                    label.textContent = `${record.element_id || record.artist_id}:${record.kind}`;
                    overlay.appendChild(label);
                  }

                  const heading = document.createElementNS(NS, 'text');
                  heading.setAttribute('x', 8);
                  heading.setAttribute('y', 14);
                  heading.setAttribute('fill', '#111827');
                  heading.setAttribute('stroke', 'white');
                  heading.setAttribute('stroke-width', '3');
                  heading.setAttribute('paint-order', 'stroke');
                  heading.setAttribute('font-size', '11');
                  heading.setAttribute('font-family', 'Arial, sans-serif');
                  heading.textContent = `${title} (${records.length})`;
                  overlay.appendChild(heading);
                }
                """,
                {"records": records, "title": title},
            )
            page.locator("svg[data-chart-root], svg").first.screenshot(path=str(output))
        finally:
            page.close()


__all__ = ["D3BrowserRuntime"]
