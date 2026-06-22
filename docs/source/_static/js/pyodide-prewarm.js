// Background-warm the Pyodide runtime and scientific stack on documentation pages that
// expose interactive examples ("Try it live" / "Launch interactive notebook"), so that
// launching one is near-instant. Assets are prefetched at low priority during browser idle
// time and reused from the browser cache when JupyterLite later loads them. The CDN base URL
// and the gwtransport wheel URL are injected by the generated pyodide-prewarm-config.js.
(function () {
  "use strict";

  var cfg = window.gwtPrewarmConfig || {};
  var PYODIDE_BASE = cfg.pyodideBase;
  var GWT_WHEEL = cfg.gwtWheel;
  if (!PYODIDE_BASE) {
    return;
  }

  // Always-loaded runtime; pyodide.asm.wasm and python_stdlib.zip dominate the payload.
  var CORE = [
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
  ];
  // Scientific stack gwtransport needs; resolved to concrete wheel names via pyodide-lock.json.
  var PACKAGES = [
    "numpy",
    "scipy",
    "openblas",
    "pandas",
    "matplotlib",
    "mpmath",
  ];

  function isInteractivePage() {
    return !!document.querySelector(
      ".try_examples_button, .jupyterlite_sphinx_try_it_button",
    );
  }

  function prefetch(href, crossOrigin) {
    if (
      !href ||
      document.querySelector('link[data-prewarm][href="' + href + '"]')
    ) {
      return;
    }
    var link = document.createElement("link");
    link.rel = "prefetch";
    link.href = href;
    link.setAttribute("data-prewarm", "");
    // CDN assets are fetched by Pyodide in CORS mode; match that so the cache entry is reused.
    // The same-origin gwtransport wheel must stay non-CORS for the same reason.
    if (crossOrigin) {
      link.crossOrigin = "anonymous";
    }
    document.head.appendChild(link);
  }

  function warm() {
    CORE.forEach(function (file) {
      prefetch(PYODIDE_BASE + file, true);
    });
    prefetch(GWT_WHEEL, false);
    fetch(PYODIDE_BASE + "pyodide-lock.json")
      .then(function (response) {
        return response.json();
      })
      .then(function (lock) {
        var packages = (lock && lock.packages) || {};
        PACKAGES.forEach(function (name) {
          var entry = packages[name];
          if (entry && entry.file_name) {
            prefetch(PYODIDE_BASE + entry.file_name, true);
          }
        });
      })
      .catch(function () {
        /* Offline or CDN unreachable: launching still works, just without the warm cache. */
      });
  }

  function schedule() {
    if (!isInteractivePage()) {
      return;
    }
    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(warm, { timeout: 5000 });
    } else {
      window.setTimeout(warm, 2000);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", schedule);
  } else {
    schedule();
  }
})();
