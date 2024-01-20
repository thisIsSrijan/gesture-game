// importScripts("https://cdn.jsdelivr.net/pyodide/v0.18.1/full/pyodide.js");

async function loadPyodide() {
    await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/",
    });
    console.log(pyodide.runPython('print("Hello from Python!")'));
}

loadPyodide();
