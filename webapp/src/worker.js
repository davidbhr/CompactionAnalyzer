/* Python runs off the UI thread. Image data never leaves this worker/browser. */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let engine;
function progress(message) { self.postMessage({type: "progress", message}); }
async function boot() {
  progress("Loading Python for your browser…");
  importScripts(PYODIDE_URL + "pyodide.js");
  engine = await loadPyodide({indexURL: PYODIDE_URL});
  progress("Loading image analysis packages…");
  await engine.loadPackage(["numpy", "scipy", "scikit-image", "pandas", "matplotlib", "pillow", "micropip", "pyyaml"]);
  await engine.runPythonAsync(`
import micropip
await micropip.install(["natsort==8.4.0", "matplotlib-scalebar==0.9.0", "tqdm==4.67.1", "tifffile==2024.8.30"])
import matplotlib
matplotlib.use("Agg")
`);
  progress("Preparing CompactionAnalyzer…");
  const [source, bridge] = await Promise.all([fetch("analysis.zip"), fetch("bridge.py")]);
  if (!source.ok || !bridge.ok) throw new Error("The analysis files could not be loaded. Reload this page.");
  engine.unpackArchive(await source.arrayBuffer(), "zip");
  const desktopExport = await fetch("saenopy_export.py");
  if (!desktopExport.ok) throw new Error("The export adapter could not be loaded.");
  engine.FS.writeFile("saenopy_export.py", await desktopExport.text());
  const calibration = await fetch("calibration.py");
  if (!calibration.ok) throw new Error("The image calibration module could not be loaded.");
  engine.FS.writeFile("calibration.py", await calibration.text());
  const comparison = await fetch("comparison.py");
  if (!comparison.ok) throw new Error("The comparison module could not be loaded.");
  engine.FS.writeFile("comparison.py", await comparison.text());
  await engine.runPythonAsync(await bridge.text());
  await engine.runPythonAsync("from comparison import import_request, comparison_request");
  self.postMessage({type: "ready"});
}
const ready = boot().catch(error => { self.postMessage({type: "fatal", message: String(error)}); throw error; });
let queue = Promise.resolve();
self.onmessage = ({data}) => {
  queue = queue.then(async () => {
    const {id, command, payload = {}, bytes} = data;
    try {
      await ready;
      if (bytes) engine.FS.writeFile("/tmp/incoming-image", new Uint8Array(bytes));
      engine.globals.set("web_request_json", JSON.stringify({command, ...payload}));
      if (command === "export") {
        const proxy = await engine.runPythonAsync("export_result(web_request_json)");
        const data = proxy.toJs();
        proxy.destroy();
        self.postMessage({id, data}, [data.buffer]);
      } else if (command === "compare_import") {
        try {
          const result = await engine.runPythonAsync("import_request(web_request_json)");
          self.postMessage({id, data: JSON.parse(result)});
        } finally { engine.FS.unlink("/tmp/incoming-image"); }
      } else if (command === "compare_plot") {
        const result = await engine.runPythonAsync("comparison_request(web_request_json)");
        self.postMessage({id, data: JSON.parse(result)});
      } else {
        const result = await engine.runPythonAsync("handle_request(web_request_json)");
        self.postMessage({id, data: JSON.parse(result)});
      }
    } catch (error) { self.postMessage({id, error: String(error)}); }
  });
};
