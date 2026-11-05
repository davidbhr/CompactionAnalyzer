"use strict";
const $ = id => document.getElementById(id);
const state = {ready:false, busy:false, images:{}, blobs:{}, point:null, line:[], overlay:null,
  result:null, axis:"x", revision:0, manifest:null, worker:null, requests:new Map(), nextId:0};
let maskTimer, bootTimer;
const labels = {full:["Cell-relative", "Cell-relative alignment", "cell"], axis:["X / Y axes", "Orientation relative to X and Y", "axis"],
  point_orientation:["Point reference", "Orientation towards a point", "reference"], line_orientation:["Line reference", "Orientation relative to a line", "reference"]};
function mode() { return document.querySelector('input[name="mode"]:checked').value; }
function setStatus(message, loading=false, failed=false) { $("status").textContent=message; $("status-dot").className="status-dot"+(loading?" loading":"")+(failed?" failed":""); document.dispatchEvent(new Event("analysis-status-change")); }
function errorMessage(error) { console.error(error); const text=String(error.message || error); return text.split("\n").filter(Boolean).at(-1).replace(/^(ValueError|RuntimeError|Error):\s*/, ""); }
function showError(error) { $("error").textContent=errorMessage(error); $("error").hidden=false; }
function clearDownloadError() { $("download-error").hidden=true; $("download-error").textContent=""; $("pixel-size").removeAttribute("aria-invalid"); }
function showDownloadError(message) { $("download-error").textContent=message; $("download-error").hidden=false; }
function clearError() { $("error").hidden=true; clearDownloadError(); }
function hasMaskInput() {
  if(!state.images.orientation)return false;
  const region=mode()==="full"?"second":$("region").value;
  return region==="same" || (region==="second" && !!state.images.mask);
}
function updateButtons() {
  document.querySelectorAll(".settings input,.settings select,.settings button,#example,#load-example").forEach(el=>el.disabled=state.busy || !state.ready);
  $("cancel").disabled=false;
  $("analyze").disabled=!state.ready || state.busy || !state.images.orientation;
  const maskDisabled=!state.ready || state.busy || !hasMaskInput();
  ["threshold","threshold-range","preview-mask"].forEach(id=>$(id).disabled=maskDisabled);
  $("threshold-control").classList.toggle("is-disabled",maskDisabled);
  ["save-map","save-histogram","save-numpy","save-all","save-saenopy"].forEach(id=>$(id).disabled=!state.result || state.busy);
  document.dispatchEvent(new Event("analysis-state-change"));
}
function setBusy(value, message) { state.busy=value; updateButtons(); if(message)setStatus(message,value); }
function invalidate() {
  clearTimeout(maskTimer); state.revision++; state.result=null; state.overlay=null;
  $("results").hidden=true; $("before-results").hidden=false; $("overlay-control").hidden=true;
  clearError(); drawInput(); updateButtons();
  if(state.ready&&!state.busy)setStatus("Ready to analyze");
}
function workerCall(command,payload={},bytes) {
  return new Promise((resolve,reject)=>{
    const id=++state.nextId;
    const timeout=setTimeout(()=>{state.requests.delete(id);reject(new Error("This operation took too long. Cancel and try a smaller image."));},180000);
    state.requests.set(id,{resolve,reject,timeout});
    state.worker.postMessage({id,command,payload,bytes},bytes?[bytes]:[]);
  });
}
function startEngine() {
  if(state.worker)state.worker.terminate();
  for(const entry of state.requests.values()){clearTimeout(entry.timeout);entry.reject(new Error("Analysis cancelled."));}
  state.requests.clear(); state.ready=false; state.busy=false; state.images={}; state.blobs={};state.point=null;state.line=[];
  invalidate(); $("orientation-name").textContent="Drop image or browse"; $("mask-name").textContent="Choose image";
  $("preview-caption").textContent="";$("image-dimensions").textContent="TIFF, PNG or JPEG";
  $("retry").hidden=true; $("cancel").hidden=true;
  setStatus("Loading analysis engine…",true);
  const worker=new Worker("worker.js");state.worker=worker;
  clearTimeout(bootTimer);
  bootTimer=setTimeout(()=>{if(!state.ready){$("retry").hidden=false;updateButtons();}},90000);
  worker.onmessage=({data})=>{
    if(data.type==="progress"){setStatus(data.message,true);return;}
    if(data.type==="ready"){clearTimeout(bootTimer);state.ready=true;setStatus("Ready");updateButtons();return;}
    if(data.type==="fatal"){clearTimeout(bootTimer);setStatus("Could not load the analysis engine",false,true);showError(data.message);$("retry").hidden=false;updateButtons();return;}
    const entry=state.requests.get(data.id); if(!entry)return;
    clearTimeout(entry.timeout);state.requests.delete(data.id);
    if(data.error)entry.reject(new Error(data.error));else entry.resolve(data.data);
  };
  worker.onerror=event=>{setStatus("The analysis engine stopped",false,true);showError(event.message || "Reload to restart the analysis engine.");$("retry").hidden=false;setBusy(false);};
  updateButtons();updateMode(false);
}
function configuration() {
  for(const id of ["sigma","edge","blur","threshold","cell-blur","background-blur","mask-blur"]){if(!$(id).checkValidity()){$(id).reportValidity();throw new Error("Check the analysis settings.");}}
  return {mode:mode(),region:mode()==="full"?"second":$("region").value,
    sigma:Number($("sigma").value),edge:Number($("edge").value),blur:Number($("blur").value),
    threshold:Number($("threshold").value),mask_blur:Number($("mask-blur").value),cell_blur:Number($("cell-blur").value),
    background_blur:Number($("background-blur").value),invert:$("invert").checked,
    point:state.point,line:state.line.length===2?state.line:null};
}
function updateMode(resetThreshold=true) {
  const current=mode(),isCell=current==="full",hasMask=isCell||$("region").value!=="whole";
  $("region-control").hidden=isCell;$("second-control").hidden=!isCell&&$("region").value!=="second";
  $("second-label").textContent=isCell?"Cell image":"Mask image";
  $("cell-settings").hidden=!isCell;$("mask-blur-field").hidden=isCell||!hasMask;
  $("invert-control").hidden=!hasMask;
  $("reference-control").hidden=!["point_orientation","line_orientation"].includes(current);
  $("reference-hint").textContent=current==="point_orientation"?"Click a point in the image.":"Click two points in the image.";
  $("input-canvas").classList.toggle("picking",!$("reference-control").hidden);
  $("mode-badge").textContent=labels[current][0];$("mode-badge").className="badge "+labels[current][2];
  $("mask-status").textContent=isCell?"Cell-relative analysis measures alignment around the segmented cell.":hasMask?"The mask selects the area used for orientation measurements.":"Orientation is measured across the whole image.";
  if(resetThreshold){$("threshold").value=isCell?1:.5;$("threshold-range").value=$("threshold").value;}
  renderCoordinates();drawInput();
}
function renderCoordinates() {
  const host=$("reference-values");host.replaceChildren();
  const count=mode()==="point_orientation"?1:mode()==="line_orientation"?2:0;
  for(let i=0;i<count;i++){
    const point=count===1?state.point:state.line[i];
    for(let axis=0;axis<2;axis++){
      const label=document.createElement("label");label.textContent=(count===2?`P${i+1} `:"")+(axis===0?"x ":"y ");
      const input=document.createElement("input");input.type="number";input.min="0";input.step="0.5";input.style.width="60px";input.style.fontSize="10px";
      if(state.images.orientation)input.max=String((axis===0?state.images.orientation.width:state.images.orientation.height)-1);
      input.value=point?point[axis]:"";
      input.addEventListener("change",()=>{
        if(!input.checkValidity()){input.reportValidity();return;}
        const fields=[...host.querySelectorAll("input")];const coordinates=[];
        for(let j=0;j<count;j++)if(fields[2*j].value!==""&&fields[2*j+1].value!=="")coordinates.push([Number(fields[2*j].value),Number(fields[2*j+1].value)]);
        if(count===1)state.point=coordinates[0]||null;else state.line=coordinates;
        invalidate();drawInput();
      });
      label.append(input);host.append(label);
    }
  }
}
function loadBitmap(data) { return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error("Image preview failed."));image.src="data:image/png;base64,"+data;}); }
async function loadFile(file,slot) {
  if(file.size>32*1024*1024)throw new Error("Choose an image smaller than 32 MB for this browser demo.");
  delete state.images[slot];delete state.blobs[slot];drawInput();
  const data=await workerCall("load",{slot,name:file.name},await file.arrayBuffer());
  const image=await loadBitmap(data.preview);
  state.images[slot]={...data,image,name:file.name};state.blobs[slot]=file;
  if(slot==="orientation"){
    state.point=null;state.line=[];$("orientation-name").textContent=file.name;
    $("preview-caption").textContent=file.name;$("image-dimensions").textContent=`${data.width} × ${data.height} px`;
  }else $("mask-name").textContent=file.name;
  drawInput();renderCoordinates();
}
async function chooseFile(file,slot) {
  if(!file||state.busy||!state.ready)return;
  invalidate();setBusy(true,"Reading image…");
  try{
    if(slot==="orientation"){
      $("pixel-size").value="";
      delete state.images.mask;delete state.blobs.mask;await workerCall("clear",{slot:"mask"});$("mask-name").textContent="Choose an image on the same pixel grid";
    }
    await loadFile(file,slot);setStatus("Image loaded · ready to analyze");
  }catch(error){showError(error);setStatus("Check the selected image",false,true);}finally{setBusy(false);}
}
function drawInput() {
  const canvas=$("input-canvas"),item=state.images.orientation;
  canvas.hidden=!item;$("empty-state").hidden=!!item;
  if(!item)return;
  canvas.width=item.width;canvas.height=item.height;
  const ctx=canvas.getContext("2d");ctx.drawImage(item.image,0,0);
  if(state.overlay&&$("show-mask").checked)ctx.drawImage(state.overlay,0,0);
  const width=Math.max(2,item.width/280),radius=item.width/110;
  ctx.strokeStyle="#ffe28a";ctx.fillStyle="#ffe28a";ctx.lineWidth=width;
  function marker([x,y]){ctx.beginPath();ctx.moveTo(x-radius,y);ctx.lineTo(x+radius,y);ctx.moveTo(x,y-radius);ctx.lineTo(x,y+radius);ctx.stroke();ctx.beginPath();ctx.arc(x,y,2*width,0,2*Math.PI);ctx.stroke();}
  if(mode()==="point_orientation"&&state.point)marker(state.point);
  if(mode()==="line_orientation"){
    if(state.line.length===2){const [a,b]=state.line,dx=b[0]-a[0],dy=b[1]-a[1];ctx.save();ctx.setLineDash([width*4,width*3]);ctx.beginPath();ctx.moveTo(a[0]-dx*100,a[1]-dy*100);ctx.lineTo(b[0]+dx*100,b[1]+dy*100);ctx.stroke();ctx.restore();}
    state.line.forEach(marker);
  }
}
async function previewMask() {
  if(!hasMaskInput()||state.busy||!state.ready)return;
  const revision=state.revision;setBusy(true,"Preparing mask preview…");clearError();
  try{
    const output=await workerCall("mask",configuration());
    const overlay=await loadBitmap(output.overlay);
    if(revision===state.revision){state.overlay=overlay;$("overlay-control").hidden=false;$("show-mask").checked=true;drawInput();$("mask-status").textContent=output.kind==="cell_mask"?"Pink: segmented cell. Alignment is evaluated around it.":"Pink: selected analysis area.";}
    setStatus("Mask preview ready");
  }catch(error){showError(error);setStatus("Adjust the mask settings",false,true);}finally{setBusy(false);}
}
function format(value,decimals=3){return value===null||!Number.isFinite(value)?"—":value.toFixed(decimals);}
function renderResult(result) {
  state.result=result;state.axis="x";
  const suffix=result.suffix||"x",name={cell:"cell",point:"point",line:"line",x:"X axis"}[suffix],summary=result.summary;
  $("results-title").textContent=labels[mode()][1];$("alignment-label").textContent="Alignment to "+name;
  $("alignment-value").textContent=format(summary["mean_alignment_"+suffix]);
  $("deviation-label").textContent="Angular deviation to "+name;$("deviation-value").textContent=format(summary["mean_deviation_"+suffix],1)+"°";
  $("coherency-value").textContent=format(summary.mean_coherency);
  $("run-info").textContent=`${result.pixels.toLocaleString()} valid pixels · ${result.seconds}s`;
  [...$("map-choice").options].forEach(option=>{if(option.value.endsWith("reference"))option.hidden=!result.suffix;});
  $("map-choice").value=result.suffix?"alignment_reference":"angle";
  $("histogram-image").src="data:image/png;base64,"+result.plots.histogram;
  $("results").hidden=false;$("before-results").hidden=true;showMap();updateButtons();
}
function currentMapKey() { const choice=$("map-choice").value;return choice.endsWith("reference")?choice.replace("reference",state.result.suffix):["angle","alignment"].includes(choice)?choice+"_"+state.axis:choice; }
function showMap() {
  if(!state.result)return;
  $("map-image").src="data:image/png;base64,"+state.result.plots[currentMapKey()];
  $("axis-switch").hidden=!["angle","alignment"].includes($("map-choice").value);
  document.querySelectorAll("[data-axis]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.axis===state.axis)));
  if(!state.result.suffix){const axis=state.axis,s=state.result.summary;$("alignment-label").textContent="Alignment to "+axis.toUpperCase()+" axis";$("alignment-value").textContent=format(s["mean_alignment_"+axis]);$("deviation-label").textContent="Angular deviation to "+axis.toUpperCase()+" axis";$("deviation-value").textContent=format(s["mean_deviation_"+axis],1)+"°";}
}
async function runAnalysis() {
  if(state.busy)return;
  clearTimeout(maskTimer);clearError();let config;
  try{config=configuration();}catch(error){showError(error);return;}
  if(mode()==="point_orientation"&&!state.point){showError("Click a reference point in the image first.");return;}
  if(mode()==="line_orientation"&&state.line.length!==2){showError("Click two points to define the reference line first.");return;}
  state.result=null;$("results").hidden=true;setBusy(true,"Analyzing image…");$("cancel").hidden=false;
  try{const result=await workerCall("analyze",config);renderResult(result);setStatus("Analysis complete");}
  catch(error){showError(error);setStatus("Analysis could not be completed",false,true);}
  finally{setBusy(false);$("cancel").hidden=true;}
}
function saveBlob(data,name,type){const url=URL.createObjectURL(new Blob([data],{type}));const link=document.createElement("a");link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function savePng(key){
  clearDownloadError();
  try{const data=Uint8Array.from(atob(state.result.plots[key]),character=>character.charCodeAt(0));saveBlob(data,"compaction_"+key+".png","image/png");}
  catch(error){showDownloadError(errorMessage(error));}
}
async function download(format){
  if(!state.result)return;
  clearDownloadError();
  const scale=Number($("pixel-size").value);
  if(format==="saenopy"&&(!Number.isFinite(scale)||scale<=0||!$("pixel-size").checkValidity())){
    $("pixel-size").setAttribute("aria-invalid","true");
    showDownloadError("Enter a positive pixel size (µm/px) above to download a Saenopy project. PNG, NumPy and All results work without it.");return;
  }
  setBusy(true,format==="saenopy"?"Preparing Saenopy project…":"Preparing download…");
  try{const bytes=await workerCall("export",{format,scale:scale>0?scale:null});saveBlob(bytes,format==="npz"?"compaction_arrays.npz":format==="saenopy"?"compaction_saenopy.zip":"compaction_results.zip","application/octet-stream");setStatus("Download ready");}
  catch(error){showDownloadError(errorMessage(error));setStatus("Download could not be prepared",false,true);}finally{setBusy(false);}
}
async function loadExample() {
  if(state.busy||!state.ready||!state.manifest)return;
  const preset=state.manifest[$("example").value];invalidate();setBusy(true,"Loading example images…");
  try{
    delete state.images.mask;await workerCall("clear",{slot:"mask"});
    document.querySelector(`input[name="mode"][value="${preset.mode}"]`).checked=true;
    $("region").value=preset.region||"whole";
    $("pixel-size").value=preset.pixel_size||"";
    for(const [name,value]of Object.entries(preset.settings))$(name).value=value;
    $("invert").checked=false;$("threshold-range").value=$("threshold").value;updateMode(false);
    for(const [slot,path]of [["orientation",preset.image],["mask",preset.mask]]){
      if(!path)continue;const response=await fetch("examples/"+path);if(!response.ok)throw new Error("The example image could not be loaded.");
      await loadFile(new File([await response.blob()],path.split("/").at(-1)),slot);
    }
    state.point=preset.point||null;state.line=preset.line||[];renderCoordinates();drawInput();setStatus("Example loaded · ready to analyze");
  }catch(error){showError(error);setStatus("Could not load the example",false,true);}finally{setBusy(false);}
  if(state.images.orientation&&(mode()==="full"||$("region").value!=="whole"))await previewMask();
}
for(const slot of ["orientation","mask"]){
  $(slot+"-file").addEventListener("change",event=>{chooseFile(event.target.files[0],slot);event.target.value="";});
  const zone=$(slot+"-drop");
  zone.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();if(state.ready&&!state.busy)$(slot+"-file").click();}});
  zone.addEventListener("dragover",event=>{event.preventDefault();zone.classList.add("dragging");});
  zone.addEventListener("dragleave",()=>zone.classList.remove("dragging"));
  zone.addEventListener("drop",event=>{event.preventDefault();zone.classList.remove("dragging");chooseFile(event.dataTransfer.files[0],slot);});
}
document.querySelectorAll('input[name="mode"]').forEach(input=>input.addEventListener("change",()=>{invalidate();updateMode();}));
$("region").addEventListener("change",()=>{invalidate();updateMode(false);});
for(const id of ["sigma","edge","blur","threshold","cell-blur","background-blur","mask-blur","invert"]){$(id).addEventListener("input",()=>{invalidate();if(id==="threshold")$("threshold-range").value=$(id).value;});}
$("threshold-range").addEventListener("input",()=>{$("threshold").value=$("threshold-range").value;invalidate();if(hasMaskInput())maskTimer=setTimeout(previewMask,450);});
$("show-mask").addEventListener("change",drawInput);$("preview-mask").addEventListener("click",previewMask);
$("input-canvas").addEventListener("click",event=>{
  if(state.busy||!state.images.orientation)return;
  const canvas=event.currentTarget,rect=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(canvas.width-1,(event.clientX-rect.left)*canvas.width/rect.width)),y=Math.max(0,Math.min(canvas.height-1,(event.clientY-rect.top)*canvas.height/rect.height));
  const point=[Math.round(x*2)/2,Math.round(y*2)/2];
  if(mode()==="point_orientation")state.point=point;else if(mode()==="line_orientation"){if(state.line.length===2)state.line=[];state.line.push(point);}else return;
  invalidate();renderCoordinates();drawInput();
});
$("clear-reference").addEventListener("click",()=>{state.point=null;state.line=[];invalidate();renderCoordinates();});
$("analyze").addEventListener("click",runAnalysis);$("load-example").addEventListener("click",loadExample);
$("map-choice").addEventListener("change",showMap);document.querySelectorAll("[data-axis]").forEach(button=>button.addEventListener("click",()=>{state.axis=button.dataset.axis;showMap();}));
$("save-map").addEventListener("click",()=>savePng(currentMapKey()));$("save-histogram").addEventListener("click",()=>savePng("histogram"));
$("save-numpy").addEventListener("click",()=>download("npz"));$("save-all").addEventListener("click",()=>download("zip"));
$("save-saenopy").addEventListener("click",()=>download("saenopy"));
$("pixel-size").addEventListener("input",()=>{const scale=Number($("pixel-size").value);if(Number.isFinite(scale)&&scale>0&&$("pixel-size").checkValidity())clearDownloadError();});
$("retry").addEventListener("click",startEngine);$("cancel").addEventListener("click",()=>{startEngine();setStatus("Restarting after cancellation. Load your image again when ready.",true);});
fetch("examples/manifest.json").then(response=>{if(!response.ok)throw new Error("Could not load example list.");return response.json();}).then(data=>state.manifest=data).catch(showError);
startEngine();
