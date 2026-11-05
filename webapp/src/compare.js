"use strict";
const compareState = {groups:[], selected:null, nextId:0, plot:null};
const compareMetrics = [
  ["cell_alignment","Cell-relative alignment"], ["cell_deviation","Angular deviation to cell centre"],
  ["alignment_x","Alignment to X axis"], ["alignment_y","Alignment to Y axis"],
  ["deviation_x","Angular deviation to X axis"], ["deviation_y","Angular deviation to Y axis"],
  ["coherency","Coherency"], ["point_alignment","Alignment towards point"],
  ["point_deviation","Angular deviation to point"], ["line_alignment","Alignment to reference line"],
  ["line_deviation","Angular deviation to reference line"], ["histogram_x","Angle histogram: X axis"],
  ["histogram_y","Angle histogram: Y axis"], ["histogram_line","Angle histogram: reference line"],
  ["histogram_point","Angular deviation histogram: point"], ["histogram_cell","Angular deviation histogram: cell centre"]
];
const modeTitles = {full:"Cell",axis_orientation_only:"X/Y · whole image",axis_orientation_masked:"X/Y · mask",point_orientation:"Point",line_orientation:"Line"};
function comparisonError(message){$("compare-error").textContent=message;$("compare-error").hidden=!message;}
function comparisonFiles(){return compareState.groups.flatMap(group=>group.files);}
function clearComparisonPlot(){
  compareState.plot=null;$("compare-image").hidden=true;$("compare-results").hidden=true;
  $("compare-empty").hidden=false;$("compare-empty").textContent=comparisonFiles().length?"Choose a measurement and select Plot.":"Add measurements or load the example comparison.";
}
function metricAvailable(key){return compareState.groups.some(group=>group.files.some(file=>file.included&&(key.startsWith("histogram_")?file.histograms[key]:Number.isFinite(file.values[key]))));}
function syncComparisonControls(){
  const disabled=state.busy||!state.ready;
  document.querySelectorAll("#compare-view button,#compare-view input,#compare-view select").forEach(control=>control.disabled=disabled);
  $("compare-plot").disabled=disabled||!metricAvailable($("compare-metric").value);
  $("compare-metric").disabled=disabled||!compareMetrics.some(([key])=>metricAvailable(key));
  for(const id of ["compare-png","compare-svg","compare-csv","compare-summary"])$(id).disabled=disabled||!compareState.plot;
  $("compare-engine-status").hidden=state.ready;
  $("compare-engine-status").textContent=$("status").textContent;
  $("compare-retry").hidden=$("retry").hidden;$("compare-retry").disabled=false;
  if(!state.ready&&!$("error").hidden)comparisonError($("error").textContent);
}
function updateComparisonMetrics(){
  const select=$("compare-metric"),old=select.value;select.replaceChildren();
  for(const [key,title]of compareMetrics){const option=document.createElement("option");option.value=key;option.textContent=title;option.disabled=!metricAvailable(key);select.append(option);}
  select.value=metricAvailable(old)?old:(compareMetrics.find(([key])=>metricAvailable(key))||compareMetrics[0])[0];
  syncComparisonControls();
}
function newCondition(name){
  const id=++compareState.nextId;
  const group={id,name,files:[]};compareState.groups.push(group);compareState.selected=id;return group;
}
function selectCondition(id){compareState.selected=id;renderConditions();}
function conditionNameAvailable(name,except){return name&&!compareState.groups.some(group=>group.id!==except&&group.name.toLowerCase()===name.toLowerCase());}
function renderConditions(){
  const host=$("conditions");host.replaceChildren();
  for(const group of compareState.groups){
    const card=document.createElement("article");card.className="condition-card"+(group.id===compareState.selected?" selected":"");
    const head=document.createElement("div");head.className="condition-head";
    const select=document.createElement("input");select.type="radio";select.name="compare-condition";select.checked=group.id===compareState.selected;select.setAttribute("aria-label","Select condition "+group.name);select.addEventListener("change",()=>selectCondition(group.id));
    const name=document.createElement("input");name.type="text";name.maxLength=80;name.value=group.name;name.setAttribute("aria-label","Condition name: "+group.name);
    name.addEventListener("change",()=>{const value=name.value.trim();if(!conditionNameAvailable(value,group.id)){comparisonError("Use a different, nonempty name for each condition.");name.value=group.name;return;}group.name=value;comparisonError("");clearComparisonPlot();renderConditions();});
    const remove=document.createElement("button");remove.className="remove-condition";remove.textContent="×";remove.setAttribute("aria-label","Remove condition "+group.name);
    remove.addEventListener("click",()=>{compareState.groups=compareState.groups.filter(item=>item.id!==group.id);if(compareState.selected===group.id)compareState.selected=compareState.groups[0]?.id??null;clearComparisonPlot();renderConditions();});
    head.append(select,name,remove);card.append(head);
    const drop=document.createElement("button");drop.className="condition-drop";drop.textContent="Drop .saenopyOrientation files or choose files";
    drop.addEventListener("click",()=>{compareState.selected=group.id;$("compare-files").click();});card.append(drop);
    card.addEventListener("dragover",event=>{event.preventDefault();if(!state.busy&&state.ready)card.classList.add("dragging");});
    card.addEventListener("dragleave",()=>card.classList.remove("dragging"));
    card.addEventListener("drop",event=>{event.preventDefault();card.classList.remove("dragging");importMeasurements(event.dataTransfer.files,group.id);});
    if(group.files.length){
      const list=document.createElement("ul");list.className="measurement-list";
      for(const file of group.files){
        const row=document.createElement("li"),label=document.createElement("label"),checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.checked=file.included;checkbox.setAttribute("aria-label","Include "+file.name);
        checkbox.addEventListener("change",()=>{file.included=checkbox.checked;clearComparisonPlot();updateComparisonMetrics();});
        const text=document.createElement("span");text.textContent=file.name;text.title=file.name;
        const badge=document.createElement("small");badge.textContent=modeTitles[file.mode]||file.mode;
        label.append(checkbox,text);row.append(label,badge);
        const removeFile=document.createElement("button");removeFile.className="remove-file";removeFile.textContent="×";removeFile.setAttribute("aria-label","Remove "+file.name);
        removeFile.addEventListener("click",()=>{group.files=group.files.filter(item=>item.id!==file.id);clearComparisonPlot();renderConditions();});row.append(removeFile);list.append(row);
      }
      card.append(list);
    }
    host.append(card);
  }
  updateComparisonMetrics();
}
async function importMeasurements(files,groupId){
  if(state.busy||!state.ready)return;
  const group=compareState.groups.find(item=>item.id===groupId);
  if(!group){comparisonError("Add or select a condition first.");return;}
  const incoming=[...files];if(!incoming.length)return;
  if(comparisonFiles().length+incoming.length>200){comparisonError("Use at most 200 measurements per comparison.");return;}
  clearComparisonPlot();comparisonError("");setBusy(true,"Reading saved measurements…");
  let added=0,duplicates=0;const errors=[];
  try{
    for(const [index,file]of incoming.entries()){
      $("compare-import-status").textContent=`Reading ${index+1}/${incoming.length}: ${file.name}`;
      try{
        if(!file.name.toLowerCase().endsWith(".saenopyorientation"))throw new Error("Choose an evaluated .saenopyOrientation file.");
        if(file.size>512*1024*1024)throw new Error("This file exceeds the 512 MB browser comparison limit.");
        const measurement=await workerCall("compare_import",{name:file.name},await file.arrayBuffer());
        if(comparisonFiles().some(item=>item.id===measurement.id)){duplicates++;continue;}
        group.files.push({...measurement,included:true});added++;
      }catch(error){errors.push(file.name+": "+errorMessage(error));}
    }
    $("compare-import-status").textContent=`${added} measurement${added===1?"":"s"} added to ${group.name}.`+(duplicates?` ${duplicates} duplicate${duplicates===1?"":"s"} skipped.`:"");
    comparisonError(errors.join("\n"));setStatus("Saved measurements loaded");
  }finally{setBusy(false);renderConditions();}
}
async function plotComparison(){
  if(state.busy||!state.ready)return;
  comparisonError("");clearComparisonPlot();setBusy(true,"Combining measurements…");
  $("compare-import-status").textContent="Preparing comparison…";
  try{
    const metric=$("compare-metric").value;
    const groups=compareState.groups.map(group=>({name:group.name,files:group.files.filter(file=>file.included)}));
    const result=await workerCall("compare_plot",{metric,groups});compareState.plot=result;
    $("compare-image").src="data:image/png;base64,"+result.png;$("compare-image").hidden=false;
    $("compare-empty").hidden=true;$("compare-results").hidden=false;
    $("compare-note").textContent=metric.startsWith("histogram_")?"Each image's bin probabilities sum to 1. Curves show the group mean ± SEM; every image counts equally.":"Each image counts equally. Bars show mean ± SEM; dots show individual measurements.";
    $("compare-count").textContent=`${result.n} measurement${result.n===1?"":"s"} included.`+(result.excluded?` ${result.excluded} without this metric excluded.`:"");
    $("compare-import-status").textContent="Comparison ready";setStatus("Comparison ready");
  }catch(error){comparisonError(errorMessage(error));$("compare-import-status").textContent="Check the selected measurements.";}
  finally{setBusy(false);}
}
async function loadComparisonDemo(){
  if(state.busy||!state.ready)return;
  comparisonError("");setBusy(true,"Loading comparison example…");
  try{
    const response=await fetch("comparison-demo.json");if(!response.ok)throw new Error("The comparison example could not be loaded.");
    const demo=await response.json();
    if(!comparisonFiles().length)compareState.groups=[];
    let added=0;
    for(const source of demo.groups){
      const fresh=source.files.filter(file=>!comparisonFiles().some(item=>item.id===file.id));
      if(!fresh.length)continue;
      if(compareState.groups.length>=20||comparisonFiles().length+fresh.length>200)throw new Error("The comparison limit has been reached.");
      let name=source.name,suffix=2;while(!conditionNameAvailable(name))name=source.name+" "+suffix++;
      const group=newCondition(name);group.files=fresh.map(file=>({...file,included:true}));added+=fresh.length;
    }
    clearComparisonPlot();renderConditions();$("compare-metric").value="alignment_x";
    $("compare-import-status").textContent=added?"Synthetic example: three conditions with three images each.":"The example measurements are already loaded.";
  }catch(error){comparisonError(errorMessage(error));}
  finally{setBusy(false);}
  if(metricAvailable($("compare-metric").value))await plotComparison();
}
function switchWorkspace(compare){
  $("image-view").hidden=compare;$("compare-view").hidden=!compare;
  $("image-tab").setAttribute("aria-selected",String(!compare));$("compare-tab").setAttribute("aria-selected",String(compare));
  $("image-tab").tabIndex=compare?-1:0;$("compare-tab").tabIndex=compare?0:-1;
  document.querySelector(".examples").hidden=compare;
  document.querySelector("h1").textContent=compare?"Compare fiber orientation":"Measure fiber orientation";
  history.replaceState(null,"",compare?"#compare-data":location.pathname);
}
$("image-tab").addEventListener("click",()=>switchWorkspace(false));$("compare-tab").addEventListener("click",()=>switchWorkspace(true));
for(const id of ["image-tab","compare-tab"])$(id).addEventListener("keydown",event=>{if(event.key==="ArrowLeft"||event.key==="ArrowRight"){event.preventDefault();const compare=id==="image-tab";switchWorkspace(compare);$(compare?"compare-tab":"image-tab").focus();}});
$("add-condition").addEventListener("click",()=>{
  const name=$("condition-name").value.trim();if(compareState.groups.length>=20){comparisonError("Use at most 20 conditions.");return;}
  if(!conditionNameAvailable(name)){comparisonError("Enter a different, nonempty condition name.");return;}
  newCondition(name);$("condition-name").value="";comparisonError("");clearComparisonPlot();renderConditions();
});
$("condition-name").addEventListener("keydown",event=>{if(event.key==="Enter")$("add-condition").click();});
$("compare-files").addEventListener("change",event=>{importMeasurements(event.target.files,compareState.selected);event.target.value="";});
$("compare-metric").addEventListener("change",clearComparisonPlot);
$("compare-plot").addEventListener("click",plotComparison);$("compare-demo").addEventListener("click",loadComparisonDemo);
$("compare-png").addEventListener("click",()=>{const result=compareState.plot;if(result)saveBlob(Uint8Array.from(atob(result.png),c=>c.charCodeAt(0)),"comparison_"+result.metric+".png","image/png");});
for(const [id,key,suffix,type]of [["compare-svg","svg",".svg","image/svg+xml"],["compare-csv","raw_csv","_individual.csv","text/csv;charset=utf-8"],["compare-summary","summary_csv","_summary.csv","text/csv;charset=utf-8"]]){
  $(id).addEventListener("click",()=>{const result=compareState.plot;if(result)saveBlob(result[key],"comparison_"+result.metric+suffix,type);});
}
document.addEventListener("analysis-state-change",syncComparisonControls);
document.addEventListener("analysis-status-change",()=>{if(!state.ready)$("compare-engine-status").textContent=$("status").textContent;});
$("compare-retry").addEventListener("click",()=>{comparisonError("");startEngine();});
newCondition("Condition 1");renderConditions();switchWorkspace(location.hash==="#compare-data");
