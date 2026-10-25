import {readGzip, clamp, containsPoint, distanceKm} from './math.js';

const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value).toLocaleString('en-US', {maximumFractionDigits: 0});
const title = value => value.toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase()).replace(/\b(Nr|At|Near|Above|Below)\b/g, s => s.toLowerCase());
const state = {spaceKey: 'embeddings', period: 'avg', mode: 'embedding', selected: null, tab: 'neighbors', excludeOverlap: false, distance: 0, request: 0, busy: true};
let catalog, basins, geometry, layers = [], boxes = [], sortedIndices = [], dataResult, selectedLayer, gaugeLayer, hoverFrame, toastTimeout;
let colorLow = -1, colorHigh = 1, scatterBounds, scatterPixels = [], latestTrendRequest = 0;
const trendCache = new Map();
const map = L.map('map', {zoomControl: false, preferCanvas: true, minZoom: 3, maxZoom: 13, zoomSnap: .25, attributionControl: true});
L.control.zoom({position: 'bottomright'}).addTo(map);
L.control.scale({position: 'bottomleft', imperial: false}).addTo(map);
map.attributionControl.addAttribution('Basin boundaries: USGS GAGES-II · State outlines: PublicaMundi');
map.createPane('base'); map.getPane('base').style.zIndex = 200;
map.createPane('labels'); map.getPane('labels').style.zIndex = 350; map.getPane('labels').style.pointerEvents = 'none';
map.createPane('selection'); map.getPane('selection').style.zIndex = 420; map.getPane('selection').style.pointerEvents = 'none';
const renderer = L.canvas({padding: .3, tolerance: 0});
function resetMap() { map.fitBounds([[24.7, -125], [49.4, -66.6]], {paddingTopLeft: [20, 80], paddingBottomRight: [20, 115], animate: false}); }
resetMap();
const worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});

function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimeout); toastTimeout = setTimeout(() => $('toast').hidden = true, 4300); }
function setBusy(busy) { state.busy = busy; $('export').disabled = busy || state.selected === null; $('ready-status').textContent = busy ? 'Loading full-dimensional embeddings…' : `${number(basins.length)} basins · ${state.period === 'avg' ? '40-year mean' : state.period} · ${state.spaceKey === 'embeddings' ? 'Projection z' : 'Hydrologic h'}`; }
function syncURL() { const params = new URLSearchParams({space: state.spaceKey, year: state.period, view: state.mode}); if (state.selected !== null) params.set('basin', basins[state.selected].id); history.replaceState(null, '', `#${params}`); }
function requestEmbedding() {
  if (!catalog) return;
  setBusy(true); dataResult = null;
  $('year-label').textContent = state.period === 'avg' ? '1985–2024' : state.period;
  $('mean').classList.toggle('active', state.period === 'avg'); $('mean').setAttribute('aria-pressed', state.period === 'avg');
  $('time-help').textContent = state.mode === 'change' ? 'Change always compares 1985 with 2024. The timeline controls signature and neighbors.' : state.period === 'avg' ? 'Drag the timeline to explore an individual year.' : 'Annual vector · choose 40-year mean to return to the long-term signature.';
  $('selection').setAttribute('aria-busy', 'true');
  renderSelection(); renderDetails();
  worker.postMessage({request: ++state.request, spaceKey: state.spaceKey, period: state.period, selected: state.selected, space: catalog.spaces[state.spaceKey], count: basins.length});
  syncURL();
}
worker.onmessage = ({data}) => {
  if (data.request !== state.request) return;
  if (data.error) { showError(data.error); return; }
  dataResult = data; setBusy(false); $('selection').setAttribute('aria-busy', 'false'); $('loading').hidden = true;
  if (data.scores) {
    const sorted = Array.from(data.scores).sort((a,b) => a-b);
    colorLow = sorted[Math.floor(sorted.length * .05)]; colorHigh = sorted[Math.floor(sorted.length * .95)];
  }
  renderMap(); renderDetails(); drawScatter();
};
worker.onerror = () => showError('The embedding worker could not start. Reload to try again.');
function showError(message) {
  state.busy = false; $('export').disabled = true;
  $('loading').hidden = false; $('loading').innerHTML = `<strong>We couldn’t load the atlas</strong><span>${esc(message)}</span><button class="error-retry" id="retry">Reload data</button>`;
  $('retry').onclick = () => location.reload(); $('ready-status').textContent = 'Data unavailable · retry to continue';
}

const palette = ['#263a68','#356f94','#489b9e','#91c9a0','#f0dda0'];
function mixColor(t, colors = palette) {
  const p = clamp(t, 0, 1) * (colors.length - 1), low = Math.floor(p), high = Math.min(low+1, colors.length-1), f = p-low;
  const a = colors[low].match(/[a-f\d]{2}/gi).map(v => parseInt(v,16)), b = colors[high].match(/[a-f\d]{2}/gi).map(v => parseInt(v,16));
  return `rgb(${a.map((v,j) => Math.round(v+(b[j]-v)*f)).join(',')})`;
}
function rgbColor(index) {
  if (!dataResult) return '#548e89';
  const limits = catalog.spaces[state.spaceKey].limits;
  const c = limits.map(([a,b], k) => Math.round(55 + 170 * clamp((dataResult.points[index*3+k]-a)/(b-a),0,1)));
  return `rgb(${c.join(',')})`;
}
function basinColor(index) {
  if (state.mode === 'similarity' && dataResult?.scores) return mixColor((dataResult.scores[index]-colorLow)/(colorHigh-colorLow || 1));
  if (state.mode === 'change') { const s = catalog.spaces[state.spaceKey]; return mixColor(s.change[index]/s.changeMax, ['#254c5a','#5f8f93','#c0b997','#eeab70','#d96957']); }
  return rgbColor(index);
}
function renderMap() {
  layers.forEach((layer, index) => layer.setStyle({fillColor: basinColor(index), fillOpacity: .82, color: $('outlines').checked ? '#122831' : basinColor(index), weight: $('outlines').checked ? .65 : 0}));
  drawSelection(); renderLegend();
}
function drawSelection() {
  if (selectedLayer) map.removeLayer(selectedLayer);
  if (state.selected === null) return;
  selectedLayer = L.geoJSON(geometry.features[state.selected], {pane: 'selection', interactive: false, style: {color: '#ffe1a0', weight: 2.6, fillOpacity: .10, fillColor: '#ffe1a0'}}).addTo(map);
}
function renderLegend() {
  if (state.mode === 'similarity' && dataResult?.scores) $('legend').innerHTML = `<div class="legend-title"><span>Cosine similarity</span><span>64D</span></div><div class="swatch" style="background:linear-gradient(90deg,${palette.join(',')})"></div><div class="legend-labels"><span>≤ ${colorLow.toFixed(3)}</span><span>≥ ${colorHigh.toFixed(3)}</span></div><p>5–95% color stretch · gold outline = selected basin</p>`;
  else if (state.mode === 'change') {
    const max = catalog.spaces[state.spaceKey].changeMax;
    $('legend').innerHTML = `<div class="legend-title"><span>Representation change</span><span>1985 → 2024</span></div><div class="swatch" style="background:linear-gradient(90deg,#254c5a,#5f8f93,#c0b997,#eeab70,#d96957)"></div><div class="legend-labels"><span>0 · stable</span><span>≥ ${max.toFixed(3)}</span></div><p>1 − cosine · colors clipped at 98th percentile</p>`;
  } else $('legend').innerHTML = '<div class="legend-title"><span>Embedding signatures</span><span>PC 1 · 2 · 3</span></div><div class="swatch" style="background:linear-gradient(90deg,#7ca9b3,#bf86a9,#c9a772,#98be85,#6cb8ba)"></div><p>Three principal components → RGB<br>Color summarizes the full 64D signature.</p>';
}
function setMode(mode) {
  if (mode === 'similarity' && state.selected === null) { toast('Select a basin on the map or search to compare its signature.'); $('search').focus(); return; }
  state.mode = mode;
  document.querySelectorAll('[data-mode]').forEach(b => { b.classList.toggle('active', b.dataset.mode === mode); b.setAttribute('aria-pressed', b.dataset.mode === mode); });
  $('time-help').textContent = mode === 'change' ? 'Change always compares 1985 with 2024. The timeline controls signature and neighbors.' : state.period === 'avg' ? 'Drag the timeline to explore an individual year.' : 'Annual vector · choose 40-year mean to return to the long-term signature.';
  renderMap(); syncURL();
}
function selectBasin(index, {zoom = false, similarity = true} = {}) {
  state.selected = index;
  if (similarity) state.mode = 'similarity';
  document.querySelectorAll('[data-mode]').forEach(b => { b.classList.toggle('active', b.dataset.mode === state.mode); b.setAttribute('aria-pressed', b.dataset.mode === state.mode); });
  $('search-results').hidden = true; $('search').setAttribute('aria-expanded', 'false'); $('search').value = '';
  $('overlap-picker').hidden = true;
  requestEmbedding(); drawSelection();
  if (zoom) zoomToBasin(index);
}
function zoomToBasin(index) { map.fitBounds(layers[index].getBounds(), {padding: [65, 85], maxZoom: 9, animate: !matchMedia('(prefers-reduced-motion: reduce)').matches}); }
function renderSelection() {
  if (state.selected === null) {
    $('selection').innerHTML = '<div class="selection-kicker">EXPLORE THE ATLAS <span>↗</span></div><h2>Every basin has<br>a signature.</h2><p>Choose a basin to discover its closest counterparts, across the country and through time.</p>';
    return;
  }
  const b = basins[state.selected];
  $('selection').innerHTML = `<div class="selection-kicker">SELECTED BASIN <button id="clear-basin" aria-label="Clear selected basin">×</button></div><h2 class="basin-name">${esc(title(b.name))}</h2><div class="selection-meta">USGS ${b.id}<span>·</span>${esc(b.state)}<span class="pill">HUC ${b.huc}</span></div><div class="basin-stats"><div><strong>${number(b.area)} <small>km²</small></strong><span>Basin area</span></div><div><strong>${b.overlaps.length}</strong><span>Recorded strong overlaps</span></div></div><div class="basin-links"><button id="zoom-basin">⌖ Zoom to basin</button><a href="https://waterdata.usgs.gov/monitoring-location/USGS-${b.id}/" target="_blank" rel="noopener">USGS station ↗</a></div>`;
  $('zoom-basin').onclick = () => zoomToBasin(state.selected);
  $('clear-basin').onclick = () => { state.selected = null; state.mode = 'embedding'; $('overlap-picker').hidden = true; setMode('embedding'); requestEmbedding(); };
}
function neighbors() {
  if (state.selected === null || !dataResult?.scores) return [];
  const anchor = basins[state.selected], overlapping = new Set(anchor.overlaps.map(x=>x[0]));
  return Array.from(dataResult.scores, (score, index) => ({index, score, distance: distanceKm(anchor, basins[index])}))
    .filter(row => row.index !== state.selected && (!state.excludeOverlap || !overlapping.has(row.index)) && row.distance >= state.distance)
    .sort((a,b) => b.score-a.score || a.index-b.index);
}
function renderDetails() {
  $('details').setAttribute('aria-busy', state.busy);
  if (state.selected === null) {
    const examples = ['04101500','12457000','01595500'].map(id => basins.findIndex(b=>b.id===id)).filter(i=>i>=0);
    $('details').innerHTML = `<div class="section-heading"><h3>Start somewhere familiar</h3><span>↗</span></div><p class="empty-help">Click any basin, search by name, or explore one of these watersheds.</p><div class="examples">${examples.map(i=>`<button data-select="${i}"><span>${esc(title(basins[i].name))}<small>${basins[i].state} · USGS ${basins[i].id}</small></span><span>→</span></button>`).join('')}</div><p class="list-note">Nested basins? A click reveals every basin at that location, so none are hidden behind a larger watershed.</p>`;
    bindSelectButtons(); return;
  }
  if (state.tab === 'overlaps') { renderOverlaps(); return; }
  if (!dataResult) { $('details').innerHTML = '<p class="empty-help">Calculating the full 64-dimensional signature…</p>'; return; }
  if (state.tab === 'signature') { renderSignature(); return; }
  const rows = neighbors();
  $('details').innerHTML = `<div class="section-heading"><h3>Closest signatures</h3><span class="mini-badge">COSINE</span></div><div class="filter-row"><label><input id="exclude-overlap" type="checkbox" ${state.excludeOverlap?'checked':''}> Exclude recorded strong overlaps</label><label for="distance">Distance</label><select id="distance"><option value="0">Any distance from selected basin</option><option value="100">At least 100 km away</option><option value="500">At least 500 km away</option><option value="1000">At least 1,000 km away</option></select></div><div class="list-heading"><span>${number(rows.length)} CANDIDATES</span><span>SIMILARITY</span></div>${rows.slice(0,10).map((r,j)=>`<button class="neighbor" data-select="${r.index}" title="Select ${esc(title(basins[r.index].name))}"><span class="rank">${String(j+1).padStart(2,'0')}</span><span><span class="neighbor-title">${esc(title(basins[r.index].name))}</span><small>${basins[r.index].state} · ${number(r.distance)} km · ${basins[r.index].id}</small></span><span class="score">${r.score.toFixed(3)}<span class="score-bar"><i style="width:${clamp((r.score+1)/2,0,1)*100}%"></i></span></span></button>`).join('')}${!rows.length?'<p class="empty-help">No basins match these filters. Reduce the distance or include overlaps.</p>':''}<p class="list-note">Ranked in the full 64D space. Distances use basin centroids. These filters affect the list and CSV export; the map still shows all basins.</p>`;
  $('distance').value = String(state.distance);
  $('exclude-overlap').onchange = event => { state.excludeOverlap = event.target.checked; renderDetails(); };
  $('distance').onchange = event => { state.distance = Number(event.target.value); renderDetails(); };
  bindSelectButtons();
}
function bindSelectButtons() { document.querySelectorAll('[data-select]').forEach(button => button.onclick = () => selectBasin(Number(button.dataset.select))); }
function renderOverlaps() {
  const b = basins[state.selected], rows = b.overlaps.slice().sort((a,c) => basins[a[0]].area - basins[c[0]].area);
  $('details').innerHTML = `<div class="section-heading"><h3>Shared drainage areas</h3><span class="mini-badge">${rows.length} LINKS</span></div><p class="empty-help">Shared area as a fraction of the smaller basin. Near 100% suggests near containment.</p>${rows.map(([index, score])=>`<button class="overlap-item" data-select="${index}">${esc(title(basins[index].name))}<small><span class="overlap-relation">${score >= .99 ? (basins[index].area < b.area ? 'Smaller · nearly contained' : 'Larger · nearly contains this basin') : 'Partially overlapping'} · ${(score*100).toFixed(1)}%</span><br>${number(basins[index].area)} km² · ${basins[index].id}</small></button>`).join('')}${!rows.length?'<p class="list-note">No strong overlaps are recorded for this basin. This does not rule out smaller intersections.</p>':''}<p class="list-note">These relationships come from original geometry in EPSG:5070. The map click chooser uses simplified display boundaries and can show additional basins.</p>`;
  bindSelectButtons();
}
async function renderSignature() {
  const values = dataResult.vector, max = Math.max(...Array.from(values, Math.abs)), b = basins[state.selected];
  const change = catalog.spaces[state.spaceKey].change[state.selected];
  $('details').innerHTML = `<div class="section-heading"><h3>The 64D signature</h3><span class="mini-badge">${state.period === 'avg' ? 'MEAN' : state.period}</span></div><p class="empty-help">Each tile is one coordinate of the normalized vector. Hover to inspect its value.</p><div class="signature-grid">${Array.from(values,(v,i)=>`<span class="signature-cell" tabindex="0" title="Dimension ${i+1}: ${v.toFixed(6)}" aria-label="Dimension ${i+1}: ${v.toFixed(6)}" style="background:${mixColor((v/max+1)/2,['#647db4','#233843','#d4be88'])}"></span>`).join('')}</div><div class="signature-labels"><span>−${max.toFixed(3)}</span><span>0</span><span>+${max.toFixed(3)}</span></div><button id="download-vector" class="vector-download">↓ Download 64D vector · JSON</button><div class="section-heading"><h3>A signature through time</h3><span class="mini-badge">40 YEARS</span></div><p class="empty-help">Cosine similarity to this basin’s 1985 vector.</p><div id="trend"><p class="list-note">Loading annual history…</p></div><div class="change-metric">${change.toFixed(3)}<span>1985 → 2024 representation change</span></div><p class="list-note">A change in representation can reflect multiple input features. It is not a direct measurement of hydrologic change.</p>`;
  $('download-vector').onclick = () => download(`basin2vec-${b.id}-${state.spaceKey}-${state.period}.json`, JSON.stringify({usgs_id:b.id, name:b.name, representation:state.spaceKey, period:state.period, dimensions:64, normalization:'L2', values:Array.from(values)}, null, 2), 'application/json');
  const request = ++latestTrendRequest, index = state.selected, spaceKey = state.spaceKey;
  try {
    if (!trendCache.has(spaceKey)) trendCache.set(spaceKey, readGzip(`data/${spaceKey}/trend.f32.gz`, 'array').catch(e=>{trendCache.delete(spaceKey);throw e;}));
    const trend = await trendCache.get(spaceKey);
    if (request !== latestTrendRequest || state.tab !== 'signature' || state.selected !== index || state.spaceKey !== spaceKey || !$('trend')) return;
    const points = Array.from(trend.slice(index*40,(index+1)*40));
    const low = Math.max(-1, Math.min(...points)-.04), high = 1;
    const coordinates = points.map((v,j)=>`${12+j/39*266},${12+(1-v)/(high-low)*65}`).join(' ');
    $('trend').innerHTML = `<svg class="trend-chart" viewBox="0 0 290 95" role="img" aria-label="Annual cosine similarity to 1985, from ${low.toFixed(2)} to 1"><path d="M12 12H278M12 77H278" stroke="#344952" stroke-width="1" stroke-dasharray="3 4"/><polyline points="${coordinates}" fill="none" stroke="#91daba" stroke-width="1.7"/>${points.map((v,j)=>`<circle cx="${12+j/39*266}" cy="${12+(1-v)/(high-low)*65}" r="${Number(state.period)===1985+j?3.5:1.5}" fill="${Number(state.period)===1985+j?'#f1cc81':'#91daba'}"><title>${1985+j}: ${v.toFixed(6)}</title></circle>`).join('')}</svg><div class="trend-labels"><span>1985</span><span>Range ${low.toFixed(2)}–1.00</span><span>2024</span></div>`;
  } catch (error) { if ($('trend')) $('trend').textContent = error.message; }
}

function drawScatter() {
  if (!dataResult) return;
  const canvas = $('scatter'), rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (!rect.width) return;
  canvas.width = rect.width*dpr; canvas.height = rect.height*dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const limits = catalog.spaces[state.spaceKey].limits;
  scatterBounds = limits.slice(0,2).map(([a,b])=>[a-(b-a)*.18,b+(b-a)*.18]);
  ctx.strokeStyle='#20353f';ctx.lineWidth=.6;
  for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(rect.width*i/4,12);ctx.lineTo(rect.width*i/4,rect.height-15);ctx.stroke();ctx.beginPath();ctx.moveTo(12,rect.height*i/4);ctx.lineTo(rect.width-12,rect.height*i/4);ctx.stroke();}
  scatterPixels = [];
  for (let i=0;i<basins.length;i++) {
    const x = 12 + clamp((dataResult.points[i*3]-scatterBounds[0][0])/(scatterBounds[0][1]-scatterBounds[0][0]),0,1)*(rect.width-24);
    const y = 12 + (1-clamp((dataResult.points[i*3+1]-scatterBounds[1][0])/(scatterBounds[1][1]-scatterBounds[1][0]),0,1))*(rect.height-27);
    scatterPixels.push([x,y]);ctx.globalAlpha=.48;ctx.fillStyle=rgbColor(i);ctx.fillRect(x,y,1.6,1.6);
  }
  ctx.globalAlpha=1;
  if (state.selected !== null) {const [x,y]=scatterPixels[state.selected];ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle='#f7d89a';ctx.fill();ctx.strokeStyle='#101b23';ctx.lineWidth=1.4;ctx.stroke();ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.strokeStyle='#f7d89a';ctx.lineWidth=1;ctx.stroke();}
  $('variance').textContent = `${(catalog.spaces[state.spaceKey].variance.slice(0,2).reduce((a,b)=>a+b,0)*100).toFixed(1)}% variance · fixed mean basis`;
}
$('scatter').onclick = event => {
  const rect = $('scatter').getBoundingClientRect(), x = event.clientX-rect.left, y=event.clientY-rect.top;
  let closest=-1, distance=144;
  scatterPixels.forEach(([px,py],i)=>{const d=(x-px)**2+(y-py)**2;if(d<distance){distance=d;closest=i;}});
  if (closest>=0) selectBasin(closest); else toast('Click closer to a point, or use basin search.');
};
new ResizeObserver(drawScatter).observe($('scatter'));

function hitsAt(latlng) {
  const {lat, lng} = latlng;
  return sortedIndices.filter(i => {const b=boxes[i]; return lng>=b[0] && lat>=b[1] && lng<=b[2] && lat<=b[3] && containsPoint([lng,lat],geometry.features[i].geometry);});
}
function showPicker(hits) {
  $('overlap-picker').innerHTML = `<div class="picker-heading"><strong>${hits.length} basins at this location</strong><button id="close-picker" aria-label="Close basin chooser">×</button></div><p>Smallest first · choose the watershed to compare.</p>${hits.map(i=>`<button class="pick" data-select="${i}">${esc(title(basins[i].name))}<small>${number(basins[i].area)} km² · USGS ${basins[i].id}</small></button>`).join('')}`;
  $('overlap-picker').hidden = false; $('close-picker').onclick = () => $('overlap-picker').hidden = true; bindSelectButtons();
  $('overlap-picker').querySelector('.pick')?.focus({preventScroll:true});
}
map.on('click', event => {
  if (!geometry) return;
  $('search-results').hidden = true;
  const hits = hitsAt(event.latlng);
  if (hits.length > 1) showPicker(hits);
  else if (hits.length === 1) selectBasin(hits[0]);
  else {$('overlap-picker').hidden=true;toast('No mapped basin at this location. Try a colored basin or use search.');}
});
map.on('mousemove', event => {
  if (!geometry) return;
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = requestAnimationFrame(() => {
    const hits=hitsAt(event.latlng), tip=$('map-tooltip');
    if (!hits.length) {tip.hidden=true;return;}
    const i=hits[0], b=basins[i];
    tip.innerHTML=`${esc(title(b.name))}<small>${number(b.area)} km²${dataResult?.scores?` · cosine ${dataResult.scores[i].toFixed(3)}`:''}${hits.length>1?`<br>${hits.length} overlapping basins · click to choose`:''}</small>`;
    tip.hidden=false;tip.style.left=`${Math.min(event.containerPoint.x+14,$('map').clientWidth-275)}px`;tip.style.top=`${Math.min(event.containerPoint.y+14,$('map').clientHeight-95)}px`;
  });
});
map.on('mouseout movestart',()=> $('map-tooltip').hidden=true);
function toggleGauges() {
  if (gaugeLayer) {map.removeLayer(gaugeLayer);gaugeLayer=null;}
  if (!$('gauges').checked) return;
  gaugeLayer=L.layerGroup(basins.map((b,i)=>L.circleMarker(b.gauge,{radius:2.1,weight:.6,color:'#d7e9e4',fillColor:'#122934',fillOpacity:1,bubblingMouseEvents:false}).on('click',()=>selectBasin(i)))).addTo(map);
}
function search() {
  if (!basins) return;
  const query = $('search').value.toLowerCase().trim();
  if (!query) {$('search-results').hidden=true;$('search').setAttribute('aria-expanded','false');return;}
  const words=query.split(/\s+/), matches=basins.map((b,i)=>({b,i})).filter(({b})=>words.every(word=>`${b.id} ${b.name} ${b.state} ${b.huc}`.toLowerCase().includes(word))).sort((a,b)=>Number(b.b.id===query)-Number(a.b.id===query)).slice(0,30);
  $('search-results').innerHTML=matches.length?matches.map(({b,i})=>`<button data-search="${i}">${esc(title(b.name))}<small>${b.id} · ${esc(b.state)} · ${number(b.area)} km²</small></button>`).join(''):'<p>No matching basins. Try a USGS ID, river name, or two-letter state code.</p>';
  $('search-results').hidden=false;$('search').setAttribute('aria-expanded','true');
  document.querySelectorAll('[data-search]').forEach(b=>b.onclick=()=>selectBasin(Number(b.dataset.search),{zoom:true}));
}
$('search').oninput=search;
$('search').onkeydown=event=>{if(event.key==='ArrowDown'){event.preventDefault();$('search-results').querySelector('button')?.focus();}if(event.key==='Enter')$('search-results').querySelector('button')?.click();};
$('search-results').onkeydown=event=>{if(event.key==='ArrowDown'){event.preventDefault();event.target.nextElementSibling?.focus();}if(event.key==='ArrowUp'){event.preventDefault();(event.target.previousElementSibling||$('search')).focus();}};
document.addEventListener('keydown',event=>{if(event.key==='Escape'){$('search-results').hidden=true;$('search').setAttribute('aria-expanded','false');$('overlap-picker').hidden=true;}if(event.key==='/'&&!['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)&&!$('about-dialog').open){event.preventDefault();$('search').focus();}});
document.addEventListener('click',event=>{if(!event.target.closest('.search-wrap')){$('search-results').hidden=true;$('search').setAttribute('aria-expanded','false');}});
$('reset-map').onclick=resetMap;
$('outlines').onchange=()=>catalog&&renderMap();$('gauges').onchange=()=>catalog&&toggleGauges();
$('space').onchange=event=>{state.spaceKey=event.target.value;requestEmbedding();};
let yearTimer;
$('year').oninput=event=>{state.period=event.target.value;$('year-label').textContent=state.period;$('mean').classList.remove('active');setBusy(true);dataResult=null;clearTimeout(yearTimer);yearTimer=setTimeout(requestEmbedding,160);};
$('mean').onclick=()=>{clearTimeout(yearTimer);state.period='avg';requestEmbedding();};
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>catalog&&setMode(button.dataset.mode));
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>{state.tab=button.dataset.tab;document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',b===button);});if(catalog)renderDetails();});
$('about').onclick=()=>$('about-dialog').showModal();$('close-about').onclick=()=>$('about-dialog').close();
$('about-dialog').onclick=event=>{if(event.target===$('about-dialog')){const r=$('about-dialog').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)$('about-dialog').close();}};
function download(name, content, type) {const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export').onclick=()=>{
  if (!dataResult?.scores || state.selected===null || state.busy) return;
  const anchor=basins[state.selected], quote=value=>`"${String(value).replace(/"/g,'""')}"`;
  const header=['rank','reference_usgs_id','usgs_id','name','state','area_km2','centroid_distance_km','cosine_similarity','representation','period','exclude_recorded_strong_overlaps','minimum_distance_km'];
  const rows=neighbors().map((r,j)=>[j+1,anchor.id,basins[r.index].id,basins[r.index].name,basins[r.index].state,basins[r.index].area,r.distance.toFixed(3),r.score.toFixed(8),state.spaceKey,state.period,state.excludeOverlap,state.distance].map(quote).join(','));
  download(`basin2vec-neighbors-${anchor.id}-${state.spaceKey}-${state.period}.csv`,[header.join(','),...rows].join('\r\n'),'text/csv');toast(`Exported ${number(rows.length)} ranked basin comparisons.`);
};

async function init() {
  try {
    const [cat, geo, states] = await Promise.all([readGzip('data/catalog.json.gz'),readGzip('data/basins.geojson.gz'),fetch('vendor/states.json').then(r=>{if(!r.ok)throw Error('State outlines unavailable');return r.json();})]);
    catalog=cat;basins=cat.basins;geometry=geo;
    if(geo.features.length!==basins.length)throw Error('Boundary count does not match the embedding catalog.');
    const continental={...states,features:states.features.filter(f=>!['Alaska','Hawaii','Puerto Rico'].includes(f.properties.name))};
    L.geoJSON(continental,{pane:'base',interactive:false,style:{color:'#344b59',weight:1,fillColor:'#1a2b36',fillOpacity:1}}).addTo(map);
    const stateAbbr={'Washington':'WA','Oregon':'OR','California':'CA','Nevada':'NV','Idaho':'ID','Montana':'MT','Wyoming':'WY','Utah':'UT','Arizona':'AZ','Colorado':'CO','New Mexico':'NM','North Dakota':'ND','South Dakota':'SD','Nebraska':'NE','Kansas':'KS','Oklahoma':'OK','Texas':'TX','Minnesota':'MN','Iowa':'IA','Missouri':'MO','Arkansas':'AR','Louisiana':'LA','Wisconsin':'WI','Illinois':'IL','Mississippi':'MS','Michigan':'MI','Indiana':'IN','Ohio':'OH','Kentucky':'KY','Tennessee':'TN','Alabama':'AL','Georgia':'GA','Florida':'FL','South Carolina':'SC','North Carolina':'NC','Virginia':'VA','West Virginia':'WV','Pennsylvania':'PA','New York':'NY','Maine':'ME','Vermont':'VT','New Hampshire':'NH','Massachusetts':'MA','New Jersey':'NJ','Maryland':'MD','Connecticut':'CT','Rhode Island':'RI','Delaware':'DE'};
    continental.features.forEach(f=>{const bounds=L.geoJSON(f).getBounds(),label=stateAbbr[f.properties.name];if(label)L.marker(bounds.getCenter(),{pane:'labels',interactive:false,icon:L.divIcon({className:'state-label',html:label,iconSize:[20,10]})}).addTo(map);});
    [['Seattle',47.61,-122.33],['San Francisco',37.77,-122.42],['Denver',39.74,-104.99],['Dallas',32.78,-96.8],['Chicago',41.88,-87.63],['Atlanta',33.75,-84.39],['New York',40.71,-74.0]].forEach(([name,lat,lon])=>L.marker([lat,lon],{pane:'labels',interactive:false,icon:L.divIcon({className:'place-label',html:name,iconSize:[65,12],iconAnchor:[-4,-3]})}).addTo(map));
    sortedIndices=basins.map((b,i)=>i).sort((a,b)=>basins[a].area-basins[b].area);
    // Large polygons first: small nested basins keep their own visible fill.
    sortedIndices.slice().reverse().forEach(i=>{
      const layer=L.geoJSON(geo.features[i],{renderer,interactive:false,smoothFactor:0.5,style:{fillColor:'#548e89',fillOpacity:.82,color:'#122831',weight:.65}}).addTo(map);
      layers[i]=layer;const b=layer.getBounds();boxes[i]=[b.getWest(),b.getSouth(),b.getEast(),b.getNorth()];
    });
    const params=new URLSearchParams(location.hash.slice(1));
    if(cat.spaces[params.get('space')])state.spaceKey=params.get('space');
    if(params.get('year')==='avg'||cat.years.includes(Number(params.get('year'))))state.period=params.get('year');
    const anchor=basins.findIndex(b=>b.id===params.get('basin'));if(anchor>=0)state.selected=anchor;
    if(['embedding','similarity','change'].includes(params.get('view')))state.mode=params.get('view');
    if(state.mode==='similarity'&&state.selected===null)state.mode='embedding';
    $('space').value=state.spaceKey;if(state.period!=='avg')$('year').value=state.period;
    document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.mode===state.mode);b.setAttribute('aria-pressed',b.dataset.mode===state.mode);});
    requestEmbedding();
    // Read-only hooks support reproducible numerical and geometry QA.
    window.basinAtlas={getState:()=>({...state}),getScores:()=>dataResult?.scores,getPoints:()=>dataResult?.points,getBasins:()=>basins,hitsAt:([lat,lng])=>hitsAt({lat,lng}),map,selectBasin};
  } catch(error) {showError(error.message);}
}
init();
