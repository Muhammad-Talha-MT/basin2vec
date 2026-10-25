import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {cosineScores, containsPoint, distanceKm} from '../dist/math.js';

const ring = [[0,0],[10,0],[10,10],[0,10],[0,0]];
test('polygon selection preserves holes, edges, nested hits and multipolygon parts', () => {
  const parent = {type:'Polygon',coordinates:[ring,[[4,4],[6,4],[6,6],[4,6],[4,4]]]};
  const child = {type:'Polygon',coordinates:[[[1,1],[3,1],[3,3],[1,3],[1,1]]]};
  assert.equal(containsPoint([5,5],parent),false);
  assert.equal(containsPoint([0,5],parent),true);
  assert.equal(containsPoint([4,5],parent),true);
  assert.deepEqual([parent,child].filter(g=>containsPoint([2,2],g)),[parent,child]);
  assert.equal(containsPoint([20,20],parent),false);
  assert.equal(containsPoint([2,2],{type:'MultiPolygon',coordinates:[parent.coordinates,child.coordinates]}),true);
});
test('cosine uses all coordinates and is independent of vector magnitude',()=>{
  const scores=cosineScores(new Float32Array([2,0,0,3,-1,0,4,0]),2,0);
  assert.deepEqual(Array.from(scores),[1,0,-1,1]);
  assert.equal(distanceKm({lat:0,lon:0},{lat:0,lon:0}),0);
  assert.ok(Math.abs(distanceKm({lat:0,lon:0},{lat:0,lon:1})-111.195)<.001);
});
const catalog=JSON.parse(gunzipSync(readFileSync(new URL('../dist/data/catalog.json.gz',import.meta.url))));
const anchor=catalog.basins.findIndex(b=>b.id==='04101500');
const cases=[
  ['embeddings','avg',['04183000','03282290','01154500','05552500','03284500'],[.95626390,.88034564,.87837577,.87054002,.86996645]],
  ['hydrologic_repr','avg',['03282290','03282060','01029500','11527000','08136000'],[.98725289,.97232175,.96677119,.96608227,.96335053]],
  ['embeddings','2024',['03282290','03282000','03282060','01154500','04113000'],[.95343810,.92407668,.90039033,.87342769,.84182853]],
  ['hydrologic_repr','2024',['03282290','03282000','05430500','07297910','07232500'],[.99417150,.98769236,.98375177,.98142469,.97752440]],
];
for(const [space,period,ids,expected] of cases)test(`${space}/${period}: full-vector ranking agrees with independent NumPy results`,()=>{
  const bytes=gunzipSync(readFileSync(new URL(`../dist/data/${space}/${period}.f32.gz`,import.meta.url)));
  const vectors=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
  assert.equal(vectors.length,9067*64);
  const scores=cosineScores(vectors,64,anchor);
  const ranks=Array.from(scores,(s,i)=>({s,i})).filter(r=>r.i!==anchor).sort((a,b)=>b.s-a.s).slice(0,5);
  assert.deepEqual(ranks.map(r=>catalog.basins[r.i].id),ids);
  ranks.forEach((r,j)=>assert.ok(Math.abs(r.s-expected[j])<1e-6));
  assert.equal(scores[anchor],1);
});
test('catalog, boundary IDs and strong-overlap relationships align',()=>{
  const geometry=JSON.parse(gunzipSync(readFileSync(new URL('../dist/data/basins.geojson.gz',import.meta.url))));
  assert.equal(catalog.basins.length,9067);assert.equal(geometry.features.length,9067);
  assert.equal(catalog.years.length,40);assert.equal(catalog.years[0],1985);assert.equal(catalog.years.at(-1),2024);
  let edges=0;
  catalog.basins.forEach((b,i)=>{
    assert.equal(geometry.features[i].id,i);
    assert.ok(/^\d{8,15}$/.test(b.id));
    b.overlaps.forEach(([j,score])=>{assert.ok(catalog.basins[j].overlaps.some(([k,s])=>k===i&&s===score));edges++;});
  });
  assert.equal(edges,34291*2);
});
