import {readGzip, cosineScores, project} from './math.js';
const cache = new Map();
async function vectorsFor(key) {
  if (!cache.has(key)) {
    const promise = readGzip(`data/${key}.f32.gz`, 'array').catch(error => {cache.delete(key); throw error;});
    cache.set(key, promise);
    // Bound browser memory while retaining recent years.
    if (cache.size > 5) cache.delete(cache.keys().next().value);
  }
  return cache.get(key);
}
self.onmessage = async ({data}) => {
  try {
    const {request, spaceKey, period, selected, space} = data;
    const vectors = await vectorsFor(`${spaceKey}/${period}`);
    if (vectors.length !== data.count * space.dim) throw new Error('Embedding dimensions do not match the basin catalog.');
    const points = project(vectors, space);
    const scores = selected === null ? null : cosineScores(vectors, space.dim, selected);
    const vector = selected === null ? null : vectors.slice(selected * space.dim, (selected + 1) * space.dim);
    self.postMessage({request, points, scores, vector}, [points.buffer, ...(scores ? [scores.buffer, vector.buffer] : [])]);
  } catch (error) {
    self.postMessage({request: data.request, error: error.message});
  }
};
