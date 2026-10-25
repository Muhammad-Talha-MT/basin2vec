export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export function cosineScores(vectors, dim, index) {
  const out = new Float32Array(vectors.length / dim);
  for (let i = 0; i < out.length; i++) {
    let dot = 0, na = 0, nb = 0;
    for (let j = 0; j < dim; j++) {
      const a = vectors[index * dim + j], b = vectors[i * dim + j];
      dot += a * b; na += a * a; nb += b * b;
    }
    out[i] = clamp(dot / Math.sqrt(na * nb), -1, 1);
  }
  return out;
}
export function distanceKm(a, b) {
  const rad = Math.PI / 180, dl = (b.lat - a.lat) * rad, dn = (b.lon - a.lon) * rad;
  const h = Math.sin(dl / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dn / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(clamp(h, 0, 1)));
}
// Boundary counts as a hit. Holes and MultiPolygons are handled explicitly.
function inRing(point, ring) {
  let inside = false;
  const [x, y] = point;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    const cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi);
    if (Math.abs(cross) < 1e-10 && x >= Math.min(xi, xj) && x <= Math.max(xi, xj) && y >= Math.min(yi, yj) && y <= Math.max(yi, yj)) return 2;
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside ? 1 : 0;
}
export function containsPoint(point, geometry) {
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  return polygons.some(rings => {
    const outer = inRing(point, rings[0]);
    if (!outer) return false;
    if (outer === 2) return true;
    return !rings.slice(1).some(ring => inRing(point, ring) === 1);
  });
}
export async function readGzip(url, type = 'json') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  if (!('DecompressionStream' in globalThis)) throw new Error('This explorer needs a current browser with gzip stream support. Please update your browser.');
  const decoded = new Response(response.body.pipeThrough(new DecompressionStream('gzip')));
  return type === 'json' ? decoded.json() : new Float32Array(await decoded.arrayBuffer());
}
export function project(vectors, space) {
  const dim = space.dim, points = new Float32Array(vectors.length / dim * 3);
  for (let i = 0; i < vectors.length / dim; i++) for (let k = 0; k < 3; k++) {
    let score = 0;
    for (let j = 0; j < dim; j++) score += (vectors[i * dim + j] - space.mean[j]) * space.axes[k][j];
    points[i * 3 + k] = score;
  }
  return points;
}
