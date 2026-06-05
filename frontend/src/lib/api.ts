// Typed client for the Sentinel FastAPI backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export interface RiskScore {
  region_id: number;
  region_name: string;
  date: string;
  ensemble_score: number;
  cnn_score: number | null;
  lstm_score: number | null;
  model_version: string | null;
  computed_at: string | null;
}

export interface RegionProps {
  region_id: number;
  region_name: string;
  date: string | null;
  ensemble_score: number | null;
  cnn_score: number | null;
  lstm_score: number | null;
}

export interface RegionFeature {
  type: "Feature";
  geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon;
  properties: RegionProps;
}

export interface RegionCollection {
  type: "FeatureCollection";
  features: RegionFeature[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export function fetchRegionsGeoJson(): Promise<RegionCollection> {
  return getJson<RegionCollection>("/risk/geojson");
}

export function fetchRisk(region: string, date?: string): Promise<RiskScore> {
  const q = new URLSearchParams({ region });
  if (date) q.set("date", date);
  return getJson<RiskScore>(`/risk?${q.toString()}`);
}

export interface PointRisk {
  lat: number;
  lon: number;
  date: string;
  ensemble_score: number;
  cnn_score: number | null;
  lstm_score: number | null;
}

export function fetchPoint(lat: number, lon: number, date?: string): Promise<PointRisk> {
  const q = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  if (date) q.set("date", date);
  return getJson<PointRisk>(`/risk/point?${q.toString()}`);
}
