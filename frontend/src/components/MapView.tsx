"use client";

import "leaflet/dist/leaflet.css";

import L from "leaflet";
import type { Layer, LeafletMouseEvent, PathOptions } from "leaflet";
import { Marker, GeoJSON, MapContainer, TileLayer, useMapEvents } from "react-leaflet";

import type { RegionCollection, RegionProps } from "@/lib/api";
import { formatScore, riskColor } from "@/lib/risk";

export interface MapMark {
  id: number;
  lat: number;
  lon: number;
  score: number | null;
  loading: boolean;
}

function markIcon(mark: MapMark): L.DivIcon {
  const color = riskColor(mark.score);
  const html = mark.loading
    ? `<div class="sm-wrap sm-loading"><div class="sm-pin" style="--c:${color}"></div></div>`
    : `<div class="sm-wrap">
         <div class="sm-label" style="--c:${color}">${formatScore(mark.score)}</div>
         <div class="sm-stem" style="--c:${color}"></div>
         <div class="sm-pin" style="--c:${color}"></div>
       </div>`;
  return L.divIcon({
    html,
    className: "sm-icon",
    iconSize: [96, 58],
    iconAnchor: [48, 58],
  });
}

function ClickHandler({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e: LeafletMouseEvent) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

interface Props {
  data: RegionCollection;
  selectedId: number | null;
  marks: MapMark[];
  onPick: (lat: number, lon: number) => void;
}

export default function MapView({ data, selectedId, marks, onPick }: Props) {
  const style = (feature?: GeoJSON.Feature): PathOptions => {
    const props = feature?.properties as RegionProps | undefined;
    const active = props?.region_id === selectedId;
    return {
      color: active ? "#9a3412" : "#ffffff",
      weight: active ? 2.5 : 0.8,
      fillColor: riskColor(props?.ensemble_score ?? null),
      fillOpacity: active ? 0.85 : 0.6,
    };
  };

  const onEachFeature = (feature: GeoJSON.Feature, layer: Layer) => {
    const props = feature.properties as RegionProps;
    layer.bindTooltip(props.region_name, { sticky: true, direction: "top" });
    layer.on("click", (e: LeafletMouseEvent) => onPick(e.latlng.lat, e.latlng.lng));
  };

  return (
    <MapContainer
      center={[25, 5]}
      zoom={2}
      minZoom={2}
      worldCopyJump
      scrollWheelZoom
      style={{ height: "100%", width: "100%" }}
      preferCanvas
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        attribution="&copy; OpenStreetMap contributors &copy; CARTO"
        maxZoom={19}
      />
      <GeoJSON
        key={`${data.features.length}-${selectedId ?? "none"}`}
        data={data as unknown as GeoJSON.GeoJsonObject}
        style={style}
        onEachFeature={onEachFeature}
      />
      {marks.map((m) => (
        <Marker key={m.id} position={[m.lat, m.lon]} icon={markIcon(m)} interactive={false} />
      ))}
      <ClickHandler onPick={onPick} />
    </MapContainer>
  );
}
