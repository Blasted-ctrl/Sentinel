"use client";

import "leaflet/dist/leaflet.css";

import L from "leaflet";
import type { Layer, PathOptions } from "leaflet";
import { useEffect } from "react";
import { GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";

import type { RegionCollection, RegionProps } from "@/lib/api";
import { riskColor } from "@/lib/risk";

function FitBounds({ data }: { data: RegionCollection }) {
  const map = useMap();
  useEffect(() => {
    if (data.features.length === 0) return;
    const layer = L.geoJSON(data as unknown as GeoJSON.GeoJsonObject);
    const bounds = layer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [32, 32], maxZoom: 9 });
    }
  }, [data, map]);
  return null;
}

interface Props {
  data: RegionCollection;
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function MapView({ data, selectedId, onSelect }: Props) {
  const style = (feature?: GeoJSON.Feature): PathOptions => {
    const props = feature?.properties as RegionProps | undefined;
    const active = props?.region_id === selectedId;
    return {
      color: active ? "#9a3412" : "#ffffff",
      weight: active ? 2.5 : 1,
      fillColor: riskColor(props?.ensemble_score ?? null),
      fillOpacity: active ? 0.85 : 0.7,
    };
  };

  const onEachFeature = (feature: GeoJSON.Feature, layer: Layer) => {
    const props = feature.properties as RegionProps;
    layer.on("click", () => onSelect(props.region_id));
    layer.bindTooltip(props.region_name, { sticky: true, direction: "top" });
  };

  return (
    <MapContainer
      center={[39, -121]}
      zoom={6}
      scrollWheelZoom
      style={{ height: "100%", width: "100%" }}
      preferCanvas
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; OpenStreetMap contributors &copy; CARTO'
        maxZoom={19}
      />
      <GeoJSON
        key={`${data.features.length}-${selectedId ?? "none"}`}
        data={data as unknown as GeoJSON.GeoJsonObject}
        style={style}
        onEachFeature={onEachFeature}
      />
      <FitBounds data={data} />
    </MapContainer>
  );
}
