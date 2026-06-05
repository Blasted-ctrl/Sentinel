"use client";

import "leaflet/dist/leaflet.css";

import type { Layer, PathOptions } from "leaflet";
import {
  CircleMarker,
  GeoJSON,
  MapContainer,
  TileLayer,
  Tooltip,
  useMapEvents,
} from "react-leaflet";

import type { RegionCollection, RegionProps } from "@/lib/api";
import { riskColor } from "@/lib/risk";

function ClickHandler({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export interface PickedPoint {
  lat: number;
  lon: number;
  score: number | null;
}

interface Props {
  data: RegionCollection;
  selectedId: number | null;
  picked: PickedPoint | null;
  onSelect: (id: number) => void;
  onPick: (lat: number, lon: number) => void;
}

export default function MapView({ data, selectedId, picked, onSelect, onPick }: Props) {
  const style = (feature?: GeoJSON.Feature): PathOptions => {
    const props = feature?.properties as RegionProps | undefined;
    const active = props?.region_id === selectedId;
    return {
      color: active ? "#9a3412" : "#ffffff",
      weight: active ? 2.5 : 0.8,
      fillColor: riskColor(props?.ensemble_score ?? null),
      fillOpacity: active ? 0.85 : 0.62,
    };
  };

  const onEachFeature = (feature: GeoJSON.Feature, layer: Layer) => {
    const props = feature.properties as RegionProps;
    layer.on("click", () => onSelect(props.region_id));
    layer.bindTooltip(props.region_name, { sticky: true, direction: "top" });
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
      {picked && (
        <CircleMarker
          center={[picked.lat, picked.lon]}
          radius={9}
          pathOptions={{
            color: "#1a1714",
            weight: 2,
            fillColor: riskColor(picked.score),
            fillOpacity: 0.95,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -8]}>
            {picked.lat.toFixed(2)}, {picked.lon.toFixed(2)}
          </Tooltip>
        </CircleMarker>
      )}
      <ClickHandler onPick={onPick} />
    </MapContainer>
  );
}
