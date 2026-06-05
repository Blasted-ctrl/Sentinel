"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  CalendarDays,
  Flame,
  Layers,
  Loader2,
  MapPin,
  Satellite,
  ThermometerSun,
} from "lucide-react";

import {
  fetchPoint,
  fetchRegionsGeoJson,
  fetchRisk,
  type RegionCollection,
  type RegionProps,
  type RiskScore,
} from "@/lib/api";
import { formatScore, riskBand, riskColor } from "@/lib/risk";

const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => <div className="empty">Loading map…</div>,
});

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

function ScoreBar({
  label,
  value,
  icon,
}: {
  label: string;
  value: number | null;
  icon: React.ReactNode;
}) {
  const pct = value === null ? 0 : Math.round(value * 100);
  return (
    <div className="bar-row">
      <div className="bar-label">
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          {icon}
          {label}
        </span>
        <span className="mono">{value === null ? "—" : `${pct}`}</span>
      </div>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{ width: `${pct}%`, background: riskColor(value) }}
        />
      </div>
    </div>
  );
}

interface DetailView {
  name: string;
  ensemble: number | null;
  cnn: number | null;
  lstm: number | null;
  loading: boolean;
}

export function Dashboard() {
  const [data, setData] = useState<RegionCollection | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [date, setDate] = useState<string>(todayISO());
  const [detail, setDetail] = useState<RiskScore | null>(null);

  // Click-to-predict (any point on Earth).
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [pointResult, setPointResult] = useState<RiskScore | null>(null);
  const [pointLoading, setPointLoading] = useState(false);

  useEffect(() => {
    let active = true;
    fetchRegionsGeoJson()
      .then((fc) => {
        if (!active) return;
        setData(fc);
        setStatus("ready");
        const top = [...fc.features].sort(
          (a, b) => (b.properties.ensemble_score ?? -1) - (a.properties.ensemble_score ?? -1),
        )[0];
        if (top) setSelectedId(top.properties.region_id);
      })
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  const selected: RegionProps | null = useMemo(() => {
    if (!data || selectedId === null) return null;
    return data.features.find((f) => f.properties.region_id === selectedId)?.properties ?? null;
  }, [data, selectedId]);

  // Refresh the selected region's score when the region or date changes.
  useEffect(() => {
    if (!selected || coords) return;
    let active = true;
    fetchRisk(selected.region_name, date)
      .then((s) => active && setDetail(s))
      .catch(() => active && setDetail(null));
    return () => {
      active = false;
    };
  }, [selected, date, coords]);

  // Predict an arbitrary picked point (and re-predict when the date changes).
  useEffect(() => {
    if (!coords) return;
    let active = true;
    setPointLoading(true);
    setPointResult(null);
    fetchPoint(coords.lat, coords.lon, date)
      .then((p) => {
        if (!active) return;
        setPointResult({
          region_id: -1,
          region_name: "",
          date: p.date,
          ensemble_score: p.ensemble_score,
          cnn_score: p.cnn_score,
          lstm_score: p.lstm_score,
          model_version: null,
          computed_at: null,
        });
      })
      .catch(() => active && setPointResult(null))
      .finally(() => active && setPointLoading(false));
    return () => {
      active = false;
    };
  }, [coords, date]);

  const selectRegion = (id: number) => {
    setCoords(null);
    setPointResult(null);
    setSelectedId(id);
  };
  const pickPoint = (lat: number, lon: number) => {
    setSelectedId(null);
    setCoords({ lat, lon });
  };

  const view: DetailView | null = coords
    ? {
        name: `Point  ${coords.lat.toFixed(3)}, ${coords.lon.toFixed(3)}`,
        ensemble: pointResult?.ensemble_score ?? null,
        cnn: pointResult?.cnn_score ?? null,
        lstm: pointResult?.lstm_score ?? null,
        loading: pointLoading,
      }
    : selected
      ? {
          name: selected.region_name,
          ensemble: detail?.ensemble_score ?? selected.ensemble_score,
          cnn: detail?.cnn_score ?? selected.cnn_score,
          lstm: detail?.lstm_score ?? selected.lstm_score,
          loading: false,
        }
      : null;

  const sorted = useMemo(
    () =>
      data
        ? [...data.features].sort(
            (a, b) =>
              (b.properties.ensemble_score ?? -1) - (a.properties.ensemble_score ?? -1),
          )
        : [],
    [data],
  );

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">
            <Flame size={18} strokeWidth={2.4} />
          </span>
          <span className="brand-name">Sentinel</span>
          <span className="brand-sub">Wildfire Risk</span>
        </div>
        <div className="toolbar">
          <label className="date-field">
            <CalendarDays size={15} />
            <input
              type="date"
              value={date}
              max={todayISO()}
              onChange={(e) => setDate(e.target.value)}
              aria-label="Risk date"
            />
          </label>
        </div>
      </header>

      <div className="main">
        <div className="map-wrap">
          {status === "ready" && data ? (
            <MapView
              data={data}
              selectedId={selectedId}
              picked={coords ? { ...coords, score: pointResult?.ensemble_score ?? null } : null}
              onSelect={selectRegion}
              onPick={pickPoint}
            />
          ) : (
            <div className="empty" style={{ paddingTop: 80 }}>
              {status === "loading" ? "Loading regions…" : "Could not reach the API."}
            </div>
          )}
          <div className="map-badge">
            <div className="panel-title">
              <Layers size={13} /> Ignition risk
            </div>
            <div className="legend-bar" />
            <div className="legend-scale">
              <span>Low</span>
              <span>Moderate</span>
              <span>Extreme</span>
            </div>
            <div className="legend-hint">
              <MapPin size={11} /> Click anywhere to predict
            </div>
          </div>
        </div>

        <aside className="sidebar">
          <section className="detail-card">
            {view ? (
              <>
                <div className="score-hero">
                  <span className="score-num mono" style={{ color: riskColor(view.ensemble) }}>
                    {view.loading ? "··" : formatScore(view.ensemble)}
                  </span>
                  <span className="score-unit">/ 100</span>
                  <span
                    className="band-chip"
                    style={{ background: riskColor(view.ensemble), marginLeft: "auto" }}
                  >
                    {view.loading ? "Scoring…" : riskBand(view.ensemble)}
                  </span>
                </div>
                <div className="region-name">{view.name}</div>
                <div className="bar-label" style={{ marginTop: 4 }}>
                  <span>Ensemble · CNN imagery + LSTM climate</span>
                </div>
                <ScoreBar label="Ensemble" value={view.ensemble} icon={<Activity size={13} />} />
                <ScoreBar
                  label="Climate (LSTM)"
                  value={view.lstm}
                  icon={<ThermometerSun size={13} />}
                />
                <ScoreBar label="Imagery (CNN)" value={view.cnn} icon={<Satellite size={13} />} />
              </>
            ) : (
              <div className="empty">
                Select a region or click the map to predict wildfire risk anywhere.
              </div>
            )}
          </section>

          <section>
            <div className="panel-title">
              {status === "loading" && <Loader2 size={13} className="spin-icon" />}
              Regions by risk
            </div>
            <div className="region-list">
              {sorted.map((f) => {
                const p = f.properties;
                return (
                  <button
                    key={p.region_id}
                    className="region-row"
                    data-active={p.region_id === selectedId}
                    onClick={() => selectRegion(p.region_id)}
                  >
                    <span
                      className="region-dot"
                      style={{ background: riskColor(p.ensemble_score) }}
                    />
                    <span className="name">{p.region_name}</span>
                    <span className="val mono">{formatScore(p.ensemble_score)}</span>
                  </button>
                );
              })}
              {status === "ready" && sorted.length === 0 && (
                <div className="empty">No regions yet.</div>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
