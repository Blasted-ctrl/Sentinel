"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
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
import type { MapMark } from "@/components/MapView";
import { formatScore, riskBand, riskColor } from "@/lib/risk";

const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => <div className="empty">Loading map…</div>,
});

const MARK_TTL_MS = 10_000;

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

interface Mark {
  id: number;
  lat: number;
  lon: number;
  ensemble: number | null;
  cnn: number | null;
  lstm: number | null;
  loading: boolean;
}

interface DetailView {
  name: string;
  ensemble: number | null;
  cnn: number | null;
  lstm: number | null;
  loading: boolean;
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
        <div className="bar-fill" style={{ width: `${pct}%`, background: riskColor(value) }} />
      </div>
    </div>
  );
}

export function Dashboard() {
  const [data, setData] = useState<RegionCollection | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [date, setDate] = useState<string>(todayISO());
  const [detail, setDetail] = useState<RiskScore | null>(null);
  const [marks, setMarks] = useState<Mark[]>([]);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

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

  // Clear any pending mark-removal timers on unmount.
  useEffect(() => {
    const map = timers.current;
    return () => {
      map.forEach((t) => clearTimeout(t));
      map.clear();
    };
  }, []);

  const selected: RegionProps | null = useMemo(() => {
    if (!data || selectedId === null) return null;
    return data.features.find((f) => f.properties.region_id === selectedId)?.properties ?? null;
  }, [data, selectedId]);

  useEffect(() => {
    if (!selected) return;
    let active = true;
    fetchRisk(selected.region_name, date)
      .then((s) => active && setDetail(s))
      .catch(() => active && setDetail(null));
    return () => {
      active = false;
    };
  }, [selected, date]);

  const pickPoint = (lat: number, lon: number) => {
    const id = Date.now() + Math.floor(Math.random() * 1000);
    setSelectedId(null);
    setMarks((m) => [
      ...m,
      { id, lat, lon, ensemble: null, cnn: null, lstm: null, loading: true },
    ]);
    fetchPoint(lat, lon, date)
      .then((p) =>
        setMarks((m) =>
          m.map((x) =>
            x.id === id
              ? {
                  ...x,
                  ensemble: p.ensemble_score,
                  cnn: p.cnn_score,
                  lstm: p.lstm_score,
                  loading: false,
                }
              : x,
          ),
        ),
      )
      .catch(() =>
        setMarks((m) => m.map((x) => (x.id === id ? { ...x, loading: false } : x))),
      )
      .finally(() => {
        const t = setTimeout(() => {
          setMarks((m) => m.filter((x) => x.id !== id));
          timers.current.delete(id);
        }, MARK_TTL_MS);
        timers.current.set(id, t);
      });
  };

  const activeMark = marks.length > 0 ? marks[marks.length - 1]! : null;

  const view: DetailView | null = activeMark
    ? {
        name: `Point  ${activeMark.lat.toFixed(3)}, ${activeMark.lon.toFixed(3)}`,
        ensemble: activeMark.ensemble,
        cnn: activeMark.cnn,
        lstm: activeMark.lstm,
        loading: activeMark.loading,
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

  const mapMarks: MapMark[] = marks.map((m) => ({
    id: m.id,
    lat: m.lat,
    lon: m.lon,
    score: m.ensemble,
    loading: m.loading,
  }));

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
              marks={mapMarks}
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
              <MapPin size={11} /> Click anywhere to drop a risk pin
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
                    onClick={() => {
                      setSelectedId(p.region_id);
                    }}
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
