"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  CalendarDays,
  Flame,
  Layers,
  Loader2,
  Satellite,
  ThermometerSun,
} from "lucide-react";

import {
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

export function Dashboard() {
  const [data, setData] = useState<RegionCollection | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [date, setDate] = useState<string>(todayISO());
  const [detail, setDetail] = useState<RiskScore | null>(null);

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

  const ensemble = detail?.ensemble_score ?? selected?.ensemble_score ?? null;
  const cnn = detail?.cnn_score ?? selected?.cnn_score ?? null;
  const lstm = detail?.lstm_score ?? selected?.lstm_score ?? null;
  const band = riskBand(ensemble);

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
            <MapView data={data} selectedId={selectedId} onSelect={setSelectedId} />
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
          </div>
        </div>

        <aside className="sidebar">
          <section className="detail-card">
            {selected ? (
              <>
                <div className="score-hero">
                  <span className="score-num mono" style={{ color: riskColor(ensemble) }}>
                    {formatScore(ensemble)}
                  </span>
                  <span className="score-unit">/ 100</span>
                  <span
                    className="band-chip"
                    style={{ background: riskColor(ensemble), marginLeft: "auto" }}
                  >
                    {band}
                  </span>
                </div>
                <div className="region-name">{selected.region_name}</div>
                <div className="bar-label" style={{ marginTop: 4 }}>
                  <span>Ensemble · CNN imagery + LSTM climate</span>
                </div>
                <ScoreBar label="Ensemble" value={ensemble} icon={<Activity size={13} />} />
                <ScoreBar
                  label="Climate (LSTM)"
                  value={lstm}
                  icon={<ThermometerSun size={13} />}
                />
                <ScoreBar label="Imagery (CNN)" value={cnn} icon={<Satellite size={13} />} />
              </>
            ) : (
              <div className="empty">
                {status === "ready"
                  ? "No regions scored yet. Run the ingestion + scoring pipeline to populate the map."
                  : "—"}
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
                    onClick={() => setSelectedId(p.region_id)}
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
