// One series over time as a line (dataviz skill: 2px line, end dot with a
// surface ring, the latest value labelled at the end, hairline grid, a
// crosshair that snaps to the nearest day on hover and on keyboard focus).
import { useState, type KeyboardEvent, type PointerEvent } from "react";

export interface Point {
  label: string;
  value: number;
}

const W = 480;
const H = 160;
const M = { top: 12, right: 64, bottom: 24, left: 48 };

/** Round step sizes: 1, 2, 2.5 or 5 times a power of ten. */
export function niceTicks(lo: number, hi: number, count = 4): number[] {
  const span = hi - lo || Math.abs(hi) || 1;
  const rough = span / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= rough) ?? 10 * power;
  const ticks: number[] = [];
  for (let tick = Math.ceil(lo / step) * step; tick <= hi + step * 1e-9; tick += step) {
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

/** The values' range with a tenth of it as margin above and below. */
export function domain(values: readonly number[]): [number, number] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const margin = (hi - lo || Math.abs(hi) || 1) * 0.1;
  return [lo - margin, hi + margin];
}

export function LineChart({
  title,
  points,
  format,
  axis = format,
}: {
  title: string;
  points: readonly Point[];
  /** A value as the end label and the tooltip show it. */
  format: (value: number) => string;
  /** A tick on the y axis (shorter, by default the same). */
  axis?: (value: number) => string;
}) {
  const [active, setActive] = useState<number | null>(null);
  if (points.length === 0) return null;
  const [lo, hi] = domain(points.map((p) => p.value));
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const x = (i: number) =>
    M.left + (points.length === 1 ? plotW / 2 : (i * plotW) / (points.length - 1));
  const y = (v: number) => M.top + plotH - ((v - lo) / (hi - lo)) * plotH;
  const last = points.length - 1;
  const end = points[last]!;
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`).join("");

  function pointAt(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width === 0) return;
    const at = ((event.clientX - box.left) / box.width) * W;
    let nearest = 0;
    points.forEach((_, i) => {
      if (Math.abs(x(i) - at) < Math.abs(x(nearest) - at)) nearest = i;
    });
    setActive(nearest);
  }

  function step(event: KeyboardEvent<SVGSVGElement>) {
    const moves: Record<string, (i: number) => number> = {
      ArrowLeft: (i) => Math.max(0, i - 1),
      ArrowRight: (i) => Math.min(last, i + 1),
      Home: () => 0,
      End: () => last,
    };
    const move = moves[event.key];
    if (move === undefined) return;
    event.preventDefault();
    setActive((i) => move(i ?? last));
  }

  const shown = active === null ? null : points[active]!;
  return (
    <figure className="line-chart">
      <figcaption>{title}</figcaption>
      <div className="plot">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          tabIndex={0}
          aria-label={`${title}, ${points[0]!.label} to ${end.label}: latest ${format(end.value)}. Arrow keys step through the days.`}
          onPointerMove={pointAt}
          onPointerLeave={() => setActive(null)}
          onFocus={() => setActive((i) => i ?? last)}
          onBlur={() => setActive(null)}
          onKeyDown={step}
        >
          {niceTicks(lo, hi).map((tick) => (
            <g key={tick} className="tick">
              <line x1={M.left} x2={W - M.right} y1={y(tick)} y2={y(tick)} />
              <text x={M.left - 6} y={y(tick)} dy="0.32em" textAnchor="end">
                {axis(tick)}
              </text>
            </g>
          ))}
          <text className="day" x={x(0)} y={H - 6} textAnchor="start">
            {points[0]!.label}
          </text>
          {last > 0 && (
            <text className="day" x={x(last)} y={H - 6} textAnchor="end">
              {end.label}
            </text>
          )}
          <path className="series" d={path} />
          {shown !== null && active !== null && (
            <line className="crosshair" x1={x(active)} x2={x(active)} y1={M.top} y2={M.top + plotH} />
          )}
          <circle className="dot" cx={x(active ?? last)} cy={y((shown ?? end).value)} r={4} />
          <text className="end" x={x(last) + 10} y={y(end.value)} dy="0.32em">
            {format(end.value)}
          </text>
        </svg>
        {shown !== null && active !== null && (
          <div className="tooltip" role="status" style={{ left: `${(x(active) / W) * 100}%` }}>
            <strong>{format(shown.value)}</strong> <span>{shown.label}</span>
          </div>
        )}
      </div>
    </figure>
  );
}
