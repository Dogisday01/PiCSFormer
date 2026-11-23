import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { pretty, family } from '../chords';
import { useI18n } from '../i18n';
import type { KeySegment, Segment } from '../types';
import { keyName } from '../ui';

export function Timeline({ segments, keys, duration, time, active, onSeek }: {
  segments: Segment[]; keys: KeySegment[]; duration: number; time: number; active: number; onSeek: (t: number) => void;
}) {
  const { m, clock } = useI18n();
  const [zoom, setZoom] = useState(() => (duration > 120 ? 4 : duration > 45 ? 2 : 1));
  const scroll = useRef<HTMLDivElement>(null);
  const width = `${zoom * 100}%`;
  const pct = (t: number) => `${(100 * t) / Math.max(duration, 1e-6)}%`;

  useEffect(() => {
    const box = scroll.current;
    if (!box || zoom === 1) return;
    const x = (time / Math.max(duration, 1e-6)) * box.scrollWidth;
    if (x < box.scrollLeft + 40 || x > box.scrollLeft + box.clientWidth - 40) box.scrollLeft = x - box.clientWidth / 3;
  }, [time, zoom, duration]);

  function seek(e: MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(duration, ((e.clientX - rect.left) / rect.width) * duration)));
  }

  const ticks = 5;
  return <div className="timeline">
    <div className="card-head">
      <h2>{m.report.timeline}</h2>
      <div className="zoom" role="group" aria-label="zoom">
        {[1, 2, 4, 8].map(z => <button key={z} type="button" className={`btn small ${z === zoom ? '' : 'ghost'}`} aria-pressed={z === zoom} onClick={() => setZoom(z)}>×{z}</button>)}
      </div>
    </div>
    <div className="tl-scroll" ref={scroll}>
      <div style={{ width, minWidth: '100%', position: 'relative' }}>
        <div className="tl-track" onClick={seek} role="slider" tabIndex={0} aria-label={m.report.playhead} aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(time)}
          onKeyDown={e => { if (e.key === 'ArrowRight') onSeek(Math.min(duration, time + 2)); if (e.key === 'ArrowLeft') onSeek(Math.max(0, time - 2)); }}>
          {segments.map((s, i) => <div key={i} className={`tl-seg${i === active ? ' active' : ''}`} data-family={family(s.label)}
            style={{ left: pct(s.start), width: pct(s.end - s.start) }} title={`${pretty(s.label)} · ${clock(s.start)}–${clock(s.end)}`}>
            {pretty(s.label)}
          </div>)}
          <div className="tl-head" style={{ left: pct(time) }} />
        </div>
        {keys.length > 0 && <div className="tl-keys" aria-label={m.report.keyStrip}>
          {keys.map((k, i) => <div key={i} className="tl-key" style={{ left: pct(k.start), width: pct(k.end - k.start) }} title={keyName(k, m.keyMode)}>{keyName(k, m.keyMode)}</div>)}
        </div>}
        <div className="tl-axis">{Array.from({ length: ticks + 1 }, (_, i) => <span key={i}>{clock((duration * i) / ticks)}</span>)}</div>
      </div>
    </div>
  </div>;
}
