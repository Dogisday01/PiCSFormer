import { useEffect, useMemo, useRef, useState } from 'react';
import { errorCode, request, upload } from '../api';
import { family, segmentAt, validateSegments } from '../chords';
import { fill, useI18n } from '../i18n';
import type { Report, Segment } from '../types';
import { useResource } from '../use-resource';
import { Badge, Card, Chord, ErrorBox, Legend, Loading, Page, keyName } from '../ui';
import { Timeline } from './Timeline';

type Tab = 'sheet' | 'table' | 'edit' | 'evaluate' | 'export';
const LEVELS = ['root', 'majmin', 'triads', 'sevenths', 'tetrads', 'mirex'] as const;

export function ReportPage({ id }: { id: string }) {
  const { m } = useI18n();
  const report = useResource<Report>(`/api/reports/${encodeURIComponent(id)}`);
  if (report.loading && !report.data) return <Loading />;
  if (report.error || !report.data) return <ErrorBox code={report.error || 'not_found'} onRetry={report.refresh} />;
  return <ReportView report={report.data} onChange={report.setData} key={report.data.id} title={m.report.eyebrow} />;
}

function ReportView({ report, onChange, title }: { report: Report; onChange: (r: Report) => void; title: string }) {
  const { m, clock, num } = useI18n();
  const audio = useRef<HTMLAudioElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playError, setPlayError] = useState(false);
  const [tab, setTab] = useState<Tab>('sheet');
  const segs = report.segments;
  const duration = report.audio.duration;
  const active = segmentAt(segs, time);
  const current = active >= 0 ? segs[active] : null;
  const next = active >= 0 && active + 1 < segs.length ? segs[active + 1] : null;
  const keyNow = report.keys.find(k => time >= k.start && time < k.end) ?? report.keys[0];

  useEffect(() => {
    const el = audio.current;
    if (!el) return;
    let frame = 0;
    const tick = () => { setTime(el.currentTime); frame = requestAnimationFrame(tick); };
    const play = () => { setPlaying(true); frame = requestAnimationFrame(tick); };
    const stop = () => { setPlaying(false); cancelAnimationFrame(frame); setTime(el.currentTime); };
    const seeked = () => setTime(el.currentTime);
    const fail = () => setPlayError(true);
    el.addEventListener('play', play); el.addEventListener('pause', stop); el.addEventListener('ended', stop);
    el.addEventListener('seeked', seeked); el.addEventListener('error', fail);
    return () => { cancelAnimationFrame(frame); el.removeEventListener('play', play); el.removeEventListener('pause', stop); el.removeEventListener('ended', stop); el.removeEventListener('seeked', seeked); el.removeEventListener('error', fail); };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA', 'BUTTON', 'SELECT'].includes(target.tagName) && target.getAttribute('role') !== 'slider') { e.preventDefault(); toggle(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  function seek(t: number, startPlaying = false) {
    const el = audio.current;
    if (!el) return;
    el.currentTime = t; setTime(t);
    if (startPlaying) void el.play().catch(() => setPlayError(true));
  }
  function toggle() {
    const el = audio.current;
    if (!el) return;
    if (el.paused) void el.play().catch(() => setPlayError(true)); else el.pause();
  }

  const stats = useMemo(() => {
    const total: Record<string, number> = {};
    for (const s of segs) total[family(s.label)] = (total[family(s.label)] ?? 0) + (s.end - s.start);
    return total;
  }, [segs]);

  return <Page eyebrow={title} title={report.audio.name}
    lead={<span className="key-chips">
      <Badge tone="accent">{report.mode === 'single' ? m.modes.single : m.modes.ensemble}</Badge>
      <Badge>{report.rules ? m.modes.rulesOn : m.modes.rulesOff}</Badge>
      <Badge>{fill(m.report.chordsCount, { value: segs.length })}</Badge>
      <Badge>{fill(m.report.revision, { value: report.revision })}</Badge>
    </span>}>
    <div className="stack">
      <Card>
        <div className="player-card">
          <div>
            <div className="now">
              <div><div className="now-label">{m.report.now}</div>{current ? <Chord label={current.label} size="xl" /> : <Chord label="N" size="xl" />}</div>
              {next && <div className="now-next"><div className="now-label">{m.report.next}</div><Chord label={next.label} size="lg" /></div>}
            </div>
            <div className="controls">
              <button type="button" className="play" aria-label={playing ? 'pause' : 'play'} onClick={toggle}>
                {playing
                  ? <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" /><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" /></svg>
                  : <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>}
              </button>
              <input className="seek" type="range" min={0} max={duration} step={0.05} value={time} aria-label={m.report.playhead} onChange={e => seek(Number(e.target.value))} />
              <span className="time">{clock(time)} / {clock(duration)}</span>
            </div>
            <audio ref={audio} src={`/api/runs/${report.run_id}/audio`} preload="auto" />
            {playError && <ErrorBox code="playback" />}
          </div>
          <dl className="meta-list">
            <dt>{m.report.key}</dt><dd>{keyNow ? keyName(keyNow, m.keyMode) : m.common.noData}</dd>
            <dt>{m.report.keys}</dt><dd className="num">{Math.max(0, report.keys.length - 1)}</dd>
            <dt>{m.report.duration}</dt><dd className="num">{clock(duration)}</dd>
            <dt>{m.about.eyebrow}</dt><dd>{report.model.name} × {report.model.models}</dd>
          </dl>
        </div>
      </Card>

      <Card>
        <Timeline segments={segs} keys={report.keys} duration={duration} time={time} active={active} onSeek={t => seek(t)} />
        <div style={{ marginTop: 12 }}><Legend /></div>
        <div className="muted small" style={{ marginTop: 8 }}>
          {Object.entries(stats).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([f, s]) => `${m.families[f as keyof typeof m.families]} ${num((100 * s) / Math.max(duration, 1e-6), 0)} %`).join(' · ')}
        </div>
      </Card>

      <Card>
        <div className="tabs" role="tablist">
          {(['sheet', 'table', 'edit', 'evaluate', 'export'] as const).map(t => <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{m.report.tabs[t]}</button>)}
        </div>
        {tab === 'sheet' && <>
          <p className="muted small" style={{ marginTop: 0 }}>{m.report.sheetHint}</p>
          <div className="sheet">{segs.map((s, i) => <button key={i} type="button" className={i === active ? 'active' : ''} data-family={family(s.label)} onClick={() => seek(s.start, true)}>
            <Chord label={s.label} />
            <small>{clock(s.start)}</small>
            {typeof s.confidence === 'number' && <span className="conf" title={`${m.report.confidence} ${num(s.confidence * 100, 0)} %`}><span style={{ width: `${Math.round(s.confidence * 100)}%` }} /></span>}
          </button>)}</div>
        </>}
        {tab === 'table' && <div className="table-wrap"><table>
          <thead><tr><th>#</th><th>{m.report.start}</th><th>{m.report.end}</th><th>{m.report.shortLabel}</th><th>Harte</th><th className="right">{m.report.confidence}</th><th /></tr></thead>
          <tbody>{segs.map((s, i) => <tr key={i} className={i === active ? 'current' : ''}>
            <td className="muted num">{i + 1}</td><td className="num">{clock(s.start)}</td><td className="num">{clock(s.end)}</td>
            <td><Chord label={s.label} size="sm" /></td><td><code>{s.label}</code></td>
            <td className="right num">{typeof s.confidence === 'number' ? `${num(s.confidence * 100, 0)} %` : '—'}</td>
            <td><button type="button" className="btn ghost small" onClick={() => seek(s.start, true)}>{m.report.listen}</button></td>
          </tr>)}</tbody>
        </table></div>}
        {tab === 'edit' && <Editor report={report} onSaved={onChange} />}
        {tab === 'evaluate' && <Evaluate report={report} onDone={onChange} />}
        {tab === 'export' && <div className="exports">
          <p className="muted small" style={{ margin: 0 }}>{m.report.exportHint} {fill(m.report.revision, { value: report.revision })}</p>
          {([['json', m.report.exportJson], ['lab', m.report.exportLab], ['csv', m.report.exportCsv]] as const).map(([f, text]) =>
            <a key={f} href={`/api/reports/${report.id}/export?format=${f}`} download><span>{text}</span><Badge>.{f}</Badge></a>)}
        </div>}
      </Card>
    </div>
  </Page>;
}

function Editor({ report, onSaved }: { report: Report; onSaved: (r: Report) => void }) {
  const { m } = useI18n();
  const fresh = () => report.segments.map(s => ({ ...s, start: Math.round(s.start * 1000) / 1000, end: Math.round(s.end * 1000) / 1000 }));
  const [rows, setRows] = useState<Segment[]>(fresh);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(0);
  const [saving, setSaving] = useState(false);
  const problem = validateSegments(rows, report.audio.duration);

  function update(i: number, patch: Partial<Segment>) {
    setSaved(0);
    setRows(prev => prev.map((r, j) => {
      if (j === i) return { ...r, ...patch, confidence: null };
      if (j === i + 1 && patch.end !== undefined) return { ...r, start: patch.end };
      if (j === i - 1 && patch.start !== undefined) return { ...r, end: patch.start };
      return r;
    }));
  }
  function addAfter(i: number) {
    setSaved(0);
    setRows(prev => {
      const r = prev[i]; const mid = Math.round(((r.start + r.end) / 2) * 1000) / 1000;
      return [...prev.slice(0, i), { ...r, end: mid }, { start: mid, end: r.end, label: r.label, confidence: null }, ...prev.slice(i + 1)];
    });
  }
  function removeRow(i: number) {
    setSaved(0);
    setRows(prev => {
      if (prev.length < 2) return prev;
      const out = prev.filter((_, j) => j !== i);
      if (i > 0) out[i - 1] = { ...out[i - 1], end: prev[i].end }; else out[0] = { ...out[0], start: prev[0].start };
      return out;
    });
  }
  async function save() {
    setSaving(true); setError('');
    try {
      const r = await request<Report>(`/api/reports/${report.id}/corrections`, { method: 'POST', body: JSON.stringify({ expected_revision: report.revision, segments: rows, note }) });
      setSaved(r.revision); setNote(''); onSaved(r);
    } catch (err) { setError(errorCode(err)); }
    finally { setSaving(false); }
  }

  const problemText = problem ? fill(m.report.invalid[problem.code as keyof typeof m.report.invalid] ?? m.errors.invalid_input, { index: problem.index + 1 }) : '';
  return <div className="editor">
    <p className="muted small" style={{ marginTop: 0 }}>{m.report.editHint}</p>
    <div className="table-wrap" style={{ maxHeight: 460, overflowY: 'auto' }}><table>
      <thead><tr><th>#</th><th>{m.report.start}</th><th>{m.report.end}</th><th>{m.report.label}</th><th /><th /></tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>
        <td className="muted num">{i + 1}</td>
        <td className="num-cell"><input type="number" step={0.01} min={0} value={r.start} aria-label={`${m.report.start} ${i + 1}`} disabled={i === 0}
          className={problem?.index === i && problem.code !== 'label' ? 'bad' : ''} onChange={e => update(i, { start: Number(e.target.value) })} /></td>
        <td className="num-cell"><input type="number" step={0.01} min={0} value={r.end} aria-label={`${m.report.end} ${i + 1}`}
          className={problem?.index === i && problem.code !== 'label' ? 'bad' : ''} onChange={e => update(i, { end: Number(e.target.value) })} /></td>
        <td><input type="text" value={r.label} spellCheck={false} aria-label={`${m.report.label} ${i + 1}`}
          className={problem?.index === i && problem.code === 'label' ? 'bad' : ''} onChange={e => update(i, { label: e.target.value.trim() })} /></td>
        <td><Chord label={/^(N|X|[A-G])/.test(r.label) ? r.label : 'X'} size="sm" /></td>
        <td><div className="row-actions">
          <button type="button" className="btn ghost small" title={m.report.addAfter} aria-label={m.report.addAfter} onClick={() => addAfter(i)}>+</button>
          <button type="button" className="btn ghost small" title={m.report.remove} aria-label={m.report.remove} disabled={rows.length < 2} onClick={() => removeRow(i)}>−</button>
        </div></td>
      </tr>)}</tbody>
    </table></div>
    {problem && <div className="alert" role="alert">{problemText}</div>}
    {error && <ErrorBox code={error} />}
    {saved > 0 && <p className="notice">{fill(m.report.saved, { value: saved })}</p>}
    <div className="editor-foot">
      <label><span className="field-label" style={{ marginTop: 0 }}>{m.report.note}</span><input type="text" maxLength={500} value={note} onChange={e => setNote(e.target.value)} /></label>
      <button type="button" className="btn ghost" onClick={() => { setRows(fresh()); setSaved(0); }}>{m.report.reset}</button>
      <button type="button" className="btn" disabled={!!problem || saving} onClick={() => void save()}>{m.report.saveRevision}</button>
    </div>
  </div>;
}

function Evaluate({ report, onDone }: { report: Report; onDone: (r: Report) => void }) {
  const { m, num } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const ev = report.evaluation;

  async function send(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError('');
    try { onDone(await upload<Report>(`/api/reports/${report.id}/evaluate`, file)); }
    catch (err) { setError(errorCode(err)); }
    finally { setBusy(false); }
  }

  return <div>
    <p className="muted small" style={{ marginTop: 0 }}>{m.report.evaluateHint}</p>
    <button type="button" className="btn" disabled={busy} onClick={() => input.current?.click()}>{busy ? m.common.loading : m.report.chooseLab}</button>
    <input ref={input} type="file" hidden accept=".lab,.txt" onChange={e => { void send(e.target.files?.[0]); e.target.value = ''; }} />
    {error && <ErrorBox code={error} />}
    {ev && <>
      <p className="muted small">{fill(m.report.referenceSegments, { value: ev.reference_segments })} · {fill(m.report.evaluatedRevision, { value: ev.revision })}</p>
      <div className="scores">{LEVELS.map(l => {
        const v = ev.levels[l];
        return <div className="score" key={l}>
          <b>{typeof v === 'number' ? num(v, 1) : '—'}</b>
          <span>{m.report.levels[l]}</span>
          <div className="bar"><span style={{ width: `${typeof v === 'number' ? Math.round(v) : 0}%` }} /></div>
        </div>;
      })}</div>
    </>}
  </div>;
}
