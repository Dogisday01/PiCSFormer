import { useEffect, useRef, useState } from 'react';
import type { DragEvent } from 'react';
import { errorCode, request, upload } from '../api';
import { useBootstrap } from '../bootstrap';
import { fill, useI18n } from '../i18n';
import { navigate } from '../navigation';
import type { AudioAsset, Mode, ProgressEvent, Run } from '../types';
import { Card, Chord, ErrorBox, Page, ProgressBar } from '../ui';

const STAGES = ['audio', 'features', 'model', 'decode', 'done'] as const;
const DEMO = ['C:maj', 'A:min', 'D:min7', 'G:7', 'C:maj7', 'F:maj', 'B:dim7', 'E:7/3', 'A:min', 'G:sus4'];

export function RecognizePage() {
  const { m, num, clock, error: describe } = useI18n();
  const boot = useBootstrap();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [asset, setAsset] = useState<AudioAsset | null>(null);
  const [uploading, setUploading] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>('single');
  const [rules, setRules] = useState(true);
  const [run, setRun] = useState<Run | null>(null);
  const [event, setEvent] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState('');
  const [failure, setFailure] = useState('');
  const [starting, setStarting] = useState(false);

  async function send(file: File | undefined) {
    if (!file) return;
    setError(''); setFailure(''); setRun(null); setEvent(null); setAsset(null);
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!boot.limits.formats.includes(ext)) { setError('invalid_format'); return; }
    if (file.size > boot.limits.max_upload_bytes) { setError('invalid_size'); return; }
    setUploading(0);
    try { setAsset(await upload<AudioAsset>('/api/audio', file, setUploading)); }
    catch (err) { setError(errorCode(err)); }
    finally { setUploading(null); }
  }

  async function start() {
    if (!asset) return;
    setStarting(true); setError(''); setFailure('');
    try { setRun(await request<Run>('/api/runs', { method: 'POST', body: JSON.stringify({ audio_id: asset.id, mode, rules }) })); }
    catch (err) { setError(errorCode(err)); }
    finally { setStarting(false); }
  }

  const runId = run?.id;
  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/runs/${runId}/events`);
    source.addEventListener('progress', e => setEvent(JSON.parse((e as MessageEvent).data) as ProgressEvent));
    source.addEventListener('done', e => {
      source.close();
      const done = JSON.parse((e as MessageEvent).data) as { status: string; report_id: string | null };
      if (done.status === 'completed' && done.report_id) setTimeout(() => navigate(`/reports/${done.report_id}`), 500);
      else request<Run>(`/api/runs/${runId}`).then(r => setFailure(r.error ?? 'error')).catch(() => setFailure('error'));
    });
    source.onerror = () => { /* EventSource reconnects by itself */ };
    return () => source.close();
  }, [runId]);

  function onDrop(e: DragEvent) { e.preventDefault(); setOver(false); void send(e.dataTransfer.files[0]); }

  const busy = uploading !== null || starting || (!!run && !failure);
  const stageIndex = event ? STAGES.indexOf(event.stage) : -1;

  return <Page eyebrow={m.recognize.eyebrow} title={m.recognize.title} lead={m.recognize.lead}>
    <div className="hero">
      <Card>
        {!asset && uploading === null && <div className={`drop${over ? ' over' : ''}`} role="button" tabIndex={0}
          onClick={() => input.current?.click()} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current?.click(); } }}
          onDragOver={e => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
          <span className="drop-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="26" height="26"><path d="M9 18V5l11-2v13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" /><circle cx="6" cy="18" r="3" fill="currentColor" /><circle cx="17" cy="16" r="3" fill="currentColor" /></svg>
          </span>
          <strong>{m.recognize.drop}</strong>
          <span className="muted">{m.recognize.or}</span>
          <span className="btn small">{m.recognize.choose}</span>
          <span className="muted small">{m.recognize.formats}</span>
        </div>}
        <input ref={input} type="file" hidden accept={boot.limits.formats.map(f => '.' + f).join(',')} onChange={e => { void send(e.target.files?.[0]); e.target.value = ''; }} />

        {uploading !== null && <div className="stack">
          <p className="muted">{fill(m.recognize.uploading, { value: Math.round(uploading * 100) })}</p>
          <ProgressBar value={uploading} label={m.recognize.uploading} />
        </div>}

        {asset && <>
          <div className="file-row">
            <span className="drop-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="22" height="22"><path d="M3 12h2M7 7v10M11 4v16M15 8v8M19 10v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg></span>
            <div className="grow">
              <div className="file-name" title={asset.name}>{asset.name}</div>
              <div className="muted small num">{clock(asset.duration)} · {fill(m.recognize.sampleRate, { value: asset.sample_rate })} · {asset.channels === 1 ? m.recognize.channels.one : fill(m.recognize.channels.many, { value: asset.channels })}</div>
            </div>
            <button type="button" className="btn ghost small" disabled={busy} onClick={() => { setAsset(null); setRun(null); setEvent(null); input.current?.click(); }}>{m.recognize.replace}</button>
          </div>

          <span className="field-label">{m.recognize.mode}</span>
          <div className="choices" role="radiogroup" aria-label={m.recognize.mode}>
            {(['single', 'ensemble'] as const).filter(x => boot.modes.includes(x)).map(x => <label className="choice" key={x}>
              <input type="radio" name="mode" value={x} checked={mode === x} disabled={busy} onChange={() => setMode(x)} />
              <strong>{x === 'single' ? m.recognize.single : m.recognize.ensemble}</strong>
              <span>{x === 'single' ? m.recognize.singleHint : m.recognize.ensembleHint}</span>
            </label>)}
          </div>
          <label className="toggle">
            <input type="checkbox" checked={rules} disabled={busy} onChange={e => setRules(e.target.checked)} />
            <div><strong>{m.recognize.rules}</strong><span>{m.recognize.rulesHint}</span></div>
          </label>

          {!run && <div style={{ marginTop: 18 }}>
            <button type="button" className="btn big" disabled={busy} onClick={() => void start()}>{starting ? m.recognize.starting : m.recognize.start}</button>
          </div>}

          {run && !failure && <div style={{ marginTop: 18 }} aria-live="polite">
            <div className="card-head"><h2>{m.recognize.progress}</h2><span className="num muted">{num((event?.progress ?? 0) * 100, 0)} %</span></div>
            <ProgressBar value={event?.progress ?? 0.02} label={m.recognize.progress} />
            <ol className="stages">
              {STAGES.map((s, i) => <li key={s} data-state={i < stageIndex || event?.stage === 'done' ? 'done' : i === stageIndex ? 'active' : 'todo'}>{m.recognize.stage[s]}</li>)}
            </ol>
            {event?.stage === 'done' && <p className="notice">{m.recognize.done}</p>}
          </div>}
        </>}

        {error && <ErrorBox code={error} />}
        {failure && <div className="alert" role="alert"><span>{describe('failed', { error: failure })}</span>
          <button type="button" className="btn ghost small" onClick={() => { setRun(null); setFailure(''); setEvent(null); }}>{m.common.retry}</button></div>}
      </Card>

      <div className="stack">
        <Card title={m.recognize.how}>
          <div className="preview-strip" aria-hidden="true">{DEMO.map((l, i) => <Chord key={i} label={l} size="sm" />)}</div>
          <ul className="howlist">{m.recognize.howItems.map(t => <li key={t}>{t}</li>)}</ul>
          <p className="muted small" style={{ marginBottom: 0 }}>{fill(m.about.device, { value: boot.model.device })}</p>
        </Card>
      </div>
    </div>
  </Page>;
}
