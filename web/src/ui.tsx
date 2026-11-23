import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useI18n } from './i18n';
import { FAMILIES, family, pretty } from './chords';
import type { KeySegment } from './types';

export function Page({ eyebrow, title, lead, actions, children }: { eyebrow?: string; title: string; lead?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  useEffect(() => { document.title = `${title} · Chord Recognition`; }, [title]);
  return <section className="page">
    <header className="page-head">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {lead && <p className="lead">{lead}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
    {children}
  </section>;
}

export function Card({ title, children, className = '', aside }: { title?: ReactNode; children: ReactNode; className?: string; aside?: ReactNode }) {
  return <section className={`card ${className}`}>
    {(title || aside) && <header className="card-head">{title && <h2>{title}</h2>}{aside}</header>}
    {children}
  </section>;
}

export function Loading() {
  const { m } = useI18n();
  return <div className="state" role="status"><span className="spinner" aria-hidden="true" />{m.common.loading}</div>;
}

export function ErrorBox({ code, onRetry, values }: { code: string; onRetry?: () => void; values?: Record<string, string> }) {
  const { m, error } = useI18n();
  return <div className="alert" role="alert">
    <span>{error(code, values)}</span>
    {onRetry && <button type="button" className="btn ghost small" onClick={onRetry}>{m.common.retry}</button>}
  </div>;
}

export function Chord({ label, size = 'md' }: { label: string; size?: 'sm' | 'md' | 'lg' | 'xl' }) {
  return <span className={`chord chord-${size}`} data-family={family(label)} title={label}>{pretty(label)}</span>;
}

export function Legend() {
  const { m } = useI18n();
  return <ul className="legend" aria-label={m.report.legend}>
    {FAMILIES.map(f => <li key={f}><span className="swatch" data-family={f} />{m.families[f]}</li>)}
  </ul>;
}

export function keyName(k: KeySegment, mode: Record<string, string>): string {
  return `${k.tonic} ${mode[k.mode] ?? k.mode}`;
}

export function Badge({ tone = 'neutral', children }: { tone?: 'neutral' | 'ok' | 'warn' | 'bad' | 'accent'; children: ReactNode }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return <div className="progress" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
    <div className="progress-fill" style={{ width: `${pct}%` }} />
  </div>;
}

const THEME = 'chordrec.theme';
type Theme = 'light' | 'dark' | 'system';

export function ThemeSwitch() {
  const { m } = useI18n();
  const [theme, setTheme] = useState<Theme>(() => {
    try { const t = localStorage.getItem(THEME); if (t === 'light' || t === 'dark') return t; } catch { /* ignore */ }
    return 'system';
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', theme);
    try { if (theme === 'system') localStorage.removeItem(THEME); else localStorage.setItem(THEME, theme); } catch { /* ignore */ }
  }, [theme]);
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  return <button type="button" className="icon-btn" aria-label={m.nav.theme} title={m.nav.theme} onClick={() => setTheme(dark ? 'light' : 'dark')}>
    {dark
      ? <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="currentColor" /><g stroke="currentColor" strokeWidth="2" strokeLinecap="round">{[0, 45, 90, 135, 180, 225, 270, 315].map(a => <line key={a} x1="12" y1="2.5" x2="12" y2="4.5" transform={`rotate(${a} 12 12)`} />)}</g></svg>
      : <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="currentColor" /></svg>}
  </button>;
}
