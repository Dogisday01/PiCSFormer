import { useState } from 'react';
import { errorCode, request } from '../api';
import { useI18n } from '../i18n';
import { Link } from '../navigation';
import type { Run } from '../types';
import { useResource } from '../use-resource';
import { Badge, Card, ErrorBox, Loading, Page } from '../ui';

const TONE = { completed: 'ok', failed: 'bad', running: 'accent', queued: 'warn' } as const;

export function HistoryPage() {
  const { m, date, clock } = useI18n();
  const runs = useResource<Run[]>('/api/runs');
  const [error, setError] = useState('');

  async function remove(run: Run) {
    if (!window.confirm(m.history.confirmDelete)) return;
    setError('');
    try { await request(`/api/runs/${run.id}`, { method: 'DELETE' }); runs.refresh(); }
    catch (err) { setError(errorCode(err)); }
  }

  return <Page eyebrow={m.history.eyebrow} title={m.history.title}>
    <Card>
      {error && <ErrorBox code={error} />}
      {runs.loading && !runs.data && <Loading />}
      {runs.error && <ErrorBox code={runs.error} onRetry={runs.refresh} />}
      {runs.data && runs.data.length === 0 && <div className="empty"><p>{m.history.empty}</p><Link className="btn" href="/">{m.history.emptyLink}</Link></div>}
      {runs.data && runs.data.length > 0 && <div className="table-wrap"><table>
        <thead><tr><th>{m.history.file}</th><th>{m.history.date}</th><th>{m.history.mode}</th><th>{m.history.status}</th><th className="right" /></tr></thead>
        <tbody>{runs.data.map(r => <tr key={r.id}>
          <td><div className="file-name" style={{ maxWidth: 320 }} title={r.audio.name}>{r.audio.name}</div><span className="muted small num">{clock(r.audio.duration)}</span></td>
          <td className="muted small">{date(r.created_at)}</td>
          <td><Badge>{r.mode === 'single' ? m.modes.single : m.modes.ensemble}</Badge> <Badge>{r.rules ? m.modes.rulesOn : m.modes.rulesOff}</Badge></td>
          <td><Badge tone={TONE[r.status]}>{m.status[r.status]}</Badge></td>
          <td><div className="row-actions">
            {r.report_id && <Link className="btn small" href={`/reports/${r.report_id}`}>{m.common.open}</Link>}
            <button type="button" className="btn danger small" disabled={r.status === 'running' || r.status === 'queued'} onClick={() => void remove(r)}>{m.common.delete}</button>
          </div></td>
        </tr>)}</tbody>
      </table></div>}
    </Card>
  </Page>;
}
