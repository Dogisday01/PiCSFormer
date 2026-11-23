import { useBootstrap } from '../bootstrap';
import { fill, useI18n } from '../i18n';
import { Card, Legend, Page } from '../ui';

// Итог на отложенных наборах (открыты один раз 23.11.2026), среднее 15 моделей с пятью правилами.
const HELD_OUT = {
  bpsd: [70.71, 75.74, 69.10, 61.70, 56.83, 73.31],
  jazz: [68.67, 62.86, 58.87, 41.31, 38.57, 70.09],
} as const;
const LEVELS = ['root', 'majmin', 'triads', 'sevenths', 'tetrads', 'mirex'] as const;

export function AboutPage() {
  const { m, num, language } = useI18n();
  const boot = useBootstrap();
  const millions = (boot.model.parameters / 1e6).toLocaleString(language === 'ru' ? 'ru-RU' : 'en-GB', { maximumFractionDigits: 2 }) + (language === 'ru' ? ' млн' : 'M');
  return <Page eyebrow={m.about.eyebrow} title={m.about.title} lead={m.about.lead}>
    <div className="stat-row">
      <div className="stat"><b>{millions}</b><span>{language === 'ru' ? 'параметров' : 'parameters'}</span></div>
      <div className="stat"><b>15</b><span>{language === 'ru' ? 'моделей: 5 фолдов × 3 seeds' : 'models: 5 folds × 3 seeds'}</span></div>
      <div className="stat"><b>{num(HELD_OUT.bpsd[3], 1)}</b><span>{language === 'ru' ? 'септаккорды, сонаты' : 'sevenths, sonatas'}</span></div>
      <div className="stat"><b>{boot.model.device}</b><span>{language === 'ru' ? 'устройство' : 'device'}</span></div>
    </div>
    <div className="grid two">
      <div className="stack">
        <Card title={m.about.results}>
          <div className="table-wrap"><table className="results-table">
            <thead><tr><th>{m.about.set}</th>{LEVELS.map(l => <th key={l} className="num">{m.report.levels[l]}</th>)}</tr></thead>
            <tbody>
              {(['bpsd', 'jazz'] as const).map(s => <tr key={s}><td>{m.about[s]}</td>{HELD_OUT[s].map((v, i) => <td key={i} className="num">{num(v, 2)}</td>)}</tr>)}
            </tbody>
          </table></div>
          <p className="muted small">{m.about.resultsNote}</p>
        </Card>
        <Card title={m.about.rulesTitle}><ul className="rules">{m.about.rules.map(r => <li key={r}>{r}</li>)}</ul></Card>
      </div>
      <div className="stack">
        <Card title={m.report.legend}><Legend /></Card>
        <Card title={m.about.paper}><p style={{ margin: 0 }}>{m.about.paperText}</p></Card>
        <Card title={m.about.licence}><p style={{ margin: 0 }}>{m.about.licenceText}</p>
          <p className="muted small" style={{ marginBottom: 0 }}>{fill(m.about.device, { value: boot.model.device })}</p></Card>
      </div>
    </div>
  </Page>;
}
