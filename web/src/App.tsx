import { useEffect, useState } from 'react';
import { errorCode, request, setToken } from './api';
import { BootstrapContext } from './bootstrap';
import { LanguageSwitch, useI18n } from './i18n';
import { Link, usePath } from './navigation';
import { AboutPage } from './pages/AboutPage';
import { HistoryPage } from './pages/HistoryPage';
import { RecognizePage } from './pages/RecognizePage';
import { ReportPage } from './pages/ReportPage';
import type { Bootstrap } from './types';
import { ErrorBox, Loading, ThemeSwitch } from './ui';

function Route({ path }: { path: string }) {
  const { m } = useI18n();
  if (path === '/' || path === '') return <RecognizePage />;
  if (path === '/history') return <HistoryPage />;
  if (path === '/about') return <AboutPage />;
  const report = /^\/reports\/([\w-]+)$/.exec(path);
  if (report) return <ReportPage id={report[1]} />;
  return <div className="empty"><h1>{m.nav.missing}</h1><Link className="btn" href="/">{m.nav.home}</Link></div>;
}

export function App() {
  const { m } = useI18n();
  const path = usePath();
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    setError('');
    request<Bootstrap>('/api/bootstrap')
      .then(b => { if (active) { setToken(b.token); setBoot(b); } })
      .catch(err => { if (active) setError(errorCode(err)); });
    return () => { active = false; };
  }, [attempt]);

  useEffect(() => { window.scrollTo(0, 0); }, [path]);

  const links = [['/', m.nav.recognize], ['/history', m.nav.history], ['/about', m.nav.about]] as const;
  const current = (href: string) => (href === '/' ? path === '/' : path.startsWith(href)) || (href === '/history' && path.startsWith('/reports'));

  return <>
    <a className="skip" href="#main">{m.nav.skip}</a>
    <header className="topbar">
      <div className="topbar-inner">
        <Link className="brand" href="/">
          <span className="brand-mark" aria-hidden="true">♯</span>
          <span><span className="brand-name">{m.brand.name}</span><span className="brand-tag">{m.brand.tagline}</span></span>
        </Link>
        <nav className="nav" aria-label="main">
          {links.map(([href, text]) => <Link key={href} href={href} aria-current={current(href) ? 'page' : undefined}>{text}</Link>)}
        </nav>
        <div className="tools"><LanguageSwitch /><ThemeSwitch /></div>
      </div>
    </header>
    <main id="main">
      {error && <ErrorBox code={error} onRetry={() => setAttempt(a => a + 1)} />}
      {!error && !boot && <Loading />}
      {boot && <BootstrapContext.Provider value={boot}><Route path={path} /></BootstrapContext.Provider>}
    </main>
    <footer className="footer"><span>{m.footer.text}</span><span>{m.footer.code}</span></footer>
  </>;
}
