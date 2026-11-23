import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { MESSAGES } from './messages';
import type { Language, Messages } from './messages';

const STORAGE = 'chordrec.language';

function readLanguage(): Language {
  try {
    const saved = localStorage.getItem(STORAGE);
    if (saved === 'ru' || saved === 'en') return saved;
  } catch { /* storage may be blocked */ }
  return navigator.language?.startsWith('ru') ? 'ru' : 'en';
}

export function fill(text: string, values: Record<string, string | number> = {}): string {
  return text.replace(/\{(\w+)\}/g, (match, name: string) => (name in values ? String(values[name]) : match));
}

function make(language: Language, setLanguage: (l: Language) => void) {
  const m: Messages = MESSAGES[language];
  const locale = language === 'ru' ? 'ru-RU' : 'en-GB';
  return {
    language, setLanguage, m,
    num: (value: number | null | undefined, digits = 2) =>
      typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : m.common.noData,
    clock: (value: number) => {
      const s = Math.max(0, value);
      return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
    },
    date: (value: string) => {
      const d = new Date(value);
      return Number.isFinite(d.getTime()) ? d.toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' }) : value;
    },
    error: (code: string, values: Record<string, string> = {}) =>
      fill((m.errors as Record<string, string>)[code] ?? m.errors.request, values),
  };
}

type Locale = ReturnType<typeof make>;
const Context = createContext<Locale>(make('ru', () => {}));

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(readLanguage);
  useEffect(() => {
    document.documentElement.lang = language;
    try { localStorage.setItem(STORAGE, language); } catch { /* ignore */ }
  }, [language]);
  const value = useMemo(() => make(language, setLanguage), [language]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useI18n(): Locale { return useContext(Context); }

export function LanguageSwitch() {
  const { language, setLanguage, m } = useI18n();
  return <div className="switch" role="group" aria-label={m.nav.language}>
    {(['ru', 'en'] as const).map(l => <button type="button" key={l} aria-pressed={language === l} onClick={() => setLanguage(l)}>{l.toUpperCase()}</button>)}
  </div>;
}
