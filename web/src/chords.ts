import type { Segment } from './types';

/** Семейство аккорда по метке Harte — для цвета и легенды. */
export type Family = 'major' | 'minor' | 'dominant' | 'major7' | 'minor7' | 'diminished' | 'suspended' | 'augmented' | 'none';

export const FAMILIES: Family[] = ['major', 'minor', 'dominant', 'major7', 'minor7', 'diminished', 'suspended', 'augmented', 'none'];

export function quality(label: string): string {
  if (label === 'N' || label === 'X') return label;
  const body = label.includes(':') ? label.split(':')[1] : 'maj';
  return body.split('/')[0].split('(')[0] || 'maj';
}

export function family(label: string): Family {
  const q = quality(label);
  if (q === 'N' || q === 'X') return 'none';
  if (q === 'maj' || q === 'maj6') return 'major';
  if (q === 'min' || q === 'min6') return 'minor';
  if (q === '7' || q === '9' || q === '11' || q === '13') return 'dominant';
  if (q.startsWith('maj')) return 'major7';
  if (q.startsWith('min')) return 'minor7';
  if (q === 'dim' || q === 'dim7' || q === 'hdim7') return 'diminished';
  if (q.startsWith('sus')) return 'suspended';
  if (q === 'aug') return 'augmented';
  return 'major';
}

const SHORT: Record<string, string> = {
  maj: '', min: 'm', '7': '7', maj7: 'maj7', min7: 'm7', dim: '°', dim7: '°7', hdim7: 'ø7', aug: '+', sus2: 'sus2', sus4: 'sus4',
  maj6: '6', min6: 'm6', '9': '9', minmaj7: 'm(maj7)', 'sus4(b7)': '7sus4',
};

const DEGREE: Record<string, number> = { '1': 0, b2: 1, '2': 2, b3: 3, '3': 4, '4': 5, b5: 6, '5': 7, '#5': 8, b6: 8, '6': 9, b7: 10, '7': 11 };
const PITCH = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const INDEX: Record<string, number> = { C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11 };

/** Короткая запись для музыканта: C:min7/b7 → Cm7/A#; N → «—». */
export function pretty(label: string): string {
  if (label === 'N') return '—';
  if (label === 'X') return '?';
  const [root, rest = 'maj'] = label.split(':');
  const [qualPart, bass] = rest.split('/');
  const q = qualPart in SHORT ? SHORT[qualPart] : qualPart;
  let text = root + q;
  if (bass && bass !== '1' && root in INDEX && bass in DEGREE) text += '/' + PITCH[(INDEX[root] + DEGREE[bass]) % 12];
  return text;
}

export function segmentAt(segments: Segment[], time: number): number {
  let lo = 0;
  let hi = segments.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (time < segments[mid].start) hi = mid - 1;
    else if (time >= segments[mid].end) lo = mid + 1;
    else return mid;
  }
  return -1;
}

/** Проверка правки: отрезки идут по порядку, без пустот и перекрытий, метки не пустые. */
export function validateSegments(segments: Segment[], duration: number): { index: number; code: string } | null {
  if (!segments.length) return { index: 0, code: 'empty' };
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    if (!Number.isFinite(s.start) || !Number.isFinite(s.end) || s.end <= s.start) return { index: i, code: 'interval' };
    if (s.end > duration + 0.5) return { index: i, code: 'beyond' };
    if (i > 0 && Math.abs(s.start - segments[i - 1].end) > 0.002) return { index: i, code: 'gap' };
    if (!/^(N|X|[A-G][#b]?(:[^\s/]+)?(\/[#b]?\d+)?)$/.test(s.label)) return { index: i, code: 'label' };
  }
  return null;
}
