import { describe, expect, it } from 'vitest';
import { family, pretty, segmentAt, validateSegments } from '../src/chords';
import { fill } from '../src/i18n';
import { MESSAGES } from '../src/i18n/messages';

describe('chords', () => {
  it('names chords for musicians', () => {
    expect(pretty('C:maj')).toBe('C');
    expect(pretty('A:min7')).toBe('Am7');
    expect(pretty('G:7/3')).toBe('G7/B');
    expect(pretty('D:min7/b7')).toBe('Dm7/C');
    expect(pretty('B:dim7')).toBe('B°7');
    expect(pretty('N')).toBe('—');
  });
  it('groups chords into colour families', () => {
    expect(family('C:maj')).toBe('major');
    expect(family('C:maj/5')).toBe('major');
    expect(family('G:7')).toBe('dominant');
    expect(family('A:min7')).toBe('minor7');
    expect(family('F:maj7')).toBe('major7');
    expect(family('B:hdim7')).toBe('diminished');
    expect(family('D:sus4')).toBe('suspended');
    expect(family('N')).toBe('none');
  });
  it('finds the segment under the playhead', () => {
    const segs = [{ start: 0, end: 1, label: 'C' }, { start: 1, end: 2.5, label: 'G' }, { start: 2.5, end: 4, label: 'N' }];
    expect(segmentAt(segs, 0)).toBe(0);
    expect(segmentAt(segs, 1)).toBe(1);
    expect(segmentAt(segs, 3.9)).toBe(2);
    expect(segmentAt(segs, 4)).toBe(-1);
  });
  it('validates edits', () => {
    expect(validateSegments([], 5)?.code).toBe('empty');
    expect(validateSegments([{ start: 0, end: 1, label: 'C:maj' }, { start: 1.2, end: 2, label: 'G' }], 5)).toEqual({ index: 1, code: 'gap' });
    expect(validateSegments([{ start: 0, end: 1, label: 'H:maj' }], 5)).toEqual({ index: 0, code: 'label' });
    expect(validateSegments([{ start: 0, end: 9, label: 'C' }], 5)?.code).toBe('beyond');
    expect(validateSegments([{ start: 0, end: 2, label: 'C:min7/b7' }, { start: 2, end: 5, label: 'N' }], 5)).toBeNull();
  });
});

describe('messages', () => {
  it('has the same keys in both languages', () => {
    const keys = (o: object, p = ''): string[] => Object.entries(o).flatMap(([k, v]) => (v && typeof v === 'object' && !Array.isArray(v) ? keys(v, `${p}${k}.`) : [`${p}${k}`]));
    expect(keys(MESSAGES.en).sort()).toEqual(keys(MESSAGES.ru).sort());
  });
  it('fills placeholders', () => {
    expect(fill('Row {index}: {x}', { index: 3 })).toBe('Row 3: {x}');
  });
});
