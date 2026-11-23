export type Mode = 'single' | 'ensemble';

export interface AudioAsset {
  id: string;
  name: string;
  duration: number;
  sample_rate: number;
  channels: number;
  created_at?: string;
}

export interface Bootstrap {
  app: string;
  token: string;
  model: { name: string; parameters: number; device: string; modes: Mode[] };
  modes: Mode[];
  limits: { max_upload_bytes: number; max_duration_seconds: number; formats: string[] };
}

export interface Segment {
  start: number;
  end: number;
  label: string;
  confidence?: number | null;
}

export interface KeySegment {
  start: number;
  end: number;
  tonic: string;
  mode: 'major' | 'minor';
}

export interface ProgressEvent {
  id: number;
  time: string;
  stage: 'audio' | 'features' | 'model' | 'decode' | 'done';
  progress: number;
  detail: string;
}

export interface Run {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  mode: Mode;
  rules: boolean;
  audio: AudioAsset;
  created_at: string;
  progress: number;
  stage: string;
  report_id: string | null;
  error: string | null;
  chords?: number;
}

export interface Evaluation {
  levels: Record<string, number | null>;
  reference_segments: number;
  revision: number;
}

export interface Report {
  id: string;
  run_id: string;
  revision: number;
  created_at: string;
  updated_at: string;
  audio: AudioAsset;
  mode: Mode;
  rules: boolean;
  model: { name: string; models: number; device: string };
  segments: Segment[];
  majmin: Segment[];
  keys: KeySegment[];
  corrections: { revision: number; time: string; note: string }[];
  evaluation: Evaluation | null;
}

export interface ReportSummary {
  id: string;
  run_id: string;
  revision: number;
  created_at: string;
  audio: AudioAsset;
  mode: Mode;
  rules: boolean;
  keys: KeySegment[];
  chords: number;
}
