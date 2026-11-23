export class ApiError extends Error {
  constructor(public code: string, public status = 0) { super(code); }
}

let token = '';
export function setToken(value: string): void { token = value; }

function unwrap<T>(body: unknown, status: number): T {
  if (body && typeof body === 'object' && 'success' in body) {
    if (body.success === true && 'data' in body) return body.data as T;
    const error = 'error' in body ? body.error : null;
    const code = error && typeof error === 'object' && 'code' in error && typeof error.code === 'string' ? error.code : 'request';
    throw new ApiError(code, status);
  }
  throw new ApiError('response', status);
}

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      credentials: 'same-origin',
      headers: {
        ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
        ...(options.method && options.method !== 'GET' ? { 'X-Chord-Token': token } : {}),
        ...options.headers,
      },
    });
  } catch {
    throw new ApiError('connection');
  }
  let body: unknown = null;
  try { body = await response.json(); } catch { throw new ApiError('response', response.status); }
  return unwrap<T>(body, response.status);
}

export function upload<T>(path: string, file: File, onProgress?: (share: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', path);
    xhr.setRequestHeader('X-Chord-Token', token);
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress?.(event.loaded / event.total); };
    xhr.onerror = () => reject(new ApiError('connection'));
    xhr.onload = () => {
      try { resolve(unwrap<T>(JSON.parse(xhr.responseText) as unknown, xhr.status)); }
      catch (error) { reject(error instanceof ApiError ? error : new ApiError('response', xhr.status)); }
    };
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export function errorCode(error: unknown): string { return error instanceof ApiError ? error.code : 'request'; }
