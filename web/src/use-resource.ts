import { useCallback, useEffect, useState } from 'react';
import { errorCode, request } from './api';

export function useResource<T>(path: string | null) {
  const [loaded, setLoaded] = useState<{ path: string; value: T } | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);
  const refresh = useCallback(() => setVersion(v => v + 1), []);
  useEffect(() => {
    if (!path) return;
    let active = true;
    setLoading(true);
    setError('');
    request<T>(path)
      .then(value => { if (active) setLoaded({ path, value }); })
      .catch(err => { if (active) setError(errorCode(err)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [path, version]);
  return { data: loaded?.path === path ? loaded.value : null, error, loading, refresh, setData: (value: T) => path && setLoaded({ path, value }) };
}
