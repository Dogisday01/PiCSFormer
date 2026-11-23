import { useEffect, useState } from 'react';
import type { AnchorHTMLAttributes } from 'react';
export function navigate(path: string): void { window.history.pushState({}, '', path); window.dispatchEvent(new PopStateEvent('popstate')); }
export function usePath(): string { const [path, setPath] = useState(window.location.pathname); useEffect(() => { const update = () => setPath(window.location.pathname); window.addEventListener('popstate', update); return () => window.removeEventListener('popstate', update); }, []); return path; }
export function Link({ href = '/analysis', onClick, ...props }: AnchorHTMLAttributes<HTMLAnchorElement>) {
    return <a {...props} href={href} onClick={event => { onClick?.(event); if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || props.target || href.startsWith('#'))
        return; if (!href.startsWith('/') || href.startsWith('/api'))
        return; event.preventDefault(); navigate(href); }}/>;
}
