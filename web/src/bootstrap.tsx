import { createContext, useContext } from 'react';
import type { Bootstrap } from './types';

export const BootstrapContext = createContext<Bootstrap | null>(null);

export function useBootstrap(): Bootstrap {
  const value = useContext(BootstrapContext);
  if (!value) throw new Error('bootstrap missing');
  return value;
}
