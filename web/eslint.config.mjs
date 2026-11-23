import parser from '@typescript-eslint/parser';
import hooks from 'eslint-plugin-react-hooks';

export default [{
  files: ['**/*.{ts,tsx,js,jsx,mjs}'],
  ignores: ['dist/**', 'node_modules/**', 'test-results/**', 'logs/**'],
  languageOptions: { parser, parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } } },
  plugins: { 'react-hooks': hooks },
  rules: { 'react-hooks/rules-of-hooks': 'error', 'react-hooks/exhaustive-deps': 'error' },
}];
