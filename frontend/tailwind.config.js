/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#07090E',
          900: '#0B0F17',
          850: '#0F1523',
          800: '#131B2E',
          750: '#182238',
          700: '#1E293B',
          600: '#334155',
          500: '#475569',
        },
        brand: {
          cyan: '#00E5FF',
          blue: '#0EA5E9',
          indigo: '#6366F1',
          purple: '#8B5CF6',
        },
        risk: {
          allow: '#10B981',
          challenge: '#F59E0B',
          block: '#EF4444',
          subtle: '#64748B',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 20px -5px rgba(0, 229, 255, 0.3)',
        'glow-red': '0 0 20px -5px rgba(239, 68, 68, 0.4)',
        'glow-amber': '0 0 20px -5px rgba(245, 158, 11, 0.35)',
        'glow-green': '0 0 20px -5px rgba(16, 185, 129, 0.35)',
        'glow-purple': '0 0 20px -5px rgba(139, 92, 246, 0.35)',
      }
    },
  },
  plugins: [],
}
