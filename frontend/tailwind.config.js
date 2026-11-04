/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f14", panel: "#11161d", panel2: "#161c25", line: "#1f2733", ink: "#e6e8eb", muted: "#a3acb9",
        dim: "#7a8390", accent: "#9085e9", sii: "#9085e9", vix: "#d95926", nfci: "#199e70",
        good: "#0ca30c", warn: "#fab219", crit: "#d03b3b", pos: "#e66767", neg: "#3987e5",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
