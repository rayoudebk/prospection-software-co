import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-dm-sans)", "system-ui", "sans-serif"],
        serif: ["var(--font-dm-serif)", "Georgia", "serif"],
        mono: ["var(--font-ibm-mono)", "Courier New", "monospace"],
      },
      colors: {
        oxford: {
          DEFAULT: "#002147",
          light: "#003366",
          dark: "#001a38",
          darker: "#00122a",
        },
        steel: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
        },
        cream: "#F8F7F4",
        success: "#059669",
        warning: "#d97706",
        danger: "#dc2626",
        info: "#2563eb",
      },
      borderRadius: {
        none: "0",
        sm: "2px",
        DEFAULT: "3px",
        md: "4px",
        lg: "4px",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
