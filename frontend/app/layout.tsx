import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { DebugStyles } from "./debug-styles";

export const metadata: Metadata = {
  title: "M&A Market Maps",
  description: "Workspace-based M&A research and due diligence",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="font-sans">
      <body 
        className="bg-steel-200 text-steel-900 antialiased font-sans" 
        style={{ fontFamily: 'Arial, Helvetica, sans-serif' }}
      >
        <DebugStyles />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
