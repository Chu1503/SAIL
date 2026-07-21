// Root layout: metadata, PWA + viewport config, service-worker registration, and the dark app shell.
import type { Metadata, Viewport } from "next";
import "./globals.css";
import RegisterSW from "@/components/RegisterSW";

export const metadata: Metadata = {
  title: "VEINZ — Vein Visualizer",
  description: "Near-infrared vein visualization and mapping",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "VEINZ" },
};

export const viewport: Viewport = {
  themeColor: "#08090a",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-dvh bg-[#08090a] text-neutral-200 antialiased">
        {children}
        <RegisterSW />
      </body>
    </html>
  );
}
