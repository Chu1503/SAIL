// Root layout: metadata, PWA configuration, font, and global app shell.
import type { Metadata, Viewport } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import RegisterSW from "@/components/RegisterSW";

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
});

export const metadata: Metadata = {
  title: "VEINZ",
  description: "External-camera vein visualization",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "VEINZ",
  },
};

export const viewport: Viewport = {
  themeColor: "#000000",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${geist.variable} min-h-dvh bg-black text-white antialiased`}
      >
        {children}
        <RegisterSW />
      </body>
    </html>
  );
}