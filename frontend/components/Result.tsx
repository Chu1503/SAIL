"use client";

import { useState } from "react";
import type { ProcessResult } from "@/lib/api";
import {
  exportScans,
  type ExportReceipt,
  type ScanExport,
} from "@/lib/exportResults";

type Props = {
  data: ProcessResult;
  scans: ScanExport[];
  onHome: () => void;
  onRestart: () => void;
  // onNewScan: () => void;
};

export default function Result({
  data,
  scans,
  onHome,
  onRestart,
}: Props) {
  const [showMask, setShowMask] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [saveError, setSaveError] = useState("");

  async function saveImages() {
    setSaving(true);
    setSaveMessage("");
    setSaveError("");

    try {
      const receipt: ExportReceipt = await exportScans(scans);
      setSaveMessage(
        `${receipt.fileCount} files saved to ${receipt.location}`
      );
    } catch (error) {
      console.error("Image export failed:", error);
      setSaveError(
        error instanceof Error
          ? error.message
          : "Images could not be saved. Please try again."
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="flex min-h-[calc(100dvh-3rem)] flex-col">
      <header className="flex items-center pb-4">
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
        >
          <BackIcon />
        </button>
      </header>

      <div className="grid flex-1 content-center gap-7 sm:grid-cols-2">
        <figure>
          <div className="bg-neutral-950">
            <img
              src={data.processed}
              alt="Post-processed scan"
              className="max-h-[55dvh] w-full object-contain"
            />
          </div>

          <figcaption className="mt-3 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-neutral-600">
            Post processed
          </figcaption>
        </figure>

        <figure>
          <div className="bg-neutral-950">
            <img
              src={showMask ? data.mask : data.overlay}
              alt={showMask ? "Vein mask" : "Detected veins"}
              className="max-h-[55dvh] w-full object-contain"
            />
          </div>

          <figcaption className="mt-3 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-emerald-400">
            {showMask ? "Vein mask" : "Detected veins"}
          </figcaption>
        </figure>
      </div>

      <div className="flex items-end justify-center gap-8 pt-4">
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => setShowMask((current) => !current)}
            aria-label={showMask ? "Show overlay" : "Show mask"}
            className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95"
          >
            <LayersIcon />
          </button>

          <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
            {showMask ? "Overlay" : "Mask"}
          </span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={onRestart}
            aria-label="Start a new scan"
            className="flex h-20 w-20 items-center justify-center rounded-full bg-emerald-400 text-black shadow-[0_0_40px_rgba(52,211,153,0.12)] transition active:scale-95"
          >
            <ScanIcon />
          </button>

          <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-500">
            New scan
          </span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={saveImages}
            disabled={saving}
            aria-label={`Save ${scans.length} scan${scans.length === 1 ? "" : "s"}`}
            className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95 disabled:cursor-wait disabled:opacity-50"
          >
            {saving ? (
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-emerald-400" />
            ) : (
              <SaveIcon />
            )}
          </button>

          <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
            Save images
          </span>
        </div>
      </div>

      <div
        aria-live="polite"
        className="min-h-6 pt-3 text-center text-xs text-neutral-500"
      >
        {saveMessage || saveError ? (
          <span className={saveError ? "text-red-400" : "text-emerald-400"}>
            {saveError || saveMessage}
          </span>
        ) : (
          <span>
            {scans.length} scan{scans.length === 1 ? "" : "s"} ·{" "}
            {scans.length * 3} files
          </span>
        )}
      </div>
    </section>
  );
}

function BackIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m15 18-6-6 6-6" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m12 2 9 5-9 5-9-5 9-5Z" />
      <path d="m3 12 9 5 9-5" />
      <path d="m3 17 9 5 9-5" />
    </svg>
  );
}

function ScanIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-7 w-7"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 8V5a1 1 0 0 1 1-1h3" />
      <path d="M16 4h3a1 1 0 0 1 1 1v3" />
      <path d="M20 16v3a1 1 0 0 1-1 1h-3" />
      <path d="M8 20H5a1 1 0 0 1-1-1v-3" />
      <path d="M7 12h10" />
    </svg>
  );
}

function SaveIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v12" />
      <path d="m7 10 5 5 5-5" />
      <path d="M5 21h14a2 2 0 0 0 2-2v-3" />
      <path d="M3 16v3a2 2 0 0 0 2 2" />
    </svg>
  );
}
