"use client";

import { useState } from "react";
import type { ProcessResult } from "@/lib/api";
import {
  exportScans,
  type ExportReceipt,
  type ScanExport,
} from "@/lib/exportResults";
import { saveToHistory } from "@/lib/history";

type Props = {
  data: ProcessResult;
  capturedAt: string;
  scans: ScanExport[];
  onHome: () => void;
  onRestart: () => void;
  hideCaptureActions?: boolean;
};

export default function Result({
  data,
  capturedAt,
  scans,
  onHome,
  onRestart,
  hideCaptureActions = false,
}: Props) {
  const [showGraph, setShowGraph] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [saveError, setSaveError] = useState("");
  const [savingHistory, setSavingHistory] = useState(false);
  const [historySaved, setHistorySaved] = useState(false);

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

  async function saveHistory() {
    setSavingHistory(true);
    try {
      await saveToHistory(capturedAt || new Date().toISOString(), data);
      setHistorySaved(true);
    } catch (error) {
      console.error("Saving to history failed:", error);
    } finally {
      setSavingHistory(false);
    }
  }

  return (
    <section className="flex min-h-dvh flex-col bg-black px-1 pt-1 pb-2">
      <header className="pb-1">
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
        >
          <BackIcon />
        </button>
      </header>

      <div className="grid gap-3 sm:grid-cols-2">
        <figure>
          <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-neutral-950">
            <img
              src={data.original}
              alt="Original input"
              className="aspect-[8/5] max-h-[32dvh] w-full object-contain sm:max-h-[50dvh]"
            />
          </div>

          <figcaption className="mt-2 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-neutral-600">
            Input
          </figcaption>
        </figure>

        <figure>
          <div className="overflow-hidden rounded-2xl border border-emerald-400/20 bg-neutral-950">
            <img
              src={showGraph ? data.graph : data.overlay}
              alt={showGraph ? "Probable vein graph" : "Probable veins overlay"}
              className="aspect-[8/5] max-h-[32dvh] w-full object-contain sm:max-h-[50dvh]"
            />
          </div>

          <figcaption className="mt-2 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-emerald-400">
            {showGraph ? "Vein graph" : "Vein overlay"}
          </figcaption>
        </figure>
      </div>

      <div className="flex items-center justify-center gap-4 pt-2">
        <div className="flex flex-col items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowGraph((current) => !current)}
            aria-label={showGraph ? "Show overlay" : "Show vein graph"}
            className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95"
          >
            <LayersIcon />
          </button>

          <span className="whitespace-nowrap text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
            {showGraph ? "Overlay" : "Graph"}
          </span>
        </div>

        {!hideCaptureActions && (
          <div className="flex flex-col items-center gap-1.5">
            <button
              type="button"
              onClick={onRestart}
              aria-label="Start a new scan"
              className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-400 text-black transition active:scale-95"
            >
              <ScanIcon />
            </button>

            <span className="whitespace-nowrap text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-500">
              New scan
            </span>
          </div>
        )}

        <div className="flex flex-col items-center gap-1.5">
          <button
            type="button"
            onClick={saveImages}
            disabled={saving}
            aria-label={`Download ${scans.length} scan${scans.length === 1 ? "" : "s"}`}
            className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95 disabled:cursor-wait disabled:opacity-50"
          >
            {saving ? (
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-emerald-400" />
            ) : (
              <SaveIcon />
            )}
          </button>

          <span className="whitespace-nowrap text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
            Download
          </span>
        </div>

        {!hideCaptureActions && (
          <div className="flex flex-col items-center gap-1.5">
            <button
              type="button"
              onClick={() => void saveHistory()}
              disabled={savingHistory || historySaved}
              aria-label={historySaved ? "Saved to history" : "Save to history"}
              className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95 disabled:cursor-default disabled:opacity-50"
            >
              {savingHistory ? (
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-emerald-400" />
              ) : (
                <BookmarkIcon filled={historySaved} />
              )}
            </button>

            <span className="whitespace-nowrap text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
              {historySaved ? "Saved" : "Save"}
            </span>
          </div>
        )}
      </div>

      <div
        aria-live="polite"
        className="min-h-6 pt-3 text-center text-xs text-neutral-500"
      >
        {(saveMessage || saveError) && (
          <span className={saveError ? "text-red-400" : "text-emerald-400"}>
            {saveError || saveMessage}
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
      className="h-5 w-5"
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

function BookmarkIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1Z" />
    </svg>
  );
}
