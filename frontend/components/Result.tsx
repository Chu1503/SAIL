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
  const [showGraph, setShowGraph] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [saveError, setSaveError] = useState("");
  const armIsolation = data.analysis.pipeline?.armIsolation ?? {
    name: data.analysis.armSegmentation?.method ?? "Unknown",
    tier: "Legacy response",
    runtime: "CPU",
    status: "fallback" as const,
  };
  const veinExtraction = data.analysis.pipeline?.veinExtraction ?? {
    name: "Frangi + Sato + multiscale black-hat",
    tier: "Training-free",
    runtime: "CPU",
    status: "primary" as const,
  };

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
      <header className="pb-5">
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
        >
          <BackIcon />
        </button>
      </header>

      <div className="grid flex-1 content-center gap-5 sm:grid-cols-2">
        <figure>
          <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-neutral-950 shadow-2xl shadow-black">
            <img
              src={data.original}
              alt="Original input"
              className="aspect-[8/5] max-h-[55dvh] w-full object-contain"
            />
          </div>

          <figcaption className="mt-3 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-neutral-600">
            Input
          </figcaption>
        </figure>

        <figure>
          <div className="overflow-hidden rounded-2xl border border-emerald-400/20 bg-neutral-950 shadow-[0_20px_70px_rgba(16,185,129,0.08)]">
            <img
              src={showGraph ? data.graph : data.overlay}
              alt={showGraph ? "Probable vein graph" : "Probable veins overlay"}
              className="aspect-[8/5] max-h-[55dvh] w-full object-contain"
            />
          </div>

          <figcaption className="mt-3 text-center text-[10px] font-medium uppercase tracking-[0.22em] text-emerald-400">
            {showGraph ? "Vein graph" : "Vein overlay"}
          </figcaption>
        </figure>
      </div>

      {/* <div className="mx-auto mt-5 grid w-full max-w-3xl gap-px overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.08] sm:grid-cols-2">
        <PipelineStage
          label="Arm isolation"
          name={armIsolation.name}
          tier={armIsolation.tier}
          runtime={armIsolation.runtime}
          fallback={armIsolation.status === "fallback"}
        />
        <PipelineStage
          label="Vein extraction"
          name={veinExtraction.name}
          tier={veinExtraction.tier}
          runtime={veinExtraction.runtime}
          fallback={veinExtraction.status === "fallback"}
        />
      </div> */}

      <div className="flex items-end justify-center gap-8 pt-5">
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => setShowGraph((current) => !current)}
            aria-label={showGraph ? "Show overlay" : "Show vein graph"}
            className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 transition active:scale-95"
          >
            <LayersIcon />
          </button>

          <span className="text-[10px] font-medium uppercase tracking-[0.16em] text-neutral-600">
            {showGraph ? "Overlay" : "Graph"}
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
        {(saveMessage || saveError) && (
          <span className={saveError ? "text-red-400" : "text-emerald-400"}>
            {saveError || saveMessage}
          </span>
        )}
      </div>
    </section>
  );
}

function PipelineStage({
  label,
  name,
  tier,
  runtime,
  fallback,
}: {
  label: string;
  name: string;
  tier: string;
  runtime: string;
  fallback: boolean;
}) {
  return (
    <div className="bg-neutral-950/95 px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[9px] font-medium uppercase tracking-[0.18em] text-neutral-600">
          {label}
        </span>
        <span
          className={
            fallback
              ? "rounded-full bg-amber-400/10 px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.14em] text-amber-300"
              : "rounded-full bg-emerald-400/10 px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.14em] text-emerald-300"
          }
        >
          {fallback ? "Fallback used" : "Primary"}
        </span>
      </div>
      <p className="mt-1.5 truncate text-xs font-medium text-neutral-200">
        {name}
      </p>
      <p className="mt-0.5 text-[10px] text-neutral-600">
        {tier} · {runtime}
      </p>
    </div>
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
