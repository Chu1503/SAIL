"use client";

import { useEffect, useState } from "react";
import type { ProcessResult } from "@/lib/api";
import {
  deleteFromHistory,
  getHistory,
  type HistoryEntry,
} from "@/lib/history";

type Props = {
  onBack: () => void;
  onSelect: (capturedAt: string, result: ProcessResult) => void;
};

export default function History({ onBack, onSelect }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHistory()
      .then((loaded) => {
        if (!cancelled) setEntries(loaded);
      })
      .catch((error) => {
        console.error("Failed to load history:", error);
        if (!cancelled) setEntries([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function removeEntry(id: string, event: React.MouseEvent) {
    event.stopPropagation();
    await deleteFromHistory(id);
    setEntries((current) => current?.filter((entry) => entry.id !== id) ?? null);
  }

  return (
    <section className="flex min-h-dvh flex-col">
      <header className="flex items-center gap-4 pb-5">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back"
          className="flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
        >
          <BackIcon />
        </button>
        <h1 className="text-sm font-semibold uppercase tracking-[0.18em] text-neutral-300">
          History
        </h1>
      </header>

      {entries === null && (
        <div className="flex flex-1 items-center justify-center">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-emerald-400" />
        </div>
      )}

      {entries !== null && entries.length === 0 && (
        <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
          <p className="text-sm text-neutral-500">
            No saved scans yet. Tap &ldquo;Save to history&rdquo; on a result to keep
            it here.
          </p>
        </div>
      )}

      {entries !== null && entries.length > 0 && (
        <div className="grid grid-cols-2 gap-3 pb-8 sm:grid-cols-3">
          {entries.map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => onSelect(entry.capturedAt, entry.result)}
              className="group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-neutral-950 text-left transition active:scale-95"
            >
              <img
                src={entry.result.overlay}
                alt="Saved vein overlay"
                className="aspect-square w-full object-cover"
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent p-2.5">
                <p className="text-[10px] font-medium text-neutral-300">
                  {formatTimestamp(entry.capturedAt)}
                </p>
              </div>
              <button
                type="button"
                onClick={(event) => void removeEntry(entry.id, event)}
                aria-label="Delete saved scan"
                className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/60 text-neutral-300 opacity-0 transition group-hover:opacity-100 active:scale-95"
              >
                <TrashIcon />
              </button>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
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

function TrashIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
    </svg>
  );
}
