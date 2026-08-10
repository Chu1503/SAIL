"use client";

import { useEffect, useState } from "react";
import {
  deleteFromHistory,
  deleteManyFromHistory,
  getHistory,
  type HistoryEntry,
} from "@/lib/history";
import ConfirmDialog from "@/components/ConfirmDialog";

type Props = {
  onBack: () => void;
  onSelect: (entry: HistoryEntry) => void;
};

export default function History({ onBack, onSelect }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [confirmOpen, setConfirmOpen] = useState(false);

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

  function toggleSelectMode() {
    setSelectMode((current) => !current);
    setSelectedIds(new Set());
  }

  function toggleSelected(id: string) {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function handleEntryClick(entry: HistoryEntry) {
    if (selectMode) {
      toggleSelected(entry.id);
    } else {
      onSelect(entry);
    }
  }

  async function confirmDelete() {
    const ids = Array.from(selectedIds);
    await deleteManyFromHistory(ids);
    setEntries((current) => current?.filter((entry) => !selectedIds.has(entry.id)) ?? null);
    setSelectedIds(new Set());
    setSelectMode(false);
    setConfirmOpen(false);
  }

  return (
    <section className="flex flex-1 flex-col">
      <header className="flex items-center justify-between gap-4 pb-5">
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={selectMode ? toggleSelectMode : onBack}
            aria-label={selectMode ? "Cancel selection" : "Back"}
            className="flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
          >
            {selectMode ? <CloseIcon /> : <BackIcon />}
          </button>
          <h1 className="text-sm font-semibold uppercase tracking-[0.18em] text-neutral-300">
            {selectMode ? `${selectedIds.size} selected` : "History"}
          </h1>
        </div>

        {entries !== null && entries.length > 0 && (
          <div className="flex items-center gap-2">
            {selectMode && selectedIds.size > 0 && (
              <button
                type="button"
                onClick={() => setConfirmOpen(true)}
                className="rounded-full bg-red-500 px-4 py-2 text-xs font-semibold text-white transition active:scale-95"
              >
                Delete ({selectedIds.size})
              </button>
            )}
            {!selectMode && (
              <button
                type="button"
                onClick={toggleSelectMode}
                className="rounded-full border border-white/15 px-4 py-2 text-xs font-medium text-neutral-300 transition active:scale-95"
              >
                Select
              </button>
            )}
          </div>
        )}
      </header>

      {entries === null && (
        <div className="flex flex-1 items-center justify-center">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-emerald-400" />
        </div>
      )}

      {entries !== null && entries.length === 0 && (
        <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
          <p className="text-sm text-neutral-500">
            No saved scans yet.
          </p>
        </div>
      )}

      {entries !== null && entries.length > 0 && (
        <div className="grid grid-cols-2 gap-3 pb-8 sm:grid-cols-3">
          {entries.map((entry) => {
            const selected = selectedIds.has(entry.id);
            return (
              <button
                key={entry.id}
                type="button"
                onClick={() => handleEntryClick(entry)}
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

                {selectMode ? (
                  <div
                    className={`absolute left-2 top-2 flex h-6 w-6 items-center justify-center rounded-full border-2 transition ${
                      selected
                        ? "border-emerald-400 bg-emerald-400 text-black"
                        : "border-white/60 bg-black/40 text-transparent"
                    }`}
                  >
                    <CheckIcon />
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={(event) => void removeEntry(entry.id, event)}
                    aria-label="Delete saved scan"
                    className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/60 text-neutral-300 opacity-0 transition group-hover:opacity-100 active:scale-95"
                  >
                    <TrashIcon />
                  </button>
                )}
              </button>
            );
          })}
        </div>
      )}

      <ConfirmDialog
        open={confirmOpen}
        title="Delete selected scans?"
        message={`Are you sure you want to delete ${selectedIds.size} item${selectedIds.size === 1 ? "" : "s"}? This cannot be undone.`}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setConfirmOpen(false)}
      />
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

function CloseIcon() {
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
      <path d="M18 6 6 18" />
      <path d="M6 6l12 12" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M20 6 9 17l-5-5" />
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
