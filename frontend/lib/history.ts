// On-device history: captures the user explicitly chooses to keep. Nothing
// is saved automatically -- a scan only appears here if the user taps "Save
// to history". Uses IndexedDB directly (not a Capacitor plugin) so the same
// code works identically on the website and inside the Android WebView.
import type { ProcessResult } from "@/lib/api";

export type HistoryEntry = {
  id: string;
  capturedAt: string;
  result: ProcessResult;
};

const DB_NAME = "veinsight-history";
const DB_VERSION = 1;
const STORE_NAME = "captures";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveToHistory(
  capturedAt: string,
  result: ProcessResult
): Promise<HistoryEntry> {
  const entry: HistoryEntry = {
    id: `${capturedAt}_${Math.random().toString(36).slice(2, 8)}`,
    capturedAt,
    result,
  };

  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();

  return entry;
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const db = await openDatabase();
  const entries = await new Promise<HistoryEntry[]>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onsuccess = () => resolve(request.result as HistoryEntry[]);
    request.onerror = () => reject(request.error);
  });
  db.close();

  return entries.sort((a, b) => b.capturedAt.localeCompare(a.capturedAt));
}

export async function deleteFromHistory(id: string): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

export async function deleteManyFromHistory(ids: string[]): Promise<void> {
  if (ids.length === 0) return;

  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    for (const id of ids) store.delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}
