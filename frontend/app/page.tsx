// Main page: capture or upload an image, run the pipeline, and show the vein overlay.
"use client";

import { useState, type ChangeEvent } from "react";
import Camera from "@/components/Camera";
import Result from "@/components/Result";
import History from "@/components/History";
import InstallPrompt from "@/components/InstallPrompt";
import DisclaimerGate from "@/components/DisclaimerGate";
import { processImage, type ProcessResult } from "@/lib/api";
import { compressImage } from "@/lib/compressImage";
import type { ScanExport } from "@/lib/exportResults";

type Stage =
  | "idle"
  | "camera"
  | "processing"
  | "processingError"
  | "result"
  | "history";

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [resultCapturedAt, setResultCapturedAt] = useState("");
  const [completedScans, setCompletedScans] = useState<ScanExport[]>([]);
  const [pendingCapture, setPendingCapture] = useState<Blob | null>(null);
  const [pendingPreview, setPendingPreview] = useState("");
  const [error, setError] = useState("");

  function goHome() {
    setStage("idle");
    setResult(null);
    setResultCapturedAt("");
    setCompletedScans([]);
    setPendingCapture(null);
    setPendingPreview("");
    setError("");
  }

  function openHistory() {
    setError("");
    setStage("history");
  }

  function viewHistoryEntry(capturedAt: string, entryResult: ProcessResult) {
    setResult(entryResult);
    setResultCapturedAt(capturedAt);
    setCompletedScans([{ capturedAt, result: entryResult }]);
    setStage("result");
  }

  function openCamera() {
    setError("");
    setStage("camera");
  }

  function cancelCamera() {
    setError("");
    setStage(result ? "result" : "idle");
  }

  async function processCapture(capturedBlob: Blob) {
    setStage("processing");
    setError("");

    try {
      const processedResult = await processImage(capturedBlob);
      const capturedAt = new Date().toISOString();
      setResult(processedResult);
      setResultCapturedAt(capturedAt);
      setCompletedScans((current) => [
        ...current,
        { capturedAt, result: processedResult },
      ]);
      setPendingCapture(null);
      setPendingPreview("");
      setStage("result");
    } catch (err) {
      console.error("Processing failed:", err);
      setError(
        err instanceof Error
          ? err.message
          : "The processing service could not analyze this capture."
      );
      setStage("processingError");
    }
  }

  async function handleCapture(capturedBlob: Blob, capturedUrl: string) {
    setPendingPreview(capturedUrl);
    setStage("processing");

    const uploadBlob = await compressImage(capturedBlob);
    setPendingCapture(uploadBlob);
    void processCapture(uploadBlob);
  }

  function handleUpload(file: File) {
    const previewUrl = URL.createObjectURL(file);
    handleCapture(file, previewUrl);
  }

  function retakeCapture() {
    setPendingCapture(null);
    setPendingPreview("");
    openCamera();
  }

  return (
    <DisclaimerGate>
    <main className="mx-auto flex min-h-dvh w-full max-w-3xl flex-col bg-black">
      {stage === "idle" && (
        <>
          <header className="flex items-center justify-between">
            <button
              type="button"
              onClick={openHistory}
              aria-label="View saved history"
              className="flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] transition active:scale-95"
            >
              <HistoryIcon />
            </button>
            <InstallPrompt />
          </header>

          <Idle onStart={openCamera} onUpload={handleUpload} />

          <footer className="mt-auto pt-8 text-center text-[10px] font-medium uppercase tracking-[0.3em] text-neutral-700">
            SAIL
          </footer>
        </>
      )}

      {stage === "camera" && (
        <Camera onCapture={handleCapture} onCancel={cancelCamera} />
      )}

      {stage === "processing" && <LoadingScreen preview={pendingPreview} />}

      {stage === "processingError" && (
        <ProcessingError
          preview={pendingPreview}
          message={error}
          onRetry={() => {
            if (pendingCapture) void processCapture(pendingCapture);
          }}
          onRetake={retakeCapture}
        />
      )}

      {stage === "history" && (
        <History onBack={goHome} onSelect={viewHistoryEntry} />
      )}

      {stage === "result" && result && (
        <Result
          data={result}
          capturedAt={resultCapturedAt}
          scans={completedScans}
          onHome={goHome}
          onRestart={openCamera}
        />
      )}
    </main>
    </DisclaimerGate>
  );
}

function ProcessingError({
  preview,
  message,
  onRetry,
  onRetake,
}: {
  preview: string;
  message: string;
  onRetry: () => void;
  onRetake: () => void;
}) {
  return (
    <section className="flex min-h-dvh flex-col items-center justify-center px-6 text-center">
      {preview && (
        <img
          src={preview}
          alt="Captured image awaiting processing"
          className="mb-7 aspect-[8/5] max-h-[45dvh] w-full max-w-xl rounded-2xl border border-white/10 object-contain"
        />
      )}
      <h2 className="text-xl font-semibold text-white">Processing interrupted. Please try again later.</h2>
      
      {/* <p className="mt-2 max-w-md text-xs leading-5 text-red-400/80">
        {message}
      </p> */}
      <div className="mt-7 flex items-center gap-4">
        <button
          type="button"
          onClick={onRetake}
          className="rounded-full border border-white/15 px-5 py-3 text-sm text-neutral-300"
        >
          Retake
        </button>
        <button
          type="button"
          onClick={onRetry}
          className="flex items-center gap-2 rounded-full bg-emerald-400 px-5 py-3 text-sm font-semibold text-black"
        >
          <RetryIcon />
          Retry same image
        </button>
      </div>
    </section>
  );
}

function Idle({
  onStart,
  onUpload,
}: {
  onStart: () => void;
  onUpload: (file: File) => void;
}) {
  function selectImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    onUpload(file);
    // Permit selecting the same image again after returning to this screen.
    event.target.value = "";
  }

  return (
    <section className="flex flex-1 flex-col items-center text-center">
      <div className="pt-[16dvh]">
        {/* <p className="text-xs font-medium uppercase tracking-[0.35em] text-neutral-500">
          External vein imaging
        </p> */}

        {/* <h1 className="mt-4 text-5xl font-semibold tracking-[-0.06em] text-white sm:text-6xl">
          VeinSight
        </h1>

        <p className="mx-auto mt-4 max-w-xs text-sm leading-6 text-neutral-500">
          Short Description
        </p> */}
      </div>

      <div className="mt-auto flex flex-col items-center pb-5 pt-16">
        <button
          type="button"
          onClick={onStart}
          aria-label="Start capture"
          className="flex h-36 w-36 items-center justify-center rounded-full bg-emerald-400 text-black shadow-[0_0_60px_rgba(52,211,153,0.16)] transition active:scale-95"
        >
          <span className="text-sm font-semibold uppercase tracking-[0.12em]">
            Start
          </span>
        </button>

        <label className="mt-7 flex cursor-pointer items-center gap-2 rounded-full border border-white/15 bg-white/[0.03] px-5 py-3 text-sm font-medium text-neutral-300 transition hover:border-emerald-400/30 hover:text-white active:scale-95">
          <UploadIcon />
          Upload image
          <input
            type="file"
            accept="image/*"
            onChange={selectImage}
            className="sr-only"
          />
        </label>

      </div>
    </section>
  );
}

function HistoryIcon() {
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
      <path d="M3 12a9 9 0 1 0 2.6-6.35" />
      <path d="M3 4v6h6" />
      <path d="M12 8v4l3 2" />
    </svg>
  );
}

function LoadingScreen({ preview }: { preview: string }) {
  return (
    <section className="fixed inset-0 z-50 flex items-center justify-center bg-black">
      {preview && (
        <img
          src={preview}
          alt="Captured image being processed"
          className="absolute inset-0 h-full w-full object-cover opacity-30 blur-sm"
        />
      )}
      <div className="relative h-16 w-16">
        <div className="absolute inset-0 rounded-full border border-white/10" />
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-emerald-400 border-r-emerald-400/40" />
        <div className="absolute inset-[10px] animate-pulse rounded-full bg-emerald-400/10" />
      </div>
    </section>
  );
}

function RetryIcon() {
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
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v6h6" />
    </svg>
  );
}

function UploadIcon() {
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
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M5 20h14a2 2 0 0 0 2-2v-3" />
      <path d="M3 15v3a2 2 0 0 0 2 2" />
    </svg>
  );
}
