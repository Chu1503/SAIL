// Main page: Start Camera, capture a frame, confirm it, run the pipeline, and show the vein overlay.
"use client";

import { useState } from "react";
import Camera from "@/components/Camera";
import Result from "@/components/Result";
import InstallPrompt from "@/components/InstallPrompt";
import { processImage, type ProcessResult } from "@/lib/api";

type Stage = "idle" | "camera" | "processing" | "result";

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");

  function goHome() {
    setStage("idle");
    setResult(null);
    setError("");
  }

  function openCamera() {
    setResult(null);
    setError("");
    setStage("camera");
  }

  async function handleCapture(capturedBlob: Blob) {
    setStage("processing");
    setError("");

    try {
      const processedResult = await processImage(capturedBlob);
      setResult(processedResult);
      setStage("result");
    } catch (err) {
      console.error("Processing failed:", err);
      setError("Processing failed. Check your connection and try again.");
      setStage("camera");
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-3xl flex-col bg-black">
      {stage === "idle" && (
        <>
          <header className="flex items-center justify-end">
            <InstallPrompt />
          </header>

          <Idle onStart={openCamera} />

          <footer className="mt-auto pt-8 text-center text-[10px] font-medium uppercase tracking-[0.3em] text-neutral-700">
            SAIL
          </footer>
        </>
      )}

      {stage === "camera" && (
        <Camera onCapture={handleCapture} onCancel={goHome} />
      )}

      {stage === "processing" && <LoadingScreen />}

      {stage === "result" && result && (
        <Result
          data={result}
          onHome={goHome}
          onRestart={openCamera}
        />
      )}
    </main>
  );
}

function Idle({ onStart }: { onStart: () => void }) {
  return (
    <section className="flex flex-1 flex-col items-center text-center">
      <div className="pt-[16dvh]">
        {/* <p className="text-xs font-medium uppercase tracking-[0.35em] text-neutral-500">
          External vein imaging
        </p> */}

        <h1 className="mt-4 text-5xl font-semibold tracking-[-0.06em] text-white sm:text-6xl">
          VEINZ
        </h1>

        <p className="mx-auto mt-4 max-w-xs text-sm leading-6 text-neutral-500">
          Short Description
        </p>
      </div>

      <div className="mt-auto pb-5 pt-16">
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
      </div>
    </section>
  );
}

function LoadingScreen() {
  return (
    <section className="fixed inset-0 z-50 flex items-center justify-center bg-black">
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

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m5 12 4 4L19 6" />
    </svg>
  );
}
