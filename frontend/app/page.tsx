// Main page: Start Camera, capture a frame, confirm it, run the pipeline, and show the vein overlay.
"use client";

import { useState } from "react";
import Camera from "@/components/Camera";
import Result from "@/components/Result";
import InstallPrompt from "@/components/InstallPrompt";
import { processImage, type ProcessResult } from "@/lib/api";
import { btnGhost, btnPrimary } from "@/lib/ui";

type Stage = "idle" | "camera" | "preview" | "processing" | "result";

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [blob, setBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");

  function reset() {
    setStage("idle");
    setBlob(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
  }

  function handleCapture(b: Blob, url: string) {
    setBlob(b);
    setPreviewUrl(url);
    setError("");
    setStage("preview");
  }

  async function runPipeline() {
    if (!blob) return;
    setStage("processing");
    setError("");
    try {
      const res = await processImage(blob);
      setResult(res);
      setStage("result");
    } catch {
      setError("Processing failed. Make sure the backend is running on port 8000.");
      setStage("preview");
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-3xl flex-col">
      <header className="flex items-center justify-between">
        <span className="text-sm font-semibold tracking-wide text-emerald-400">VEINZ</span>
        <InstallPrompt />
      </header>

      <div className="mt-8 sm:mt-10">
        {stage === "idle" && <Idle onStart={() => setStage("camera")} />}

        {stage === "camera" && <Camera onCapture={handleCapture} onCancel={reset} />}

        {stage === "preview" && (
          <div className="rounded-3xl border border-white/10 bg-white/[0.02] p-3 sm:p-4">
            <img src={previewUrl} alt="Captured frame" className="w-full rounded-2xl border border-white/10" />
            {error && <p className="mt-3 text-center text-sm text-red-400">{error}</p>}
            <p className="mt-4 text-center text-sm text-neutral-400">Is this frame clear enough?</p>
            <div className="mt-4 flex justify-center gap-3">
              <button onClick={() => setStage("camera")} className={btnGhost}>Retake</button>
              <button onClick={runPipeline} className={btnPrimary}>Use this photo</button>
            </div>
          </div>
        )}

        {stage === "processing" && (
          <div className="flex flex-col items-center rounded-3xl border border-white/10 bg-white/[0.02] px-6 py-20">
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/15 border-t-emerald-400" />
            <p className="mt-5 text-sm text-neutral-400">Loading…</p>
          </div>
        )}

        {stage === "result" && result && <Result data={result} onRestart={reset} />}
      </div>

      <footer className="mt-auto pt-10 text-center text-xs text-neutral-600">SAIL</footer>
    </main>
  );
}

function Idle({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex flex-col items-center rounded-3xl border border-white/10 bg-white/[0.02] px-6 py-16 text-center">
      <h2 className="text-xl font-medium text-neutral-100">Scan a forearm</h2>
      <p className="mt-2 max-w-sm text-sm text-neutral-400">
        Start your NIR camera feed and capture a frame. Veins are detected and mapped automatically.
      </p>
      <button onClick={onStart} className={`${btnPrimary} mt-8`}>Start Camera</button>
    </div>
  );
}
