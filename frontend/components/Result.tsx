// Result view: shows the original capture beside the detected-vein overlay, with a mask toggle.
"use client";

import { useState } from "react";
import type { ProcessResult } from "@/lib/api";
import { btnGhost, btnPrimary } from "@/lib/ui";

type Props = {
  data: ProcessResult;
  onRestart: () => void;
};

export default function Result({ data, onRestart }: Props) {
  const [showMask, setShowMask] = useState(false);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.02] p-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <figure>
          <img src={data.original} alt="Original capture" className="w-full rounded-2xl border border-white/10" />
          <figcaption className="mt-2 text-center text-xs text-neutral-500">Input</figcaption>
        </figure>
        <figure>
          <img
            src={showMask ? data.mask : data.overlay}
            alt="Detected veins"
            className="w-full rounded-2xl border border-emerald-400/20"
          />
          <figcaption className="mt-2 text-center text-xs text-emerald-400/80">
            {showMask ? "Vein mask" : "Veins detected"}
          </figcaption>
        </figure>
      </div>

      <div className="mt-5 flex justify-center gap-3">
        <button onClick={() => setShowMask((s) => !s)} className={btnGhost}>
          {showMask ? "Show overlay" : "Show mask"}
        </button>
        <button onClick={onRestart} className={btnPrimary}>New scan</button>
      </div>
    </div>
  );
}
