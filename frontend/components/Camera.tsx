// Camera capture: opens the selected (IR) camera feed, lets you pick the device, and grabs a still frame.
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { btnGhost, btnPrimary } from "@/lib/ui";

type Props = {
  onCapture: (blob: Blob, url: string) => void;
  onCancel: () => void;
};

export default function Camera({ onCapture, onCancel }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  const start = useCallback(async (selectedId?: string) => {
    setError("");
    setReady(false);
    try {
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedId
          ? { deviceId: { exact: selectedId } }
          : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      const active = stream.getVideoTracks()[0]?.getSettings().deviceId;
      setDeviceId(selectedId || active || "");
      setReady(true);
    } catch {
      setError("Couldn't open the camera. Allow camera access and make sure your camera is plugged in.");
    }
  }, []);

  useEffect(() => {
    start();
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [start]);

  function onSelect(e: React.ChangeEvent<HTMLSelectElement>) {
    const id = e.target.value;
    setDeviceId(id);
    start(id);
  }

  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) onCapture(blob, URL.createObjectURL(blob));
    }, "image/png");
  }

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.02] p-3 sm:p-4">
      <div className="relative flex items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black">
        <video ref={videoRef} playsInline muted className="max-h-[65vh] w-full object-contain" />
        {!ready && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-emerald-400" />
          </div>
        )}
      </div>

      {error ? (
        <div className="mt-4 text-center">
          <p className="text-sm text-red-400">{error}</p>
          <div className="mt-4 flex justify-center gap-3">
            <button onClick={onCancel} className={btnGhost}>Back</button>
            <button onClick={() => start()} className={btnPrimary}>Try again</button>
          </div>
        </div>
      ) : (
        <>
          {devices.length > 1 && (
            <select
              value={deviceId}
              onChange={onSelect}
              className="mt-4 w-full rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2.5 text-sm text-neutral-200 outline-none focus:border-emerald-400/40"
            >
              {devices.map((d, i) => (
                <option key={d.deviceId} value={d.deviceId} className="bg-neutral-900">
                  {d.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
          )}
          <div className="mt-4 flex justify-center gap-3">
            <button onClick={onCancel} className={btnGhost}>Cancel</button>
            <button onClick={capture} disabled={!ready} className={btnPrimary}>Capture</button>
          </div>
        </>
      )}
    </div>
  );
}
