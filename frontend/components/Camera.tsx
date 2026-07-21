// Camera capture: on the web/laptop it uses getUserMedia; inside the native Android app it uses the external USB (UVC) camera plugin.
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Capacitor } from "@capacitor/core";
import { UsbCamera } from "@periksa/cap-usb-camera";
import { btnGhost, btnPrimary } from "@/lib/ui";

type Props = {
  onCapture: (blob: Blob, url: string) => void;
  onCancel: () => void;
};

type UsbResult = {
  status_code: number;
  exit_code?: string;
  data?: { dataURL?: string; fileURI?: string };
};

export default function Camera({ onCapture, onCancel }: Props) {
  const native = Capacitor.isNativePlatform();

  //
  console.log("Capacitor platform:", Capacitor.getPlatform());
  console.log("Native platform:", native);
  console.log("UsbCamera plugin available:", Capacitor.isPluginAvailable("UsbCamera"));
  //

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  const openUsbCamera = useCallback(async () => {
    setError("");
    try {
      const res = (await UsbCamera.getPhoto({ saveToStorage: false })) as UsbResult;
      if (res.status_code === -1 && res.data?.dataURL) {
        const blob = await (await fetch(res.data.dataURL)).blob();
        onCapture(blob, res.data.dataURL);
      } else if (res.exit_code === "user_canceled") {
        onCancel();
      } else if (res.exit_code === "exit_no_device") {
        setError("No camera detected.");
      } else {
        setError("Couldn't capture from the camera.");
      }
    } catch (err) {
  const message =
    err instanceof Error
      ? `${err.name}: ${err.message}`
      : JSON.stringify(err);

  console.error("USB CAMERA EXCEPTION:", err);
  setError(`USB camera error: ${message}`);
}
  }, [onCapture, onCancel]);

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
      setDevices(all.filter((d) => d.kind === "videoinput"));
      const active = stream.getVideoTracks()[0]?.getSettings().deviceId;
      setDeviceId(selectedId || active || "");
      setReady(true);
    } catch {
      setError("Couldn't open the camera. Allow camera access and make sure your camera is connected.");
    }
  }, []);

  useEffect(() => {
    if (native) {
      openUsbCamera();
      return;
    }
    start();
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [native, start, openUsbCamera]);

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

  if (native) {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black">
      {error ? (
        <div className="px-8 text-center">
          <p className="text-sm leading-6 text-red-400">{error}</p>

          <div className="mt-7 flex justify-center gap-5">
            <button
              type="button"
              onClick={onCancel}
              aria-label="Back"
              className="flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-white/[0.04]"
            >
              <BackIcon />
            </button>

            <button
              type="button"
              onClick={openUsbCamera}
              aria-label="Try again"
              className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-400 text-black"
            >
              <RetryIcon />
            </button>
          </div>
        </div>
      ) : (
        <div className="relative h-16 w-16">
          <div className="absolute inset-0 rounded-full border border-white/10" />
          <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-emerald-400 border-r-emerald-400/40" />
          <div className="absolute inset-[10px] animate-pulse rounded-full bg-emerald-400/10" />
        </div>
      )}
    </div>
  );
}

  return (
    <div className="p-3 sm:p-4">
      <div className="relative flex items-center justify-center overflow-hidden bg-black">
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
              className="mt-4 w-full rounded-xl px-4 py-2.5 text-sm text-neutral-200 outline-none focus:border-emerald-400/40"
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