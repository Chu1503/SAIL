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
        setError("No USB camera detected. Plug in your NIR camera over USB-C and try again.");
      } else {
        setError("Couldn't capture from the USB camera.");
      }
    } catch {
      setError("USB camera error. Make sure the camera is connected and allowed.");
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
      <div className="flex flex-col items-center rounded-3xl border border-white/10 bg-white/[0.02] px-6 py-16 text-center">
        {error ? (
          <>
            <p className="text-sm text-red-400">{error}</p>
            <div className="mt-6 flex justify-center gap-3">
              <button onClick={onCancel} className={btnGhost}>Back</button>
              <button onClick={openUsbCamera} className={btnPrimary}>Try again</button>
            </div>
          </>
        ) : (
          <>
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/15 border-t-emerald-400" />
            <p className="mt-5 text-sm text-neutral-400">Opening USB camera…</p>
          </>
        )}
      </div>
    );
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
