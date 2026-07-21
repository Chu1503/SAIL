// Shows an "Install app" button on Android/desktop Chrome and an Add-to-Home-Screen hint on iOS.
"use client";

import { useEffect, useState } from "react";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

type NavigatorStandalone = Navigator & { standalone?: boolean };

export default function InstallPrompt() {
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(null);
  const [isIOS, setIsIOS] = useState(false);
  const [showSheet, setShowSheet] = useState(false);
  const [installed, setInstalled] = useState(false);

  useEffect(() => {
    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      (navigator as NavigatorStandalone).standalone === true;
    if (standalone) {
      setInstalled(true);
      return;
    }

    setIsIOS(/iphone|ipad|ipod/i.test(navigator.userAgent));

    const onPrompt = (e: Event) => {
      e.preventDefault();
      setDeferred(e as BeforeInstallPromptEvent);
    };
    const onInstalled = () => setInstalled(true);

    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  if (installed) return null;

  if (deferred) {
    return (
      <button
        onClick={async () => {
          await deferred.prompt();
          setDeferred(null);
        }}
        className="rounded-full border border-emerald-400/40 bg-emerald-400/10 px-3.5 py-1.5 text-xs font-medium text-emerald-300 transition hover:bg-emerald-400/20"
      >
        Install app
      </button>
    );
  }

  if (isIOS) {
    return (
      <>
        <button
          onClick={() => setShowSheet(true)}
          className="rounded-full border border-white/15 px-3.5 py-1.5 text-xs font-medium text-neutral-300 transition hover:bg-white/5"
        >
          Install
        </button>
        {showSheet && (
          <div
            className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-3"
            onClick={() => setShowSheet(false)}
          >
            <div className="w-full max-w-md rounded-3xl border border-white/10 bg-neutral-950 p-6 text-center">
              <p className="text-sm font-medium text-neutral-100">Install VEINZ</p>
              <p className="mt-2 text-xs text-neutral-400">
                Tap the Share icon in Safari, then choose &ldquo;Add to Home Screen&rdquo;.
              </p>
              <button className="mt-5 rounded-full bg-emerald-400 px-6 py-2 text-xs font-medium text-black">
                Got it
              </button>
            </div>
          </div>
        )}
      </>
    );
  }

  return null;
}
