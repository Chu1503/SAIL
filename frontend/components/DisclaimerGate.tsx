"use client";

// Gate shown once before first use: VeinSight highlights probable veins
// and a suggested injection point using automated image analysis, not a
// certified medical device. This needs to be acknowledged before someone
// can act on what the app shows them, not just linked in a footer.
import { useEffect, useState } from "react";

const ACK_KEY = "veinsight_disclaimer_ack_v1";

export default function DisclaimerGate({
  children,
}: {
  children: React.ReactNode;
}) {
  const [ready, setReady] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    // This is a static export with no real server render, so localStorage
    // must be read post-mount, not in a lazy useState initializer (that
    // would run during the build's static prerender pass too, where
    // window does not exist, and risks a hydration mismatch).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAcknowledged(window.localStorage.getItem(ACK_KEY) === "1");
    setReady(true);
  }, []);

  function acknowledge() {
    window.localStorage.setItem(ACK_KEY, "1");
    setAcknowledged(true);
  }

  if (!ready) {
    return <div className="min-h-dvh bg-black" />;
  }

  if (!acknowledged) {
    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black px-6">
        <div className="w-full max-w-sm">
          <h1 className="text-lg font-semibold text-white">Before you start</h1>
          <p className="mt-4 text-sm leading-6 text-neutral-400">
            VeinSight uses automated image analysis to highlight probable
            veins and suggest a possible injection point. It is{" "}
            <strong className="text-neutral-200">
              not a certified medical device
            </strong>{" "}
            and can be wrong. It is not a substitute for the training or
            judgment of a qualified healthcare professional, and should not
            be the sole basis for performing venipuncture or any medical
            procedure.
          </p>
          <p className="mt-3 text-xs text-neutral-500">
            By continuing, you agree to our{" "}
            <a href="/terms" className="text-emerald-400 underline">
              Terms &amp; Disclaimer
            </a>{" "}
            and{" "}
            <a href="/privacy" className="text-emerald-400 underline">
              Privacy Policy
            </a>
            .
          </p>
          <button
            type="button"
            onClick={acknowledge}
            className="mt-6 w-full rounded-full bg-emerald-400 py-3 text-sm font-semibold text-black transition active:scale-95"
          >
            I understand, continue
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
