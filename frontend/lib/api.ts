// Sends a captured image to the Python backend and returns the result image data URLs.
export type ProcessResult = {
  original: string;
  processed: string;
  overlay: string;
  mask: string;
};

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export async function processImage(blob: Blob): Promise<ProcessResult> {
  const form = new FormData();

  form.append(
    "file",
    blob,
    blob.type === "image/jpeg" ? "capture.jpg" : "capture.png"
  );

  const endpoint = `${API_URL}/process`;

  console.log("Sending image to backend:", {
    endpoint,
    type: blob.type,
    size: blob.size,
  });

  const res = await fetch(endpoint, {
    method: "POST",
    body: form,
  });

  const responseText = await res.text();

  console.log("Backend response:", {
    status: res.status,
    statusText: res.statusText,
    bodyPreview: responseText.slice(0, 500),
  });

  if (!res.ok) {
    throw new Error(
      `Backend returned ${res.status}: ${responseText || res.statusText}`
    );
  }

  const result = JSON.parse(responseText) as Partial<ProcessResult> & {
    error?: string;
  };

  if (result.error) {
    throw new Error(result.error);
  }

  if (!result.original || !result.processed || !result.overlay || !result.mask) {
    throw new Error("Backend response is missing image results");
  }

  return result as ProcessResult;
}
