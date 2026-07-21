// Sends a captured image to the Python backend and returns the result image data URLs.
export type ProcessResult = { original: string; overlay: string; mask: string };

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export async function processImage(blob: Blob): Promise<ProcessResult> {
  const form = new FormData();
  form.append("file", blob, "capture.png");
  const res = await fetch(`${API_URL}/process`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Request failed");
  return (await res.json()) as ProcessResult;
}
