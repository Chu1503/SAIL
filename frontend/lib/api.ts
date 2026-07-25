// Sends a captured image to the Python backend and returns the result image data URLs.
export type PipelineStage = {
  name: string;
  tier: string;
  runtime: string;
  status: "primary" | "fallback";
};

export type ProcessResult = {
  original: string;
  processed: string;
  overlay: string;
  graph: string;
  analysis: {
    signalQuality: number;
    pathConfidence: number;
    vesselCoverage: number;
    connectedVessels: number;
    segments: number;
    endpoints: number;
    junctions: number;
    warnings: string[];
    armSegmentation?: {
      method: string;
      confidence: number;
    };
    pipeline?: {
      armIsolation: PipelineStage;
      veinExtraction: PipelineStage;
    };
  };
};

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export async function processImage(blob: Blob): Promise<ProcessResult> {
  const endpoint = `${API_URL}/process`;

  console.log("Sending image to backend:", {
    endpoint,
    type: blob.type,
    size: blob.size,
  });

  let res: Response | null = null;
  let responseText = "";

  for (let attempt = 0; attempt < 2; attempt += 1) {
    const form = new FormData();
    form.append(
      "file",
      blob,
      blob.type === "image/jpeg" ? "capture.jpg" : "capture.png"
    );

    res = await fetch(endpoint, {
      method: "POST",
      body: form,
    });
    responseText = await res.text();
    if (res.ok || res.status < 500 || attempt === 1) break;
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }

  if (!res) {
    throw new Error("The processing service did not return a response");
  }

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

  if (
    !result.original ||
    !result.processed ||
    !result.overlay ||
    !result.graph ||
    !result.analysis
  ) {
    throw new Error("Backend response is missing image results");
  }

  return result as ProcessResult;
}
