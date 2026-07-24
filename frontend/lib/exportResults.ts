import { Capacitor } from "@capacitor/core";
import { Directory, Filesystem } from "@capacitor/filesystem";
import JSZip from "jszip";
import type { ProcessResult } from "@/lib/api";

export type ScanExport = {
  capturedAt: string;
  result: ProcessResult;
};

export type ExportReceipt = {
  batchName: string;
  fileCount: number;
  location: string;
};

const EXPORT_ROOT = "VEINZ_Exports";

const imageFiles: Array<{
  name: string;
  getData: (result: ProcessResult) => string;
}> = [
  { name: "01_Post_Processed.png", getData: (result) => result.processed },
  { name: "02_Vein_Overlay.png", getData: (result) => result.overlay },
  { name: "03_Vein_Mask.png", getData: (result) => result.mask },
];

function compactTimestamp(value: string | Date) {
  const date = typeof value === "string" ? new Date(value) : value;
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function scanFolder(index: number, capturedAt: string) {
  return `Scan_${String(index + 1).padStart(3, "0")}_${compactTimestamp(capturedAt)}`;
}

function base64Payload(dataUrl: string) {
  const separator = dataUrl.indexOf(",");
  if (separator === -1) {
    throw new Error("An exported image has an invalid data URL");
  }
  return dataUrl.slice(separator + 1);
}

async function dataUrlToBlob(dataUrl: string) {
  const response = await fetch(dataUrl);
  if (!response.ok) {
    throw new Error("Could not prepare an image for export");
  }
  return response.blob();
}

async function saveNative(scans: ScanExport[], batchName: string) {
  const batchPath = `${EXPORT_ROOT}/${batchName}`;

  for (const [index, scan] of scans.entries()) {
    const folder = `${batchPath}/${scanFolder(index, scan.capturedAt)}`;

    for (const image of imageFiles) {
      await Filesystem.writeFile({
        path: `${folder}/${image.name}`,
        data: base64Payload(image.getData(scan.result)),
        directory: Directory.Documents,
        recursive: true,
      });
    }
  }

  return `Documents/${batchPath}`;
}

async function saveWeb(scans: ScanExport[], batchName: string) {
  const zip = new JSZip();
  const batch = zip.folder(batchName);

  if (!batch) {
    throw new Error("Could not create the export archive");
  }

  for (const [index, scan] of scans.entries()) {
    const folder = batch.folder(scanFolder(index, scan.capturedAt));
    if (!folder) continue;

    await Promise.all(
      imageFiles.map(async (image) => {
        folder.file(image.name, await dataUrlToBlob(image.getData(scan.result)));
      })
    );
  }

  const archive = await zip.generateAsync({
    type: "blob",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  const url = URL.createObjectURL(archive);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${batchName}.zip`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);

  return `Downloads/${batchName}.zip`;
}

export async function exportScans(scans: ScanExport[]): Promise<ExportReceipt> {
  if (scans.length === 0) {
    throw new Error("There are no scan images to save");
  }

  const batchName = `VEINZ_Export_${compactTimestamp(new Date())}`;
  const location = Capacitor.isNativePlatform()
    ? await saveNative(scans, batchName)
    : await saveWeb(scans, batchName);

  return {
    batchName,
    fileCount: scans.length * imageFiles.length,
    location,
  };
}
