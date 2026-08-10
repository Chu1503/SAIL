"use client";

type Props = {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
};

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Delete",
  onConfirm,
  onCancel,
}: Props) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 px-6"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-xs rounded-2xl border border-white/10 bg-neutral-950 p-5"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 className="text-sm font-semibold text-white">{title}</h2>
        <p className="mt-2 text-xs leading-5 text-neutral-400">{message}</p>

        <div className="mt-5 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-full border border-white/15 px-4 py-2 text-xs font-medium text-neutral-300 transition active:scale-95"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="rounded-full bg-red-500 px-4 py-2 text-xs font-semibold text-white transition active:scale-95"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
