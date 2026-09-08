import { useCallback, useEffect, useRef, useState } from 'react'

import { RecordingDownloadButton } from './RecordingDownloadButton'

export type RequestRecordingChange = (action: () => void, purpose: string) => void

export function useRecordingGuard(recording: File | null) {
  const [pending, setPending] = useState<{ action: () => void; purpose: string; recording: File } | null>(null)
  const pendingRef = useRef(false)
  const dialogRef = useRef<HTMLDialogElement>(null)
  const keepRef = useRef<HTMLButtonElement>(null)

  const requestChange: RequestRecordingChange = useCallback((action, purpose) => {
    if (pendingRef.current) return
    if (!recording) { action(); return }
    pendingRef.current = true
    setPending({ action, purpose, recording })
  }, [recording])

  const dismiss = () => { pendingRef.current = false; setPending(null) }

  useEffect(() => {
    if (!pending) return
    const previous = document.activeElement as HTMLElement | null
    const dialog = dialogRef.current
    if (dialog?.showModal) dialog.showModal()
    else dialog?.setAttribute('open', '')
    keepRef.current?.focus()
    return () => {
      dialog?.close?.()
      if (previous?.isConnected) previous.focus()
    }
  }, [pending])

  return {
    requestChange,
    dialog: pending ? (
      <dialog ref={dialogRef} className="recording-guard" aria-labelledby="recording-guard-title" aria-describedby="recording-guard-description" onCancel={(event) => { event.preventDefault(); dismiss() }}>
        <h2 id="recording-guard-title">{pending.purpose}</h2>
        <p id="recording-guard-description">You already have a complete recording. Continuing removes it and any current report from this session. You can download the video first.</p>
        <div className="recording-guard-actions">
          <button ref={keepRef} className="button button-primary" type="button" onClick={dismiss}>Keep current recording</button>
          <RecordingDownloadButton recording={pending.recording} />
          <button className="button button-secondary" type="button" onClick={() => { const action = pending.action; dismiss(); action() }}>Discard and continue</button>
        </div>
      </dialog>
    ) : null,
  }
}
