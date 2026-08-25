import { Download } from 'lucide-react'

function safeRecordingFilename(recording: File): string {
  const mimeType = recording.type.toLowerCase()
  if (mimeType.startsWith('video/webm')) return 'faces-research-recording.webm'
  if (mimeType.startsWith('video/mp4')) return 'faces-research-recording.mp4'
  if (mimeType.startsWith('video/quicktime')) return 'faces-research-recording.mov'
  return 'faces-research-recording.video'
}

function downloadRecording(recording: File): void {
  const objectUrl = URL.createObjectURL(recording)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = safeRecordingFilename(recording)
  anchor.rel = 'noopener'
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
}

export function RecordingDownloadButton({ recording, compact = false }: {
  recording: File
  compact?: boolean
}) {
  return (
    <div className={`recording-download-control${compact ? ' is-compact report-action-control' : ''}`}>
      <button
        className={`button button-secondary${compact ? '' : ' button-wide'}`}
        type="button"
        onClick={() => downloadRecording(recording)}
      >
        <Download aria-hidden="true" size={17} /> Download recorded video
      </button>
      {!compact ? <span>Saves the identifiable source video to this device.</span> : null}
    </div>
  )
}
