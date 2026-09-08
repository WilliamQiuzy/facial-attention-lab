import { Download } from 'lucide-react'
import { useState } from 'react'

import { getSessionFilename } from '../report/sessionIdentity'

function downloadRecording(recording: File): void {
  const objectUrl = URL.createObjectURL(recording)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = getSessionFilename(recording, 'recording')
  anchor.rel = 'noopener'
  anchor.hidden = true
  try {
    document.body.append(anchor)
    anchor.click()
  } finally {
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
  }
}

export function RecordingDownloadButton({ recording, compact = false }: {
  recording: File
  compact?: boolean
}) {
  const [downloadState, setDownloadState] = useState<'idle' | 'requested' | 'error'>('idle')
  const requestDownload = () => {
    try {
      downloadRecording(recording)
      setDownloadState('requested')
    } catch {
      setDownloadState('error')
    }
  }
  return (
    <div className={`recording-download-control${compact ? ' is-compact report-action-control' : ''}`}>
      <button
        className={`button button-secondary${compact ? '' : ' button-wide'}`}
        type="button"
        onClick={requestDownload}
      >
        <Download aria-hidden="true" size={17} /> Download recorded video
      </button>
      {!compact ? <span>Downloads the identifiable source video to this device.</span> : null}
      {downloadState !== 'idle' ? <span className="download-feedback" role={downloadState === 'error' ? 'alert' : 'status'}>{downloadState === 'error' ? 'Download could not start. Please try again.' : 'Download started; check browser downloads.'}</span> : null}
    </div>
  )
}
