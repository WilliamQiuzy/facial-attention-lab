import { CircleCheck, Focus, Info, Sun } from 'lucide-react'
import type { RefObject } from 'react'

import type { PreflightHint } from '../camera/preflight'
import { useCameraPreflight } from '../hooks/useCameraPreflight'
import './CameraPreflight.css'

const messages: Record<PreflightHint, { title: string; detail: string }> = {
  checking: { title: 'Checking camera framing', detail: 'Optional preview hints are loading. There is no need to wait to start.' },
  unavailable: { title: 'Camera hints unavailable', detail: 'Use the preview to check that your whole face is visible and evenly lit.' },
  'no-face': { title: 'Bring your face into view', detail: 'Try facing the camera with your whole face visible in the preview.' },
  'one-face': { title: 'Keep one face in view', detail: 'If possible, have only the person recording in the camera preview.' },
  center: { title: 'Center your face', detail: 'Keep your whole face near the middle of the camera preview.' },
  closer: { title: 'Come a little closer', detail: 'If comfortable, move closer so your face fills more of the preview.' },
  light: { title: 'The lighting looks dim', detail: 'Try facing a soft light so your face is easier to see.' },
  ready: { title: 'Framing looks good', detail: 'Your face is in view. Start whenever you are comfortable.' },
}

export function CameraPreflight({ videoRef, active }: {
  videoRef: RefObject<HTMLVideoElement | null>
  active: boolean
}) {
  const hint = useCameraPreflight(videoRef, active)
  if (!active) return null
  const message = messages[hint]
  const Icon = hint === 'light' ? Sun : hint === 'ready' ? CircleCheck
    : hint === 'checking' || hint === 'unavailable' ? Info : Focus
  return (
    <aside className="camera-preflight" data-hint={hint} aria-label="Optional camera framing guidance">
      <div className="camera-preflight__message" role="status" aria-live="polite" aria-atomic="true">
        <Icon className="camera-preflight__icon" size={22} aria-hidden="true" />
        <div>
          <strong>{message.title}</strong>
          <p>{message.detail}</p>
        </div>
      </div>
      <small>Optional, on-device framing only · you can start anytime.</small>
    </aside>
  )
}
