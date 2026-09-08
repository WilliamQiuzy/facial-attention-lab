import {
  Camera,
  CircleStop,
  FileVideo2,
  RefreshCw,
  UploadCloud,
  Video,
} from 'lucide-react'
import { type KeyboardEvent, useEffect, useId, useRef, useState } from 'react'

import { type CameraRecorderState, useCameraRecorder } from '../hooks/useCameraRecorder'
import {
  MAX_VIDEO_BYTES,
  parseCaptureTimelineSidecar,
  type CaptureTimelineDraft,
  type RecordingSource,
} from '../model/inference'
import type { FacesActionId } from '../protocol/facesProtocol'
import { useRecordingGuard, type RequestRecordingChange } from './RecordingGuard'
import { CameraPreflight } from './CameraPreflight'

const SUPPORTED_EXTENSIONS = ['.mov', '.mp4', '.avi', '.m4v', '.webm']

export type CaptureMode = 'upload' | 'camera'

export interface RecordingChangeOptions {
  readonly preserveProtocolChoice?: boolean
  readonly captureId?: string
  readonly actionIds?: readonly FacesActionId[]
  readonly reanimatedSmileApplicable?: boolean
  readonly timeline?: CaptureTimelineDraft
}

export type RecordingChangeHandler = (
  file: File | null,
  source: RecordingSource,
  options?: RecordingChangeOptions,
) => void

interface MediaCaptureProps {
  readonly onRecordingChange: RecordingChangeHandler
}

interface MediaCapturePanelProps extends MediaCaptureProps {
  readonly camera: CameraRecorderState
  readonly mode: CaptureMode
  readonly onModeChange: (mode: CaptureMode) => void
  readonly recordingControls?: 'manual' | 'guided'
  readonly guidedActive?: boolean
  readonly preserveProtocolChoiceOnCameraRecording?: boolean
  readonly reportCameraRecording?: boolean
  readonly showCameraError?: boolean
  readonly onRequestChange?: RequestRecordingChange
  readonly hideEnableCamera?: boolean
  readonly previewVisible?: boolean
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024 * 1_024) return `${Math.max(1, Math.round(bytes / 1_024))} KB`
  return `${(bytes / (1_024 * 1_024)).toFixed(1)} MB`
}

function isSupportedVideo(file: File): boolean {
  const lowerName = file.name.toLowerCase()
  const supportedExtension = SUPPORTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
  return supportedExtension && (file.type === '' || file.type.startsWith('video/'))
}

export function MediaCapture({ onRecordingChange }: MediaCaptureProps) {
  const [mode, setMode] = useState<CaptureMode>('camera')
  const camera = useCameraRecorder()
  return (
    <MediaCapturePanel
      camera={camera}
      mode={mode}
      onModeChange={setMode}
      onRecordingChange={onRecordingChange}
    />
  )
}

export function MediaCapturePanel({
  camera,
  mode,
  onModeChange,
  onRecordingChange,
  recordingControls = 'manual',
  guidedActive = false,
  preserveProtocolChoiceOnCameraRecording = false,
  reportCameraRecording = true,
  showCameraError = true,
  onRequestChange,
  hideEnableCamera = false,
  previewVisible = true,
}: MediaCapturePanelProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadedTimeline, setUploadedTimeline] = useState<CaptureTimelineDraft | null>(null)
  const [uploadedTimelineName, setUploadedTimelineName] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const inputId = useId()
  const timelineInputId = useId()
  const sourceId = useId()
  const uploadTabRef = useRef<HTMLButtonElement | null>(null)
  const cameraTabRef = useRef<HTMLButtonElement | null>(null)
  const timelineGenerationRef = useRef(0)
  useEffect(() => () => { timelineGenerationRef.current += 1 }, [])
  const uploadTabId = `${sourceId}-upload-tab`
  const cameraTabId = `${sourceId}-camera-tab`
  const uploadPanelId = `${sourceId}-upload-panel`
  const cameraPanelId = `${sourceId}-camera-panel`
  const localGuard = useRecordingGuard(mode === 'upload' ? uploadedFile : camera.recordingFile)
  const requestChange = onRequestChange ?? localGuard.requestChange

  useEffect(() => {
    const file = mode === 'upload' ? uploadedFile : camera.recordingFile
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [camera.recordingFile, mode, uploadedFile])

  useEffect(() => {
    if (reportCameraRecording && mode === 'camera' && camera.recordingFile) {
      if (preserveProtocolChoiceOnCameraRecording) {
        onRecordingChange(camera.recordingFile, 'browser-camera', { preserveProtocolChoice: true })
      } else {
        onRecordingChange(camera.recordingFile, 'browser-camera')
      }
    }
  }, [camera.recordingFile, mode, onRecordingChange, preserveProtocolChoiceOnCameraRecording, reportCameraRecording])

  const switchMode = (nextMode: CaptureMode) => {
    if (nextMode === mode || guidedActive) return
    requestChange(() => applyMode(nextMode), 'Change recording source?')
  }

  const applyMode = (nextMode: CaptureMode) => {
    timelineGenerationRef.current += 1
    if (nextMode === 'upload') {
      camera.closeCamera()
      camera.resetRecording()
    }
    onModeChange(nextMode)
    setUploadError(null)
    setUploadedFile(null)
    setUploadedTimeline(null)
    setUploadedTimelineName(null)
    onRecordingChange(
      null,
      nextMode === 'upload' ? 'livelink-upload' : 'browser-camera',
      { preserveProtocolChoice: true },
    )
  }

  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentMode: CaptureMode,
  ) => {
    if (guidedActive) return
    let nextMode: CaptureMode | null = null
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      nextMode = currentMode === 'upload' ? 'camera' : 'upload'
    } else if (event.key === 'Home') {
      nextMode = 'camera'
    } else if (event.key === 'End') {
      nextMode = 'upload'
    }
    if (!nextMode) return
    event.preventDefault()
    switchMode(nextMode)
    const target = nextMode === 'upload' ? uploadTabRef : cameraTabRef
    target.current?.focus()
  }

  const selectFile = (file: File | undefined) => {
    if (!file) return
    if (!isSupportedVideo(file)) {
      setUploadError('Choose a supported video file: MOV, MP4, M4V, AVI, or WebM.')
      if (!uploadedFile) onRecordingChange(null, 'livelink-upload', { preserveProtocolChoice: true })
      return
    }
    if (file.size < 1 || file.size > MAX_VIDEO_BYTES) {
      setUploadError(
        file.size < 1
          ? 'The selected video is empty. Choose a complete FACES recording.'
          : 'The selected video is larger than 512 MB. Choose an approved compressed copy.',
      )
      if (!uploadedFile) onRecordingChange(null, 'livelink-upload', { preserveProtocolChoice: true })
      return
    }
    requestChange(() => applyFile(file), 'Replace the current recording?')
  }

  const applyFile = (file: File) => {
    timelineGenerationRef.current += 1
    setUploadError(null)
    setUploadedFile(file)
    setUploadedTimeline(null)
    setUploadedTimelineName(null)
    onRecordingChange(file, 'livelink-upload')
  }

  const selectTimeline = async (file: File | undefined) => {
    if (!file || !uploadedFile) return
    const generation = ++timelineGenerationRef.current
    if (file.size < 1 || file.size > 256 * 1024 || !file.name.toLowerCase().endsWith('.json')) {
      setUploadError('Choose a bounded JSON FACES action timeline.')
      return
    }
    try {
      const source = typeof file.text === 'function'
        ? await file.text()
        : await new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onerror = () => reject(new Error('Timeline could not be read.'))
            reader.onload = () => resolve(String(reader.result ?? ''))
            reader.readAsText(file)
          })
      if (generation !== timelineGenerationRef.current) return
      const timeline = parseCaptureTimelineSidecar(source)
      setUploadedTimeline(timeline)
      setUploadedTimelineName(file.name)
      setUploadError(null)
      onRecordingChange(uploadedFile, 'livelink-upload', {
        preserveProtocolChoice: true,
        reanimatedSmileApplicable: timeline.actions.length === 8,
        timeline,
      })
    } catch (error) {
      if (generation !== timelineGenerationRef.current) return
      setUploadError(error instanceof Error ? error.message : 'Timeline was not accepted.')
    }
  }

  const resetCameraRecording = () => {
    requestChange(() => {
      camera.resetRecording()
      onRecordingChange(null, 'browser-camera', { preserveProtocolChoice: true })
    }, 'Record again?')
  }

  return (
    <section className="capture-card" aria-labelledby="capture-title">
      <div className="section-heading-row">
        <div>
          <h2 id="capture-title">{mode === 'camera' ? 'Camera preview' : 'Upload a recorded session'}</h2>
        </div>
        <span className="session-badge">Session only</span>
      </div>

      <div className="source-tabs" role="tablist" aria-label="Recording source">
        <button
          ref={cameraTabRef}
          id={cameraTabId}
          type="button"
          role="tab"
          aria-selected={mode === 'camera'}
          aria-controls={cameraPanelId}
          tabIndex={mode === 'camera' ? 0 : -1}
          disabled={guidedActive}
          className={mode === 'camera' ? 'source-tab is-selected' : 'source-tab'}
          onClick={() => switchMode('camera')}
          onKeyDown={(event) => handleTabKeyDown(event, 'camera')}
        >
          <Camera aria-hidden="true" size={19} /> Use this device
        </button>
        <button
          ref={uploadTabRef}
          id={uploadTabId}
          type="button"
          role="tab"
          aria-selected={mode === 'upload'}
          aria-controls={uploadPanelId}
          tabIndex={mode === 'upload' ? 0 : -1}
          disabled={guidedActive}
          className={mode === 'upload' ? 'source-tab is-selected' : 'source-tab'}
          onClick={() => switchMode('upload')}
          onKeyDown={(event) => handleTabKeyDown(event, 'upload')}
        >
          <UploadCloud aria-hidden="true" size={19} /> Upload from LifeLink
        </button>
      </div>

      {mode === 'upload' ? (
        <div className="upload-panel" role="tabpanel" id={uploadPanelId} aria-labelledby={uploadTabId}>
          <button className="button button-secondary" type="button" onClick={() => switchMode('camera')}>
            <Camera aria-hidden="true" size={18} /> Return to live camera
          </button>
          <input
            className="visually-hidden"
            id={inputId}
            type="file"
            accept="video/quicktime,video/mp4,video/x-msvideo,video/webm,.mov,.mp4,.m4v,.avi,.webm"
            aria-label="Choose LifeLink Face video"
            onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; selectFile(file) }}
          />
          {!uploadedFile ? (
            <label className="drop-zone" htmlFor={inputId}>
              <span className="upload-icon"><FileVideo2 aria-hidden="true" size={28} /></span>
              <strong>Choose a LifeLink Face recording</strong>
              <span>MOV, MP4, M4V, AVI or WebM · up to 512 MB</span>
              <span>You need the video and its matching FACES action timeline (.json). The timeline identifies when each movement was recorded.</span>
              <span className="button button-primary button-as-label">Browse video</span>
            </label>
          ) : (
            <div className="selected-file">
              <div className="file-preview">
                {previewUrl ? <video src={previewUrl} controls preload="metadata" aria-label="Selected recording preview" /> : null}
              </div>
              <div className="file-details">
                <span className="ready-mark"><FileVideo2 aria-hidden="true" size={18} /> Ready for protocol review</span>
                <strong>{uploadedFile.name}</strong>
                <span>{formatBytes(uploadedFile.size)} · {uploadedFile.type || 'video file'}</span>
                <label className="text-action" htmlFor={inputId}>Replace video</label>
              </div>
            </div>
          )}
          {uploadError ? <p className="inline-alert" role="alert">{uploadError}</p> : null}
          {uploadedFile ? <div className="timeline-upload">
            <input
              className="visually-hidden"
              id={timelineInputId}
              type="file"
              accept="application/json,.json"
              aria-label="Choose FACES action timeline"
              onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void selectTimeline(file) }}
            />
            <label className="button button-secondary" htmlFor={timelineInputId}>
              {uploadedTimeline ? 'Replace action timeline' : 'Add action timeline'}
            </label>
            <span>{uploadedTimelineName ?? 'Required for guided analysis'}</span>
          </div> : null}
        </div>
      ) : (
        <div className="camera-panel" role="tabpanel" id={cameraPanelId} aria-labelledby={cameraTabId}>
          <div className="camera-stage">
            {camera.recordingFile && previewUrl ? (
              <video
                src={previewUrl}
                controls
                playsInline
                preload="metadata"
                aria-label="Recorded camera preview"
              />
            ) : (
              <video ref={camera.videoRef} muted playsInline aria-label="Live front camera preview" />
            )}
            {camera.status === 'idle' || camera.status === 'requesting' || camera.status === 'error' ? (
              <div className="camera-placeholder">
                <Video aria-hidden="true" size={30} />
                <strong>Front camera preview</strong>
                <span>Face and neck should remain visible at eye level.</span>
              </div>
            ) : null}
            {camera.status === 'recording' ? <span className="recording-indicator"><i /> Recording</span> : null}
          </div>
          <CameraPreflight videoRef={camera.videoRef} active={previewVisible && camera.status === 'ready' && !guidedActive} />
          <div className="camera-actions">
            {!hideEnableCamera && (camera.status === 'idle' || camera.status === 'error') ? (
              <button className="button button-primary" type="button" onClick={camera.enableCamera}>
                <Camera aria-hidden="true" size={18} /> Enable camera
              </button>
            ) : null}
            {camera.status === 'ready' && recordingControls === 'manual' ? (
              <button className="button button-primary" type="button" onClick={camera.startRecording}>
                <span className="record-dot" /> Start protocol recording
              </button>
            ) : null}
            {camera.status === 'recording' && recordingControls === 'manual' ? (
              <button className="button button-danger" type="button" onClick={camera.stopRecording}>
                <CircleStop aria-hidden="true" size={18} /> Stop recording
              </button>
            ) : null}
            {recordingControls === 'guided' && camera.status === 'ready' ? (
              <p className="camera-control-note">Camera ready. Follow the next action at the bottom of the page.</p>
            ) : null}
            {recordingControls === 'guided' && guidedActive && (camera.status === 'starting' || camera.status === 'recording') ? (
              <p className="camera-control-note">Camera and voice guidance are linked</p>
            ) : null}
            {camera.status === 'recorded' ? (
              <button className="button button-secondary" type="button" onClick={resetCameraRecording}>
                <RefreshCw aria-hidden="true" size={18} /> Record again
              </button>
            ) : null}
          </div>
          {showCameraError && camera.error ? <p className="inline-alert" role="alert">{camera.error}</p> : null}
          {camera.recordingFile ? (
            <p className="camera-ready"><FileVideo2 aria-hidden="true" size={18} /> {camera.recordingFile.name} is ready.</p>
          ) : null}
        </div>
      )}

      <p className="privacy-footnote">
        Recording bytes are kept in this browser session only until you refresh or close the page.
      </p>
      {localGuard.dialog}
    </section>
  )
}
