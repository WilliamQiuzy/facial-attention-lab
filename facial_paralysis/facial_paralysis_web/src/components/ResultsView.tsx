import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Eye,
  FileDown,
  FileSearch,
  ScanFace,
  ShieldCheck,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { DemonstrationResult, RegionalSeverity } from '../model/demonstration'
import type { ResearchInferenceResult } from '../model/inference'
import {
  downloadResearchReportPdf,
  type PdfActionEvidence,
  type PdfMeasurement,
} from '../report/researchReportPdf'
import { RecordingDownloadButton } from './RecordingDownloadButton'

export type DisplayResult = ResearchInferenceResult | DemonstrationResult

const ACTION_LABELS: Readonly<Record<string, string>> = {
  eyebrow_raise: 'Eyebrow raise',
  gentle_eye_closure: 'Gentle eye closure',
  tight_eye_squeeze: 'Tight eye squeeze',
  relaxed_smile: 'Relaxed smile',
  lip_pucker: 'Lip pucker',
  lower_teeth_show: 'Show lower teeth',
  reanimated_smile: 'Reanimated smile',
}

const METRIC_LABELS: Readonly<Record<string, string>> = {
  brow_height_asymmetry_iod: 'Left–right brow-height difference',
  brow_height_change_from_rest_iod: 'Brow-height change from rest',
  eye_aperture_asymmetry_iod: 'Left–right eye-opening difference',
  residual_eye_aperture_iod: 'Eye opening during the hold',
  eye_closure_change_from_rest_iod: 'Eye-opening change from rest',
  mouth_corner_vertical_asymmetry_iod: 'Left–right mouth-corner height difference',
  mouth_corner_vertical_change_from_rest_iod: 'Mouth-corner movement from rest',
  mouth_corner_horizontal_asymmetry_iod: 'Left–right mouth-corner width difference',
  mouth_width_change_from_rest_iod: 'Mouth-width change from rest',
  lower_lip_change_from_rest_iod: 'Lower-lip movement from rest',
  mouth_open_change_from_rest_iod: 'Mouth-opening change from rest',
}

const REGION_LABELS = {
  brow: 'Brow region',
  eye: 'Eye region',
  mouth: 'Mouth region',
} as const

const INFLUENCE_STRENGTH_LABELS = {
  strong: 'Strong relative influence',
  moderate: 'Moderate relative influence',
  smaller: 'Smaller relative influence',
} as const

function score(value: number): string {
  return `${Math.round(value * 100)} / 100`
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function measurementPresentation(metric: string, value: number): PdfMeasurement {
  const primaryValue = `${(value * 100).toFixed(1)}% of eye-to-eye width`
  const normalizedValue = `normalized ratio ${value.toFixed(3)}`
  if (metric.includes('asymmetry')) {
    return {
      label: METRIC_LABELS[metric] ?? metric,
      kind: 'Side-to-side difference',
      primaryValue,
      normalizedValue,
      explanation: 'Smaller means the two sides were more alike in this recording.',
    }
  }
  if (metric === 'residual_eye_aperture_iod') {
    return {
      label: METRIC_LABELS[metric],
      kind: 'Position during hold',
      primaryValue,
      normalizedValue,
      explanation: 'Shows how much eye opening remained during this closure hold.',
    }
  }
  return {
    label: METRIC_LABELS[metric] ?? metric,
    kind: 'Change from neutral',
    primaryValue,
    normalizedValue,
    explanation: 'Shows the amount of geometric movement from the neutral baseline.',
  }
}

function RegionCard({ title, icon, region }: { title: string; icon: React.ReactNode; region: RegionalSeverity }) {
  const markerPosition = region.level * 50
  return (
    <article className="region-card">
      <div className="region-card-header">
        <span className="region-icon">{icon}</span>
        <div><span>Interface preview value</span><h3>{title}</h3></div>
        <strong className={`severity-chip severity-${region.level}`}>{region.label}</strong>
      </div>
      <div className="severity-scale" aria-label={`${title} ordinal level ${region.level} of 2`}>
        <span className="scale-track"><i style={{ left: `${markerPosition}%` }} /></span>
        <span className="scale-labels"><em>Normal</em><em>Slight</em><em>Strong</em></span>
      </div>
      <div className="threshold-list">
        <div><span>P(level &gt; Normal)</span><strong>{percent(region.pGt[0])}</strong><i><b style={{ width: percent(region.pGt[0]) }} /></i></div>
        <div><span>P(level &gt; Slight)</span><strong>{percent(region.pGt[1])}</strong><i><b style={{ width: percent(region.pGt[1]) }} /></i></div>
      </div>
      <p>Deterministic layout values only; no model processed this video.</p>
    </article>
  )
}

type FrameState = Readonly<Record<string, string | null>>

export function frameHasVisibleContent(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
): boolean {
  const sampleSize = Math.max(1, Math.min(16, Math.floor(width / 6), Math.floor(height / 6)))
  const positions: ReadonlyArray<readonly [number, number]> = [
    [0.1, 0.1],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.9, 0.9],
    [0.5, 0.5],
  ]
  let visible = false
  for (const [xRatio, yRatio] of positions) {
    const x = Math.round((width - sampleSize) * xRatio)
    const y = Math.round((height - sampleSize) * yRatio)
    const sample = context.getImageData(x, y, sampleSize, sampleSize).data
    for (let index = 0; index < sample.length; index += 4) {
      if (sample[index] !== 0 || sample[index + 1] !== 0 || sample[index + 2] !== 0) {
        visible = true
        break
      }
    }
  }
  return visible
}

function waitForMediaEvent(
  element: HTMLVideoElement,
  event: 'loadedmetadata' | 'durationchange' | 'seeked',
  timeoutMs = 4_000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup()
      reject(new Error(`${event} timed out`))
    }, timeoutMs)
    const onEvent = () => { cleanup(); resolve() }
    const onError = () => { cleanup(); reject(new Error('video decode failed')) }
    const cleanup = () => {
      window.clearTimeout(timeout)
      element.removeEventListener(event, onEvent)
      element.removeEventListener('error', onError)
    }
    element.addEventListener(event, onEvent, { once: true })
    element.addEventListener('error', onError, { once: true })
  })
}

function useContextFrames(
  recording: File,
  actions: ResearchInferenceResult['reportEvidence']['actions'],
): FrameState {
  const [frames, setFrames] = useState<FrameState>({})

  useEffect(() => {
    let cancelled = false
    let objectUrl: string | null = null
    const canvas = document.createElement('canvas')
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.playsInline = true

    const run = async () => {
      const failed = Object.fromEntries(actions.map((action) => [action.id, null]))
      if (navigator.userAgent.includes('jsdom')) {
        if (!cancelled) setFrames(failed)
        return
      }
      try {
        objectUrl = URL.createObjectURL(recording)
        video.src = objectUrl
        video.load()
        await waitForMediaEvent(video, 'loadedmetadata')
        if (!Number.isFinite(video.duration)) {
          const durationReady = waitForMediaEvent(video, 'durationchange')
          video.currentTime = Number.MAX_SAFE_INTEGER
          await durationReady
        }
        if (!Number.isFinite(video.duration) || video.duration <= 0 || video.videoWidth < 2 || video.videoHeight < 2) {
          throw new Error('video metadata is unusable')
        }
        const width = Math.min(video.videoWidth, 640)
        const height = Math.max(2, Math.round(video.videoHeight * width / video.videoWidth))
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d', { alpha: false })
        if (!context) throw new Error('canvas is unavailable')
        const next: Record<string, string | null> = {}
        for (const action of actions) {
          if (cancelled) return
          const target = Math.min(Math.max(action.contextFrameMs / 1_000, 0), Math.max(video.duration - 0.001, 0))
          if (Math.abs(video.currentTime - target) > 0.001) {
            const seeked = waitForMediaEvent(video, 'seeked')
            video.currentTime = target
            await seeked
          }
          context.drawImage(video, 0, 0, width, height)
          next[action.id] = frameHasVisibleContent(context, width, height)
            ? canvas.toDataURL('image/jpeg', 0.82)
            : null
          context.clearRect(0, 0, width, height)
        }
        if (!cancelled) setFrames(next)
      } catch {
        if (!cancelled) setFrames(failed)
      }
    }
    void run()
    return () => {
      cancelled = true
      if (objectUrl) {
        video.pause()
        video.removeAttribute('src')
        video.load()
      }
      canvas.width = 0
      canvas.height = 0
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [recording, actions])

  return frames
}

function DemonstrationResults({ result, onReset }: { result: DemonstrationResult; onReset: () => void }) {
  return (
    <section className="results-section" aria-labelledby="results-title">
      <div className="result-banner is-demo" role="status" aria-live="polite" aria-atomic="true">
        <AlertTriangle aria-hidden="true" size={21} />
        <div><strong>{result.provenanceLabel}</strong><span>Interface preview values only. No model processed this video.</span></div>
      </div>
      <div className="results-heading">
        <div><span className="eyebrow">Interface demonstration</span><h2 id="results-title">Movement summary</h2><p>Preview the result layout only. These values were not produced by a model.</p></div>
        <button className="button button-secondary" type="button" onClick={onReset}>Start a new session</button>
      </div>
      <div className="probability-card">
        <div className="probability-copy"><span className="region-icon"><ScanFace aria-hidden="true" size={24} /></span><div><span>Interface preview value</span><h3>Demonstration probability layout</h3><p>Synthetic display value generated from local file metadata.</p></div></div>
        <div className="probability-value"><strong>{percent(result.scores.palsyProbability)}</strong><span>not model output</span></div>
      </div>
      <div className="region-grid">
        <RegionCard title="Eye region" icon={<Eye aria-hidden="true" size={22} />} region={result.scores.eyes} />
        <RegionCard title="Mouth region" icon={<ScanFace aria-hidden="true" size={22} />} region={result.scores.mouth} />
      </div>
    </section>
  )
}

function ResearchReport({ result, recording, onBack, onReset }: {
  result: ResearchInferenceResult
  recording: File
  onBack: () => void
  onReset: () => void
}) {
  const headingRef = useRef<HTMLHeadingElement>(null)
  const frames = useContextFrames(recording, result.reportEvidence.actions)
  const cutpoint = Math.round(result.prediction.threshold * 100)
  const displayedScore = Math.round(result.prediction.probability * 100)
  const pointsFromCutpoint = Math.abs(displayedScore - cutpoint)
  const cutpointRelation = displayedScore >= cutpoint ? 'above' : 'below'
  const outputClass = result.prediction.predictedClass === 1
    ? 'Above MEEI research cutpoint'
    : 'Below MEEI research cutpoint'
  const validSamples = result.quality.actions.map((action) => action.validSamples)
  const minimumValidSamples = Math.min(...validSamples)
  const maximumValidSamples = Math.max(...validSamples)
  const framesReady = result.reportEvidence.actions.every((action) => action.id in frames)
  const [pdfState, setPdfState] = useState<'idle' | 'saving' | 'error'>('idle')

  const pdfActions: readonly PdfActionEvidence[] = result.reportEvidence.actions.map((action, index) => {
    const valid = result.quality.actions[index].validSamples
    return {
      title: ACTION_LABELS[action.id],
      region: REGION_LABELS[action.region],
      contextSeconds: `${(action.contextFrameMs / 1_000).toFixed(1)} seconds`,
      tracking: `${valid} of 32 points (${Math.round(valid / 32 * 100)}%)`,
      imageDataUrl: frames[action.id] ?? null,
      measurements: action.observations.map((observation) =>
        measurementPresentation(observation.metric, observation.value)),
      influence: action.modelInfluence.status === 'stable' ? {
        status: 'stable',
        strength: INFLUENCE_STRENGTH_LABELS[action.modelInfluence.strength],
        direction: `This ${REGION_LABELS[action.region].toLowerCase()} action signal moved the MEEI facial-movement score ${action.modelInfluence.direction === 'toward_class_1' ? 'upward' : 'downward'}.`,
        relative: `${Math.round(action.modelInfluence.relativeMagnitude * 100)}% relative to the strongest action-level influence magnitude in this recording.`,
      } : {
        status: 'unavailable',
        explanation: 'The direction is hidden because the consistency checks did not all agree.',
      },
      stability: [
        `${action.stability.ensembleSignAgreement} of 3 model members`,
        `Mirror view: ${action.stability.mirrorConsistent ? 'passed' : 'did not agree'}`,
        `${action.stability.temporalChecksPassed} of 2 timing shifts`,
      ],
    }
  })

  const savePdf = async () => {
    if (!framesReady || pdfState === 'saving') return
    setPdfState('saving')
    try {
      await downloadResearchReportPdf({
        score: score(result.prediction.probability),
        scoreMeaning: 'This is the average of three model outputs measuring similarity to the MEEI development groups. Higher values indicate more similarity to the MEEI facial-palsy examples; lower values indicate more similarity to the MEEI healthy-control examples.',
        outputClass,
        cutpointSummary: `${pointsFromCutpoint} points ${cutpointRelation} the fixed cutpoint of ${cutpoint}.`,
        recordingCoverage: [
          `Neutral baseline plus all ${result.quality.actionsUsed} active movements were included.`,
          `Face tracking ranged from ${minimumValidSamples} to ${maximumValidSamples} usable points out of 32 per movement.`,
          `The optional reanimation smile was ${result.quality.optionalActionsUnavailable.length ? 'not part of this session' : 'included'}.`,
        ],
        actions: pdfActions,
        clinicalReviewNote: 'Review the movement score together with the recorded action images and source video in the context of the clinical assessment.',
      })
      setPdfState('idle')
    } catch {
      setPdfState('error')
    }
  }

  useEffect(() => { headingRef.current?.focus() }, [])

  return (
    <article className="research-report" aria-labelledby="research-report-title">
      <nav className="report-toolbar" aria-label="Research report actions">
        <a className="report-back-link" href="#analysis" onClick={(event) => { event.preventDefault(); onBack() }}><ArrowLeft aria-hidden="true" size={17} /> Back to session summary</a>
        <div className="report-toolbar-actions">
          <div className="report-save-control report-action-control">
            <button className="button button-primary report-save-button" type="button" disabled={!framesReady || pdfState === 'saving'} onClick={() => { void savePdf() }}><FileDown aria-hidden="true" size={17} /> {pdfState === 'saving' ? 'Creating PDF…' : 'Save PDF'}</button>
          </div>
          <RecordingDownloadButton recording={recording} compact />
          <div className="report-action-control">
            <button className="button button-secondary" type="button" onClick={onReset}>Start a new session</button>
          </div>
          <p className="report-actions-note" role={pdfState === 'error' ? 'alert' : undefined}>{pdfState === 'error' ? 'PDF creation failed. Please try again.' : framesReady ? 'PDF includes the recorded evidence images · Video download saves the identifiable source · New session clears the in-browser copy' : 'Preparing recorded context images for the PDF…'}</p>
        </div>
      </nav>

      <header className="report-header">
        <div><h1 id="research-report-title" ref={headingRef} tabIndex={-1}>Research Movement Report</h1><p>Facial movement classification and recorded action evidence.</p></div>
      </header>

      <section className="report-score-section" aria-labelledby="score-title">
        <div className="report-score-card"><span className="region-icon"><BarChart3 aria-hidden="true" size={25} /></span><div><span>MEEI facial-movement classification score</span><strong>{score(result.prediction.probability)}</strong><p>{pointsFromCutpoint} points {cutpointRelation} the fixed cutpoint of {cutpoint}.</p></div></div>
        <div className="score-explanation">
          <h2 id="score-title">What this number represents</h2>
          <p>This is the average of three model outputs measuring similarity to the MEEI development groups. Higher values indicate more similarity to the MEEI facial-palsy examples; lower values indicate more similarity to the MEEI healthy-control examples.</p>
          <div className="classification-summary"><span>Current output</span><strong>{outputClass}</strong><small>{pointsFromCutpoint} points {cutpointRelation} the cutpoint</small></div>
        </div>
      </section>

      <section className="report-section" aria-labelledby="evidence-title">
        <div className="section-heading"><span className="region-icon"><FileSearch aria-hidden="true" size={22} /></span><div><h2 id="evidence-title">Recorded action evidence</h2><p>Each context image is taken at the registered midpoint of its three-second hold. It is recorded context, not a frame selected by the model.</p></div></div>
        <p className="measurement-boundary"><ShieldCheck aria-hidden="true" size={20} /><strong>Three separate evidence layers: what was measured, how each action moved the model score, and whether that influence was stable.</strong></p>
        <p className="measurement-unit-note">These observations are calculated from MediaPipe 478-point facial landmarks sampled during each registered hold and compared with the neutral baseline. Measurements are scaled to the same eye-to-eye reference width: 1.0% corresponds to a normalized ratio of 0.010.</p>
        <div className="evidence-legend evidence-layer-legend" aria-label="Evidence layer guide"><div><strong><span>1</span> Measured movement</strong><span>Geometry observed during the registered hold.</span></div><div><strong><span>2</span> Model influence</strong><span>Direction and relative strength at the shared action-token layer.</span></div><div><strong><span>3</span> Stability checks</strong><span>Agreement across model members, mirrored input, and timing shifts.</span></div></div>
        <div className="evidence-grid">{result.reportEvidence.actions.map((action, actionIndex) => (
          <article className="evidence-card" key={action.id}>
            <div className="evidence-frame">{frames[action.id] ? <img src={frames[action.id] ?? undefined} alt={`${ACTION_LABELS[action.id]} recorded context at ${(action.contextFrameMs / 1_000).toFixed(1)} seconds`} width="640" height="480" loading="lazy" decoding="async" /> : <div className="frame-fallback" role="img" aria-label={`${ACTION_LABELS[action.id]} context frame unavailable`}><ScanFace aria-hidden="true" size={30} /><span>Recorded context frame unavailable</span></div>}<span>{(action.contextFrameMs / 1_000).toFixed(1)} s</span></div>
            <div className="evidence-copy">
              <div className="evidence-action-heading"><div><h3>{ACTION_LABELS[action.id]}</h3><p>{REGION_LABELS[action.region]}</p></div><span>Action tracking</span><strong>{result.quality.actions[actionIndex].validSamples} of 32 points ({Math.round(result.quality.actions[actionIndex].validSamples / 32 * 100)}%)</strong></div>
              <section className="evidence-layer evidence-layer-measurement" aria-label={`${ACTION_LABELS[action.id]} measured movement`}>
                <h4><span>1</span> Measured movement</h4>
                <dl>{action.observations.map((observation) => { const presentation = measurementPresentation(observation.metric, observation.value); return <div key={observation.metric}><dt><span>{presentation.kind}</span>{presentation.label}</dt><dd>{presentation.primaryValue}<span>{presentation.normalizedValue}</span><small>{presentation.explanation}</small></dd></div> })}</dl>
              </section>
              <section className="evidence-layer evidence-layer-influence" aria-label={`${ACTION_LABELS[action.id]} model influence`}>
                <h4><span>2</span> Model influence</h4>
                {action.modelInfluence.status === 'stable' ? (
                  <div className="influence-result">
                    <strong>{INFLUENCE_STRENGTH_LABELS[action.modelInfluence.strength]}</strong>
                    <p>This {REGION_LABELS[action.region].toLowerCase()} action signal moved the MEEI facial-movement score {action.modelInfluence.direction === 'toward_class_1' ? 'upward' : 'downward'}.</p>
                    <div className="influence-meter" role="img" aria-label={`${Math.round(action.modelInfluence.relativeMagnitude * 100)} percent of the strongest action-level influence magnitude in this recording`}><span style={{ width: `${Math.max(4, action.modelInfluence.relativeMagnitude * 100)}%` }} /></div>
                    <small>{Math.round(action.modelInfluence.relativeMagnitude * 100)}% relative to the strongest action-level influence magnitude in this recording.</small>
                  </div>
                ) : (
                  <div className="influence-unavailable"><strong>No stable model influence to report for this action.</strong><p>The direction is hidden because the consistency checks did not all agree.</p></div>
                )}
              </section>
              <section className="evidence-layer evidence-layer-stability" aria-label={`${ACTION_LABELS[action.id]} stability checks`}>
                <h4><span>3</span> Stability checks</h4>
                <div className="stability-checks"><span className={action.stability.ensembleSignAgreement === 3 ? 'passed' : 'not-passed'}>{action.stability.ensembleSignAgreement} of 3 model members</span><span className={action.stability.mirrorConsistent ? 'passed' : 'not-passed'}>Mirror view {action.stability.mirrorConsistent ? 'passed' : 'did not agree'}</span><span className={action.stability.temporalChecksPassed === 2 ? 'passed' : 'not-passed'}>{action.stability.temporalChecksPassed} of 2 timing shifts</span></div>
              </section>
            </div>
          </article>
        ))}</div>
      </section>

      <section className="report-section compact recording-coverage" aria-labelledby="quality-title">
        <h2 id="quality-title">Recording coverage</h2>
        <dl className="report-definition-list"><div><dt>Recorded steps included in this score</dt><dd>Neutral baseline + all {result.quality.actionsUsed} active movements</dd></div><div><dt>Face tracking coverage</dt><dd>{minimumValidSamples}–{maximumValidSamples} usable of 32 checkpoints per movement</dd></div><div><dt>Optional reanimation smile</dt><dd>{result.quality.optionalActionsUnavailable.length ? 'Not part of this session' : 'Included'}</dd></div></dl>
        <div className="coverage-explanation"><p><ShieldCheck aria-hidden="true" size={19} />All {result.quality.actionsUsed + 1} recorded steps in this session were used: one neutral baseline and {result.quality.actionsUsed} active movements.</p><p>Each active movement is checked at 32 evenly spaced time points; the range above shows how many had usable face tracking. The neutral recording provides the resting baseline used for movement-change measurements.</p></div>
      </section>

      <section className="report-clinical-note" aria-labelledby="clinical-review-title"><ShieldCheck aria-hidden="true" size={28} strokeWidth={1.8} /><div><h2 id="clinical-review-title">Clinical review note</h2><p>Review the movement score together with the recorded action images and source video in the context of the clinical assessment.</p></div></section>
    </article>
  )
}

export function ResultsView({ result, recording, onReset, onBack = () => undefined }: {
  result: DisplayResult
  recording?: File
  onReset: () => void
  onBack?: () => void
}) {
  if (result.mode === 'demonstration') return <DemonstrationResults result={result} onReset={onReset} />
  if (!recording) throw new Error('Research report requires the in-memory browser recording')
  return <ResearchReport result={result} recording={recording} onBack={onBack} onReset={onReset} />
}
