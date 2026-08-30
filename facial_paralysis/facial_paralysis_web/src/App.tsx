import { ArrowLeft, ArrowRight, Camera, Check, LockKeyhole, ShieldCheck, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { AppHeader } from './components/AppHeader'
import { GuidedCaptureWorkspace } from './components/GuidedCaptureWorkspace'
import type { RecordingChangeOptions } from './components/MediaCapture'
import { RecordingDownloadButton } from './components/RecordingDownloadButton'
import { ResultsView, type DisplayResult } from './components/ResultsView'
import { WorkflowRail } from './components/WorkflowRail'
import { createDemonstrationResult } from './model/demonstration'
import {
  analyzeRecording,
  checkResearchEndpoint,
  InferenceContractError,
  type CaptureTimelineDraft,
  type RecordingSource,
  type ResearchInferenceResult,
} from './model/inference'
import { FACES_PREPARATION } from './protocol/facesProtocol'
import './styles/app.css'

type AnalyzeFunction = typeof analyzeRecording
type EndpointCheckFunction = typeof checkResearchEndpoint

interface AppProps {
  readonly apiEndpoint?: string
  readonly demonstrationEnabled?: boolean
  readonly analyze?: AnalyzeFunction
  readonly checkEndpoint?: EndpointCheckFunction
}

type JourneyStep = 1 | 2 | 3 | 4 | 5

function capturePreparation() {
  return [
    'Front camera at eye level and about arm’s length away',
    'Entire face and neck visible in a well-lit room',
    'Neutral background with no bright backlight',
    'Head and body still throughout all movements',
  ]
}

export function App({
  apiEndpoint = import.meta.env.VITE_FACIAL_PARALYSIS_API_URL ?? '/api/v1/facial-paralysis/infer',
  demonstrationEnabled = import.meta.env.VITE_ENABLE_DEMONSTRATION === 'true',
  analyze = analyzeRecording,
  checkEndpoint = checkResearchEndpoint,
}: AppProps) {
  const [recording, setRecording] = useState<File | null>(null)
  const [recordingSource, setRecordingSource] = useState<RecordingSource>('browser-camera')
  const [captureTimeline, setCaptureTimeline] = useState<CaptureTimelineDraft | null>(null)
  const [reanimatedSmileApplicable, setReanimatedSmileApplicable] = useState<boolean | null>(null)
  const [authorizedEndpoint, setAuthorizedEndpoint] = useState(false)
  const [result, setResult] = useState<DisplayResult | null>(null)
  const [analysisState, setAnalysisState] = useState<'idle' | 'running' | 'succeeded' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [analysisRetryAllowed, setAnalysisRetryAllowed] = useState(true)
  const [sessionKey, setSessionKey] = useState(0)
  const [journeyStep, setJourneyStep] = useState<JourneyStep>(1)
  const [captureSetupReady, setCaptureSetupReady] = useState(false)
  const [captureMode, setCaptureMode] = useState<'upload' | 'camera'>('camera')
  const [guidedRecordingActive, setGuidedRecordingActive] = useState(false)
  const [endpointState, setEndpointState] = useState<'checking' | 'ready' | 'unavailable' | 'unconfigured'>(
    apiEndpoint ? 'checking' : 'unconfigured',
  )
  const [endpointCheckAttempt, setEndpointCheckAttempt] = useState(0)
  const [reportRoute, setReportRoute] = useState(() => window.location.hash === '#research-report')
  const analysisGenerationRef = useRef(0)
  const inFlightRef = useRef(false)
  const journeyPanelRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const updateRoute = () => setReportRoute(window.location.hash === '#research-report')
    window.addEventListener('hashchange', updateRoute)
    window.addEventListener('popstate', updateRoute)
    return () => {
      window.removeEventListener('hashchange', updateRoute)
      window.removeEventListener('popstate', updateRoute)
    }
  }, [])

  useEffect(() => {
    document.title = reportRoute
      ? 'Research Movement Report · FACES'
      : 'FACES Research Capture'
  }, [reportRoute])

  useEffect(() => {
    if (!apiEndpoint) {
      setEndpointState('unconfigured')
      return
    }
    let active = true
    setEndpointState('checking')
    void checkEndpoint(apiEndpoint).then(
      () => { if (active) setEndpointState('ready') },
      () => { if (active) setEndpointState('unavailable') },
    )
    return () => { active = false }
  }, [apiEndpoint, checkEndpoint, endpointCheckAttempt])

  const preparationItems = useMemo(capturePreparation, [])
  const researchResult = result?.mode === 'research-inference' ? result : null

  useEffect(() => {
    if (reportRoute) return
    const panel = journeyPanelRef.current
    if (!panel) return
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    panel.scrollIntoView?.({ block: 'start', behavior: reducedMotion ? 'auto' : 'smooth' })
    const heading = panel.querySelector<HTMLElement>('[data-journey-heading]')
    heading?.focus({ preventScroll: true })
  }, [journeyStep, reportRoute])

  const handleRecordingChange = useCallback((
    file: File | null,
    source: RecordingSource,
    options?: RecordingChangeOptions,
  ) => {
    analysisGenerationRef.current += 1
    inFlightRef.current = false
    setRecording(file)
    setRecordingSource(source)
    setResult(null)
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setAnalysisState('idle')
    setAuthorizedEndpoint(false)
    setCaptureTimeline(options?.timeline ?? null)
    setJourneyStep(file ? 4 : 2)
    if (options?.preserveProtocolChoice) {
      if (typeof options.reanimatedSmileApplicable === 'boolean') {
        setReanimatedSmileApplicable(options.reanimatedSmileApplicable)
      }
    } else {
      setReanimatedSmileApplicable(null)
    }
  }, [])

  const handleReanimatedSmileApplicableChange = useCallback((applicable: boolean) => {
    analysisGenerationRef.current += 1
    setReanimatedSmileApplicable(applicable)
    setResult(null)
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setAnalysisState('idle')
  }, [])

  const runResearchAnalysis = async () => {
    if (
      inFlightRef.current || result ||
      !recording ||
      !apiEndpoint ||
      endpointState !== 'ready' ||
      !authorizedEndpoint ||
      reanimatedSmileApplicable === null
      || !captureTimeline
    ) return
    inFlightRef.current = true
    const generation = analysisGenerationRef.current + 1
    analysisGenerationRef.current = generation
    setAnalysisState('running')
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setResult(null)
    try {
      const accepted: ResearchInferenceResult = await analyze(recording, {
        endpoint: apiEndpoint,
        recordingSource,
        reanimatedSmileApplicable,
        timeline: captureTimeline,
      })
      if (analysisGenerationRef.current !== generation) return
      setResult(accepted)
      setAnalysisState('succeeded')
      setJourneyStep(5)
    } catch (error) {
      if (analysisGenerationRef.current !== generation) return
      setAnalysisState('error')
      setAnalysisError(error instanceof Error ? error.message : 'The research response was not accepted.')
      setAnalysisRetryAllowed(
        error instanceof InferenceContractError ? error.retryable : false,
      )
    } finally {
      inFlightRef.current = false
    }
  }

  const runDemonstration = () => {
    if (!recording || !demonstrationEnabled) return
    analysisGenerationRef.current += 1
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setAnalysisState('idle')
    setResult(createDemonstrationResult(recording))
    setJourneyStep(5)
  }

  const reset = () => {
    analysisGenerationRef.current += 1
    inFlightRef.current = false
    setRecording(null)
    setRecordingSource('browser-camera')
    setResult(null)
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setAnalysisState('idle')
    setAuthorizedEndpoint(false)
    setReanimatedSmileApplicable(null)
    setCaptureTimeline(null)
    setJourneyStep(1)
    setCaptureSetupReady(false)
    setCaptureMode('camera')
    setGuidedRecordingActive(false)
    setReportRoute(false)
    window.history.replaceState(null, '', '#top')
    setSessionKey((current) => current + 1)
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    window.scrollTo?.({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' })
  }

  const openReport = () => {
    window.history.pushState({ report: true }, '', '#research-report')
    setReportRoute(true)
  }

  const closeReport = () => {
    window.history.pushState({ report: false }, '', '#analysis')
    setReportRoute(false)
  }

  const handleGuidedActiveChange = useCallback((active: boolean) => {
    setGuidedRecordingActive(active)
    if (active) setJourneyStep(3)
  }, [])

  const confirmAndReset = () => {
    if (
      result?.mode === 'research-inference'
      && !window.confirm('Start a new session? This deletes the current browser recording and research report.')
    ) return
    reset()
  }

  const endpointStatusPanel = apiEndpoint ? (
    endpointState === 'ready' ? (
      <div className="endpoint-state"><span className="status-dot is-online" /><span><strong>Analysis endpoint ready</strong>Readiness and response validation verified</span></div>
    ) : endpointState === 'checking' ? (
      <div className="endpoint-state"><span className="status-dot is-checking" /><span><strong>Checking analysis endpoint</strong>Recording remains locked until the endpoint is ready</span></div>
    ) : (
      <div className="endpoint-state endpoint-state-retry"><span className="status-dot is-error" /><span><strong>Research endpoint unavailable</strong>Do not begin a patient recording yet</span><button className="text-action" type="button" onClick={() => setEndpointCheckAttempt((value) => value + 1)}>Retry endpoint check</button></div>
    )
  ) : (
    <div className="endpoint-state"><span className="status-dot" /><span><strong>Research endpoint not configured</strong>Add a vetted HTTPS endpoint to enable model inference</span></div>
  )

  return (
    <div className="app" id="top">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <AppHeader showResearchStrip={!reportRoute} />

      <main id="main-content">
        {reportRoute ? (
          result?.mode === 'research-inference' && recording ? (
            <ResultsView
              result={result}
              recording={recording}
              onBack={closeReport}
              onReset={confirmAndReset}
            />
          ) : (
            <section className="report-empty-state" aria-labelledby="report-empty-title">
              <span className="eyebrow">Private browser session</span>
              <h1 id="report-empty-title">Report not retained</h1>
              <p>The recording and report stay only in this browser session. Reloading or opening this address directly cannot restore them and never reruns analysis.</p>
              <a className="button button-primary" href="#analysis" onClick={() => setReportRoute(false)}>Return to research analysis</a>
            </section>
          )
        ) : null}
        <div hidden={reportRoute} aria-hidden={reportRoute ? 'true' : undefined}>
        <section className="journey-hero" aria-labelledby="hero-title">
          <div>
            <span className="eyebrow">Guided facial movement assessment</span>
            <h1 id="hero-title">Capture the full facial movement story.</h1>
            <p>One clear stage at a time—from preparation to a reviewable report.</p>
          </div>
          <div className="journey-hero-badge" aria-label="Seven- or eight-step automatic movement sequence">
            <strong>7–8</strong><span>guided movements<br />automatic timing</span>
          </div>
        </section>

        <div className="journey-shell" id="journey">
          <section className="workflow-section" aria-label="Current workflow stage">
            <WorkflowRail current={journeyStep} />
          </section>

          <div className="journey-panel" ref={journeyPanelRef}>
            {journeyStep === 1 ? (
              <header className="journey-stage-heading">
                <span className="journey-stage-kicker">Step 1 of 5 · Prepare</span>
                <h2 data-journey-heading tabIndex={-1}>Prepare for a consistent capture</h2>
                <p>Review the room setup and preview each facial movement before turning on the camera.</p>
              </header>
            ) : journeyStep === 2 ? (
              <header className="journey-stage-heading">
                <span className="journey-stage-kicker">Step 2 of 5 · Set up</span>
                <h2 data-journey-heading tabIndex={-1}>Set up the camera</h2>
                <p>Use this device by default, confirm framing, and keep the same protocol choice you reviewed.</p>
              </header>
            ) : journeyStep === 3 ? (
              <header className="journey-stage-heading">
                <span className="journey-stage-kicker">Step 3 of 5 · Record</span>
                <h2 data-journey-heading tabIndex={-1}>Complete the automatic recording</h2>
                <p>After Start, voice cues advance the full sequence. No Next button is needed during facial movements.</p>
              </header>
            ) : journeyStep === 4 ? (
              <header className="journey-stage-heading">
                <span className="journey-stage-kicker">Step 4 of 5 · Analyze</span>
                <h2 data-journey-heading tabIndex={-1}>Review the recording and run analysis</h2>
                <p>Confirm the retained video and endpoint before sending one analysis request.</p>
              </header>
            ) : (
              <header className="journey-stage-heading">
                <span className="journey-stage-kicker">Step 5 of 5 · Report</span>
                <h2 data-journey-heading tabIndex={-1}>Your report is ready</h2>
                <p>The completed result is locked for this browser session.</p>
              </header>
            )}

            <section className="preparation-section journey-preparation" hidden={journeyStep !== 1}>
              <div>
                <span className="eyebrow">Before recording</span>
                <h3>A consistent setup makes every visit more useful.</h3>
                <p>{FACES_PREPARATION[0]}</p>
              </div>
              <ul>
                {preparationItems.map((item) => <li key={item}><span><Check aria-hidden="true" size={17} /></span>{item}</li>)}
              </ul>
            </section>

            <div id="capture" key={sessionKey} hidden={journeyStep === 5}>
              <GuidedCaptureWorkspace
                journeyStage={journeyStep === 1 ? 'prepare' : journeyStep === 2 ? 'setup' : journeyStep === 3 ? 'record' : 'review'}
                endpointReady={demonstrationEnabled || endpointState === 'ready'}
                reanimatedSmileApplicable={reanimatedSmileApplicable}
                onReanimatedSmileApplicableChange={handleReanimatedSmileApplicableChange}
                onRecordingChange={handleRecordingChange}
                onSetupReadyChange={setCaptureSetupReady}
                onCaptureModeChange={setCaptureMode}
                onGuidedActiveChange={handleGuidedActiveChange}
              />
            </div>

            {journeyStep === 2 || journeyStep === 4 ? (
              <div className="journey-endpoint-status">{endpointStatusPanel}</div>
            ) : null}

            <section className="analysis-section" hidden={journeyStep !== 4} id="analysis" aria-labelledby="analysis-title">
          <div className="analysis-copy">
            <span className="eyebrow">Movement analysis</span>
            <h2 id="analysis-title">Validate the path before any result appears.</h2>
            <p>The server verifies the capture timeline, extracts paired MediaPipe geometry, and validates the analysis response before it appears.</p>
            <div className="model-chip"><ShieldCheck aria-hidden="true" size={18} /><span><strong>Analysis pipeline</strong>Timeline, geometry, and response checks</span></div>
          </div>
          <div className="analysis-actions-card">
            {researchResult ? (
              <div className="report-ready-card" role="status" aria-live="polite">
                <Check aria-hidden="true" size={22} />
                <div>
                  <strong>Research report ready</strong>
                  <p>The inference is locked for this session. Open the full report to review the score, movement observations, capture quality, and limits.</p>
                  <div className="analysis-button-stack">
                    <a className="button button-primary button-wide" href="#research-report" onClick={(event) => { event.preventDefault(); openReport() }}>
                      View full research report <ArrowRight aria-hidden="true" size={18} />
                    </a>
                    {recording ? <RecordingDownloadButton recording={recording} /> : null}
                    <button className="button button-secondary button-wide" type="button" onClick={confirmAndReset}>Start a new session</button>
                  </div>
                </div>
              </div>
            ) : apiEndpoint ? (
              <>
                <div className="privacy-warning"><LockKeyhole aria-hidden="true" size={21} /><p><strong>Facial video is identifiable.</strong> Send it only to an authorized research endpoint under the approved protocol.</p></div>
                <label className="authorization-check">
                  <input type="checkbox" checked={authorizedEndpoint} onChange={(event) => setAuthorizedEndpoint(event.target.checked)} />
                  <span>I confirm this is an authorized research endpoint.</span>
                </label>
                <div className="analysis-button-stack">
                  <button className="button button-primary button-wide" type="button" disabled={endpointState !== 'ready' || !recording || !authorizedEndpoint || !captureTimeline || reanimatedSmileApplicable === null || analysisState === 'running' || (analysisState === 'error' && !analysisRetryAllowed)} onClick={runResearchAnalysis}>
                    {analysisState === 'running' ? <><span className="spinner" /> Validating response…</> : analysisState === 'error' && !analysisRetryAllowed ? <>New recording required</> : <>Run research analysis <ArrowRight aria-hidden="true" size={18} /></>}
                  </button>
                  {recording ? <RecordingDownloadButton recording={recording} /> : null}
                  {recording ? (
                    <button className="button button-secondary button-wide" type="button" disabled={analysisState === 'running'} onClick={reset}>
                      Clear recording and start over
                    </button>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="analysis-hint">Configure the analysis endpoint before running this recording.</div>
            )}

            {!researchResult && demonstrationEnabled ? (
              <div className="demo-action">
                <div><Sparkles aria-hidden="true" size={20} /><span><strong>Interface demonstration</strong>Generated locally from file metadata; never model output</span></div>
                <button className="button button-secondary button-wide" type="button" disabled={!recording || analysisState === 'running'} onClick={runDemonstration}>Preview demonstration results</button>
              </div>
            ) : null}

            {!researchResult && !recording ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Add a recording to continue.</p> : null}
            {!researchResult && recording && reanimatedSmileApplicable === null ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Resolve conditional step 8 in the voice guide.</p> : null}
            {!researchResult && recording && !captureTimeline ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Analysis requires the guided capture timeline; uploaded videos need an authenticated timeline sidecar.</p> : null}
            {!researchResult && recording && reanimatedSmileApplicable === false ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Step 8 is marked unavailable; analysis will use the six completed active movements without imputation.</p> : null}
            {!researchResult && analysisError ? <p className="inline-alert" role="alert">{analysisError}</p> : null}
            {!researchResult && analysisState === 'error' && !analysisRetryAllowed ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> This recording cannot be resubmitted. Clear it, correct the capture, and record the guided sequence again.</p> : null}
          </div>
            </section>

            <div className="journey-report-stage" hidden={journeyStep !== 5}>
              {researchResult ? (
                <div className="report-ready-card" role="status" aria-live="polite">
                  <Check aria-hidden="true" size={24} />
                  <div>
                    <strong>Analysis report ready</strong>
                    <p>Open the complete report, download the recorded video, or begin a new browser session.</p>
                    <div className="analysis-button-stack">
                      <a className="button button-primary button-wide" href="#research-report" onClick={(event) => { event.preventDefault(); openReport() }}>
                        View full research report <ArrowRight aria-hidden="true" size={18} />
                      </a>
                      {recording ? <RecordingDownloadButton recording={recording} /> : null}
                      <button className="button button-secondary button-wide" type="button" onClick={confirmAndReset}>Start a new session</button>
                    </div>
                  </div>
                </div>
              ) : null}
              {result?.mode === 'demonstration' ? <ResultsView result={result} onReset={reset} /> : null}
            </div>

            {journeyStep === 3 && guidedRecordingActive ? null : <nav className="journey-actions" aria-label="Journey controls">
              {journeyStep === 1 ? (
                <>
                  <span className="journey-action-note">
                    {reanimatedSmileApplicable === null
                      ? 'Choose whether Step 8 applies before camera setup.'
                      : 'Review the movements at your own pace.'}
                  </span>
                  <button
                    className="button button-primary journey-next"
                    type="button"
                    disabled={reanimatedSmileApplicable === null}
                    onClick={() => setJourneyStep(2)}
                  >
                    {reanimatedSmileApplicable === null ? 'Choose Step 8 above to continue' : 'Continue to camera setup'} <ArrowRight aria-hidden="true" size={20} />
                  </button>
                </>
              ) : journeyStep === 2 ? (
                <>
                  <button className="button button-secondary" type="button" onClick={() => setJourneyStep(1)}>
                    <ArrowLeft aria-hidden="true" size={20} /> Back to preparation
                  </button>
                  <span className="journey-action-note">
                    {captureMode === 'upload' ? 'Choose a video above, or return to the live camera.' : captureSetupReady ? 'Camera and protocol choice are ready.' : 'Complete the setup requirement shown above to continue.'}
                  </span>
                  {captureMode === 'camera' ? (
                    <button className="button button-primary journey-next" type="button" disabled={!captureSetupReady} onClick={() => setJourneyStep(3)}>
                      Continue to recording <ArrowRight aria-hidden="true" size={20} />
                    </button>
                  ) : null}
                </>
              ) : journeyStep === 3 ? (
                <>
                  <button className="button button-secondary" type="button" disabled={guidedRecordingActive} onClick={() => setJourneyStep(2)}>
                    <ArrowLeft aria-hidden="true" size={20} /> Back to camera setup
                  </button>
                  <span className="journey-action-note">{guidedRecordingActive ? 'Recording is automatic. Follow the voice and screen.' : 'Start when you are comfortably positioned.'}</span>
                </>
              ) : journeyStep === 4 ? (
                <span className="journey-action-note">Run the analysis above, or use Record again in the video panel.</span>
              ) : (
                <span className="journey-action-note">Report complete · no further scrolling is required.</span>
              )}
            </nav>}
          </div>
        </div>

        <section className="research-boundary clinical-workflow-note" id="research-boundary">
          <span className="eyebrow">Clinical review</span>
          <h2>Designed to support clinician review.</h2>
          <p>FACES AI summarizes standardized facial movement recordings and keeps the source video available for review. Use the movement report alongside the recording and the clinical assessment appropriate to the encounter.</p>
        </section>
        </div>
      </main>

      <footer>
        <div><strong>FACES Research Capture</strong><span>Facial movement research interface</span></div>
        <p>No patient data is persisted by this browser prototype.</p>
      </footer>
    </div>
  )
}
