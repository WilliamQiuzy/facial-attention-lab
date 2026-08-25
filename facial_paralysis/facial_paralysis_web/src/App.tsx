import { ArrowRight, Camera, Check, LockKeyhole, ShieldCheck, Sparkles } from 'lucide-react'
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
  EXPECTED_CANDIDATE_ID,
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
  const [recordingSource, setRecordingSource] = useState<RecordingSource>('livelink-upload')
  const [captureTimeline, setCaptureTimeline] = useState<CaptureTimelineDraft | null>(null)
  const [reanimatedSmileApplicable, setReanimatedSmileApplicable] = useState<boolean | null>(null)
  const [authorizedEndpoint, setAuthorizedEndpoint] = useState(false)
  const [result, setResult] = useState<DisplayResult | null>(null)
  const [analysisState, setAnalysisState] = useState<'idle' | 'running' | 'succeeded' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [analysisRetryAllowed, setAnalysisRetryAllowed] = useState(true)
  const [sessionKey, setSessionKey] = useState(0)
  const [endpointState, setEndpointState] = useState<'checking' | 'ready' | 'unavailable' | 'unconfigured'>(
    apiEndpoint ? 'checking' : 'unconfigured',
  )
  const [endpointCheckAttempt, setEndpointCheckAttempt] = useState(0)
  const [reportRoute, setReportRoute] = useState(() => window.location.hash === '#research-report')
  const analysisGenerationRef = useRef(0)
  const inFlightRef = useRef(false)

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

  const currentStep: 1 | 2 | 3 | 4 = result ? 4 : analysisState === 'running' ? 3 : recording ? 2 : 1
  const preparationItems = useMemo(capturePreparation, [])
  const researchResult = result?.mode === 'research-inference' ? result : null

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
  }

  const reset = () => {
    analysisGenerationRef.current += 1
    inFlightRef.current = false
    setRecording(null)
    setResult(null)
    setAnalysisError(null)
    setAnalysisRetryAllowed(true)
    setAnalysisState('idle')
    setAuthorizedEndpoint(false)
    setReanimatedSmileApplicable(null)
    setCaptureTimeline(null)
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

  const confirmAndReset = () => {
    if (
      result?.mode === 'research-inference'
      && !window.confirm('Start a new session? This deletes the current browser recording and research report.')
    ) return
    reset()
  }

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
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-inner">
            <div className="hero-copy">
              <span className="eyebrow">Facial movement assessment · Research prototype</span>
              <h1 id="hero-title">Capture the full facial movement story.</h1>
              <p>Guide a standardized FACES recording, bring in a LifeLink Face video, and review the one binary research output the current Shared V9 model supports.</p>
              <a className="button button-primary hero-action" href="#capture">Start a capture <ArrowRight aria-hidden="true" size={19} /></a>
            </div>
            <div className="hero-visual" aria-label="Seven- or eight-step facial movement protocol overview">
              <div className="face-orbit" aria-hidden="true">
                <span className="face-outline"><i className="eye-left" /><i className="eye-right" /><i className="mouth-line" /></span>
                {['01','02','03','04','05','06','07','08'].map((label) => <b key={label}>{label}</b>)}
              </div>
              <div className="hero-stat"><strong>7–8</strong><span>guided movements<br />3-second holds</span></div>
            </div>
          </div>
        </section>

        <section className="workflow-section" aria-label="Current workflow stage">
          <WorkflowRail current={currentStep} />
        </section>

        <section className="preparation-section">
          <div>
            <span className="eyebrow">Before recording</span>
            <h2>A consistent setup makes every visit more useful.</h2>
            <p>{FACES_PREPARATION[0]}</p>
          </div>
          <ul>
            {preparationItems.map((item) => <li key={item}><span><Check aria-hidden="true" size={17} /></span>{item}</li>)}
          </ul>
        </section>

        <div id="capture" key={sessionKey}>
          <GuidedCaptureWorkspace
            endpointReady={demonstrationEnabled || endpointState === 'ready'}
            reanimatedSmileApplicable={reanimatedSmileApplicable}
            onReanimatedSmileApplicableChange={handleReanimatedSmileApplicableChange}
            onRecordingChange={handleRecordingChange}
          />
        </div>

        <section className="analysis-section" id="analysis" aria-labelledby="analysis-title">
          <div className="analysis-copy">
            <span className="eyebrow">Research analysis</span>
            <h2 id="analysis-title">Validate the path before any result appears.</h2>
            <p>The server verifies the capture timeline, extracts paired MediaPipe geometry, and runs the pinned Shared V9 ensemble.</p>
            <div className="model-chip"><ShieldCheck aria-hidden="true" size={18} /><span><strong>Target release</strong>{EXPECTED_CANDIDATE_ID} · Shared V9</span></div>
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
                {endpointState === 'ready' ? (
                  <div className="endpoint-state"><span className="status-dot is-online" /><span><strong>Research endpoint ready</strong>Pinned Shared V9 readiness verified</span></div>
                ) : endpointState === 'checking' ? (
                  <div className="endpoint-state"><span className="status-dot is-checking" /><span><strong>Checking research endpoint</strong>Recording remains locked until Shared V9 is ready</span></div>
                ) : (
                  <div className="endpoint-state endpoint-state-retry"><span className="status-dot is-error" /><span><strong>Research endpoint unavailable</strong>Do not begin a patient recording yet</span><button className="text-action" type="button" onClick={() => setEndpointCheckAttempt((value) => value + 1)}>Retry endpoint check</button></div>
                )}
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
              <div className="endpoint-state"><span className="status-dot" /><span><strong>Research endpoint not configured</strong>Add a vetted HTTPS endpoint to enable model inference</span></div>
            )}

            {!researchResult && demonstrationEnabled ? (
              <div className="demo-action">
                <div><Sparkles aria-hidden="true" size={20} /><span><strong>Interface demonstration</strong>Generated locally from file metadata; never model output</span></div>
                <button className="button button-secondary button-wide" type="button" disabled={!recording || analysisState === 'running'} onClick={runDemonstration}>Preview demonstration results</button>
              </div>
            ) : null}

            {!researchResult && !recording ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Add a recording to continue.</p> : null}
            {!researchResult && recording && reanimatedSmileApplicable === null ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Resolve conditional step 8 in the voice guide.</p> : null}
            {!researchResult && recording && !captureTimeline ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Shared V9 requires the guided capture timeline; uploaded videos need an authenticated timeline sidecar.</p> : null}
            {!researchResult && recording && reanimatedSmileApplicable === false ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Step 8 is marked unavailable; analysis will use the six completed active movements without imputation.</p> : null}
            {!researchResult && analysisError ? <p className="inline-alert" role="alert">{analysisError}</p> : null}
            {!researchResult && analysisState === 'error' && !analysisRetryAllowed ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> This recording cannot be resubmitted. Clear it, correct the capture, and record the guided sequence again.</p> : null}
          </div>
        </section>

        {result?.mode === 'demonstration' ? <ResultsView result={result} onReset={reset} /> : null}

        <section className="research-boundary" id="research-boundary">
          <span className="eyebrow">Interpretation boundary</span>
          <h2>This is a research interface, not a diagnosis.</h2>
          <div className="boundary-grid">
            <p><strong>What it can show</strong>One Shared V9 class-1 research score from a fully timed seven- or eight-step FACES capture.</p>
            <p><strong>What it cannot show</strong>Eye or mouth severity, House-Brackmann grade, treatment advice, or clinical validation.</p>
            <p><strong>What stays human</strong>The clinician reviews the source recording and decides whether any research output is useful.</p>
          </div>
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
