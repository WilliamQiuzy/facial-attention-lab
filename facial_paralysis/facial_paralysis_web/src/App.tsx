import { ArrowRight, Camera, Check, LockKeyhole, ShieldCheck, Sparkles } from 'lucide-react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { AppHeader } from './components/AppHeader'
import { GuidedCaptureWorkspace } from './components/GuidedCaptureWorkspace'
import type { RecordingChangeOptions } from './components/MediaCapture'
import { ResultsView, type DisplayResult } from './components/ResultsView'
import { WorkflowRail } from './components/WorkflowRail'
import { createDemonstrationResult } from './model/demonstration'
import {
  analyzeRecording,
  EXPECTED_MODEL_FILE,
  type RecordingSource,
  type ResearchInferenceResult,
} from './model/inference'
import { FACES_PREPARATION } from './protocol/facesProtocol'
import './styles/app.css'

type AnalyzeFunction = typeof analyzeRecording

interface AppProps {
  readonly apiEndpoint?: string
  readonly demonstrationEnabled?: boolean
  readonly analyze?: AnalyzeFunction
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
  apiEndpoint = import.meta.env.VITE_FACIAL_PARALYSIS_API_URL ?? '',
  demonstrationEnabled = import.meta.env.VITE_ENABLE_DEMONSTRATION === 'true',
  analyze = analyzeRecording,
}: AppProps) {
  const [recording, setRecording] = useState<File | null>(null)
  const [recordingSource, setRecordingSource] = useState<RecordingSource>('livelink-upload')
  const [reanimatedSmileApplicable, setReanimatedSmileApplicable] = useState<boolean | null>(null)
  const [authorizedEndpoint, setAuthorizedEndpoint] = useState(false)
  const [result, setResult] = useState<DisplayResult | null>(null)
  const [analysisState, setAnalysisState] = useState<'idle' | 'running' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [sessionKey, setSessionKey] = useState(0)
  const analysisGenerationRef = useRef(0)

  const currentStep: 1 | 2 | 3 | 4 = result ? 4 : analysisState === 'running' ? 3 : recording ? 2 : 1
  const preparationItems = useMemo(capturePreparation, [])

  const handleRecordingChange = useCallback((
    file: File | null,
    source: RecordingSource,
    options?: RecordingChangeOptions,
  ) => {
    analysisGenerationRef.current += 1
    setRecording(file)
    setRecordingSource(source)
    setResult(null)
    setAnalysisError(null)
    setAnalysisState('idle')
    setAuthorizedEndpoint(false)
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
    setAnalysisState('idle')
  }, [])

  const runResearchAnalysis = async () => {
    if (
      !recording ||
      !apiEndpoint ||
      !authorizedEndpoint ||
      reanimatedSmileApplicable === null
    ) return
    const generation = analysisGenerationRef.current + 1
    analysisGenerationRef.current = generation
    setAnalysisState('running')
    setAnalysisError(null)
    setResult(null)
    try {
      const accepted: ResearchInferenceResult = await analyze(recording, {
        endpoint: apiEndpoint,
        recordingSource,
        reanimatedSmileApplicable,
      })
      if (analysisGenerationRef.current !== generation) return
      setResult(accepted)
      setAnalysisState('idle')
    } catch (error) {
      if (analysisGenerationRef.current !== generation) return
      setAnalysisState('error')
      setAnalysisError(error instanceof Error ? error.message : 'The research response was not accepted.')
    }
  }

  const runDemonstration = () => {
    if (!recording || !demonstrationEnabled) return
    analysisGenerationRef.current += 1
    setAnalysisError(null)
    setAnalysisState('idle')
    setResult(createDemonstrationResult(recording))
  }

  const reset = () => {
    analysisGenerationRef.current += 1
    setRecording(null)
    setResult(null)
    setAnalysisError(null)
    setAnalysisState('idle')
    setAuthorizedEndpoint(false)
    setReanimatedSmileApplicable(null)
    setSessionKey((current) => current + 1)
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    window.scrollTo?.({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' })
  }

  return (
    <div className="app" id="top">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <AppHeader />

      <main id="main-content">
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-inner">
            <div className="hero-copy">
              <span className="eyebrow">Facial movement assessment · Research prototype</span>
              <h1 id="hero-title">Capture the full facial movement story.</h1>
              <p>Guide a standardized FACES recording, bring in a LifeLink Face video, and review only the regional outputs the current research model can support.</p>
              <a className="button button-primary hero-action" href="#capture">Start a capture <ArrowRight aria-hidden="true" size={19} /></a>
            </div>
            <div className="hero-visual" aria-label="Eight-step facial movement protocol overview">
              <div className="face-orbit" aria-hidden="true">
                <span className="face-outline"><i className="eye-left" /><i className="eye-right" /><i className="mouth-line" /></span>
                {['01','02','03','04','05','06','07','08'].map((label) => <b key={label}>{label}</b>)}
              </div>
              <div className="hero-stat"><strong>8</strong><span>guided movements<br />3-second holds</span></div>
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
            reanimatedSmileApplicable={reanimatedSmileApplicable}
            onReanimatedSmileApplicableChange={handleReanimatedSmileApplicableChange}
            onRecordingChange={handleRecordingChange}
          />
        </div>

        <section className="analysis-section" id="analysis" aria-labelledby="analysis-title">
          <div className="analysis-copy">
            <span className="eyebrow">Research analysis</span>
            <h2 id="analysis-title">Validate the path before any result appears.</h2>
            <p>The browser does not run the PyTorch checkpoint. An authorized server must segment the protocol, run the pinned model, and return an exact versioned response.</p>
            <div className="model-chip"><ShieldCheck aria-hidden="true" size={18} /><span><strong>Target checkpoint</strong>{EXPECTED_MODEL_FILE}</span></div>
          </div>
          <div className="analysis-actions-card">
            {apiEndpoint ? (
              <>
                <div className="endpoint-state"><span className="status-dot is-online" /><span><strong>Research endpoint configured</strong>Strict response validation is on</span></div>
                <div className="privacy-warning"><LockKeyhole aria-hidden="true" size={21} /><p><strong>Facial video is identifiable.</strong> Send it only to an authorized research endpoint under the approved protocol.</p></div>
                <label className="authorization-check">
                  <input type="checkbox" checked={authorizedEndpoint} onChange={(event) => setAuthorizedEndpoint(event.target.checked)} />
                  <span>I confirm this is an authorized research endpoint.</span>
                </label>
                <button className="button button-primary button-wide" type="button" disabled={!recording || !authorizedEndpoint || reanimatedSmileApplicable === null || analysisState === 'running'} onClick={runResearchAnalysis}>
                  {analysisState === 'running' ? <><span className="spinner" /> Validating response…</> : <>Run research analysis <ArrowRight aria-hidden="true" size={18} /></>}
                </button>
              </>
            ) : (
              <div className="endpoint-state"><span className="status-dot" /><span><strong>Research endpoint not configured</strong>Add a vetted HTTPS endpoint to enable model inference</span></div>
            )}

            {demonstrationEnabled ? (
              <div className="demo-action">
                <div><Sparkles aria-hidden="true" size={20} /><span><strong>Interface demonstration</strong>Generated locally from file metadata; never model output</span></div>
                <button className="button button-secondary button-wide" type="button" disabled={!recording || analysisState === 'running'} onClick={runDemonstration}>Preview demonstration results</button>
              </div>
            ) : null}

            {!recording ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Add a recording to continue.</p> : null}
            {recording && reanimatedSmileApplicable === null ? <p className="analysis-hint"><Camera aria-hidden="true" size={17} /> Resolve conditional step 8 in the voice guide.</p> : null}
            {analysisError ? <p className="inline-alert" role="alert">{analysisError}</p> : null}
          </div>
        </section>

        {result ? <ResultsView result={result} onReset={reset} /> : null}

        <section className="research-boundary" id="research-boundary">
          <span className="eyebrow">Interpretation boundary</span>
          <h2>This is a research interface, not a diagnosis.</h2>
          <div className="boundary-grid">
            <p><strong>What it can show</strong>Binary warm-start probability and eye/mouth ordinal outputs from an exact accepted response.</p>
            <p><strong>What it cannot show</strong>Clinical grade, treatment advice, validated patient accuracy, or spatial localization.</p>
            <p><strong>What stays human</strong>The clinician reviews the source recording and decides whether any research output is useful.</p>
          </div>
        </section>
      </main>

      <footer>
        <div><strong>FACES Research Capture</strong><span>Facial movement research interface</span></div>
        <p>No patient data is persisted by this browser prototype.</p>
      </footer>
    </div>
  )
}
