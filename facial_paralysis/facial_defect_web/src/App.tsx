import {
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
  useParams,
} from 'react-router-dom'
import { AppFooter } from './components/AppFooter'
import { AppHeader } from './components/AppHeader'
import { ScrollToTop } from './components/ScrollToTop'
import { DEMO_PATIENT_RECORDS } from './data/demoPatientRecords'
import { AnalysisPage } from './pages/AnalysisPage'
import { BatchJobsPage } from './pages/BatchJobsPage'
import { CaseRoiPage } from './pages/CaseRoiPage'
import { ClinicalReviewQueuePage } from './pages/ClinicalReviewQueuePage'
import { MethodsPage } from './pages/MethodsPage'
import { ModelDataPage } from './pages/ModelDataPage'
import { ModelComparePage } from './pages/ModelComparePage'
import { NewPatientPage } from './pages/NewPatientPage'
import { NewPatientVisitPage } from './pages/NewPatientVisitPage'
import { OverviewPage } from './pages/OverviewPage'
import { PatientDetailPage } from './pages/PatientDetailPage'
import { PatientReportPage } from './pages/PatientReportPage'
import { PatientVisitPage } from './pages/PatientVisitPage'
import { PatientsPage } from './pages/PatientsPage'
import { ResultReviewPage } from './pages/ResultReviewPage'
import { ReviewQueuePage } from './pages/ReviewQueuePage'
import { RunDetailPage } from './pages/RunDetailPage'
import { RunsPage } from './pages/RunsPage'
import { WorklistPage } from './pages/WorklistPage'
import {
  PatientWorkflowProvider,
  type PatientWorkflowProviderProps,
} from './patientWorkflow/PatientWorkflowProvider'
import { createInitialPatientWorkflowState } from './patientWorkflow/reducer'
import type { WorkbenchGateway } from './workbench/WorkbenchGateway'
import { createWorkbenchGateway } from './workbench/createWorkbenchGateway'
import {
  WorkspaceProvider,
  type WorkspaceProviderProps,
} from './workbench/WorkspaceProvider'
import './styles/tokens.css'
import './styles/global.css'
import './styles/pages.css'
import './styles/workbench.css'
import './styles/task6.css'
import './styles/patient-workflow.css'

const defaultWorkbenchGateway = createWorkbenchGateway()

function localTodayIso(): string {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export type AppProps = {
  readonly gateway?: WorkbenchGateway
  readonly initialState?: WorkspaceProviderProps['initialState']
  readonly queueDelayMs?: number
  readonly patientInitialState?: PatientWorkflowProviderProps['initialState']
  readonly patientQueueDelayMs?: number
  readonly patientAnalysisDelayMs?: number
}

function PageNotFound() {
  return (
    <section className="workspace-page workspace-placeholder page-shell">
      <p className="workspace-kicker">404</p>
      <h1>Page not found</h1>
      <p>We could not find the page you requested.</p>
      <div className="workspace-placeholder__actions">
        <Link
          className="workspace-button workspace-button--primary"
          to="/patients"
        >
          Return to Patients
        </Link>
        <Link
          className="workspace-button workspace-button--secondary"
          to="/about"
        >
          Help
        </Link>
      </div>
    </section>
  )
}

function LegacyResearchReviewRedirect() {
  const { reviewId = '' } = useParams()
  const location = useLocation()

  return (
    <Navigate
      replace
      state={location.state}
      to={{
        pathname: `/research/reviews/${encodeURIComponent(reviewId)}`,
        search: location.search,
        hash: location.hash,
      }}
    />
  )
}

function RoutedWorkspace() {
  return (
    <div className="app-root">
      <ScrollToTop />
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <AppHeader />
      <main id="main-content" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Navigate to="/patients" replace />} />
          <Route path="/patients" element={<PatientsPage />} />
          <Route path="/patients/new" element={<NewPatientPage />} />
          <Route
            path="/patients/:patientId"
            element={<PatientDetailPage />}
          />
          <Route
            path="/patients/:patientId/visits/new"
            element={<NewPatientVisitPage />}
          />
          <Route
            path="/patients/:patientId/visits/:visitId"
            element={<PatientVisitPage />}
          />
          <Route path="/cases" element={<WorklistPage />} />
          <Route path="/cases/:caseId/roi" element={<CaseRoiPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/jobs" element={<BatchJobsPage />} />
          <Route path="/models" element={<ModelComparePage />} />
          <Route
            path="/reviews"
            element={<ClinicalReviewQueuePage />}
          />
          <Route
            path="/research/reviews"
            element={<ReviewQueuePage />}
          />
          <Route
            path="/research/reviews/:reviewId"
            element={<ResultReviewPage />}
          />
          <Route
            path="/reviews/:reviewId"
            element={<LegacyResearchReviewRedirect />}
          />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/patient-report" element={<PatientReportPage />} />
          <Route path="/about" element={<OverviewPage />} />
          <Route path="/methods" element={<MethodsPage />} />
          <Route path="/integration" element={<ModelDataPage />} />
          <Route path="/model" element={<Navigate to="/integration" replace />} />
          <Route
            path="*"
            element={<PageNotFound />}
          />
        </Routes>
      </main>
      <AppFooter />
    </div>
  )
}

export function App({
  gateway = defaultWorkbenchGateway,
  initialState,
  queueDelayMs,
  patientInitialState,
  patientQueueDelayMs = 500,
  patientAnalysisDelayMs = 900,
}: AppProps) {
  const effectiveQueueDelayMs =
    queueDelayMs ?? (gateway.mode === 'mock' ? 700 : 0)
  const defaultPatientState =
    patientInitialState ??
    createInitialPatientWorkflowState(
      DEMO_PATIENT_RECORDS,
      localTodayIso(),
    )

  return (
    <WorkspaceProvider
      gateway={gateway}
      initialState={initialState}
      queueDelayMs={effectiveQueueDelayMs}
    >
      <PatientWorkflowProvider
        initialState={defaultPatientState}
        queueDelayMs={patientQueueDelayMs}
        analysisDelayMs={patientAnalysisDelayMs}
      >
        <RoutedWorkspace />
      </PatientWorkflowProvider>
    </WorkspaceProvider>
  )
}
