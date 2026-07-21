import { Route, Routes } from 'react-router-dom'
import { AppFooter } from './components/AppFooter'
import { AppHeader } from './components/AppHeader'
import { ResearchNotice } from './components/ResearchNotice'
import { ScrollToTop } from './components/ScrollToTop'
import { OverviewPage } from './pages/OverviewPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { MethodsPage } from './pages/MethodsPage'
import { ModelDataPage } from './pages/ModelDataPage'
import { PatientReportPage } from './pages/PatientReportPage'
import { WorklistPage } from './pages/WorklistPage'
import './styles/tokens.css'
import './styles/global.css'
import './styles/pages.css'

function PagePlaceholder({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <section className="page-shell page-shell--narrow">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="lede">
        This research-only surface is being prepared around a synthetic, provenance-first
        workflow.
      </p>
    </section>
  )
}

export function App() {
  return (
    <div className="app-root">
      <ScrollToTop />
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <ResearchNotice />
      <AppHeader />
      <main id="main-content" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route
            path="/cases"
            element={<WorklistPage />}
          />
          <Route
            path="/analysis"
            element={<AnalysisPage />}
          />
          <Route
            path="/patient-report"
            element={<PatientReportPage />}
          />
          <Route
            path="/model"
            element={<ModelDataPage />}
          />
          <Route
            path="/methods"
            element={<MethodsPage />}
          />
          <Route
            path="*"
            element={<PagePlaceholder eyebrow="Page not found" title="This route is not available" />}
          />
        </Routes>
      </main>
      <AppFooter />
    </div>
  )
}
