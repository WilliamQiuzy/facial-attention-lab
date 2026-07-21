import { ArrowLeft, Eye, MessageCircleQuestion, Printer, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { AttentionMap } from '../components/AttentionMap'
import { demoAnalysis } from '../data/demoCase'

export function PatientReportPage() {
  return (
    <article className="patient-page">
      <div className="patient-toolbar page-shell">
        <Link to="/analysis?case=demo-001">
          <ArrowLeft aria-hidden="true" /> Return to clinician demo
        </Link>
        <button type="button" onClick={() => window.print()} aria-label="Print patient explanation">
          <Printer aria-hidden="true" /> Print explanation
        </button>
      </div>

      <header className="patient-hero">
        <div className="patient-hero__inner page-shell">
          <div>
            <p className="eyebrow">Conversation guide · Synthetic demonstration</p>
            <h1>A guide to visual attention maps</h1>
            <p className="patient-hero__lede">
              An attention map is one way researchers can summarize where people looked at
              an image. It is not a score of a face, a feeling, or a recommendation.
            </p>
          </div>
          <div className="patient-safety-card">
            <ShieldCheck aria-hidden="true" />
            <div>
              <strong>This is not a result about you or any patient.</strong>
              <p>
                The pictures show different AI-generated people. The colored layers and
                numbers are simulated—not recorded from human eyes.
              </p>
            </div>
          </div>
        </div>
      </header>

      <section className="page-shell patient-section" aria-labelledby="map-explanation-title">
        <div className="patient-intro">
          <div>
            <p className="eyebrow">What the colors mean</p>
            <h2 id="map-explanation-title">Color shows more or less visual attention.</h2>
          </div>
          <p>
            Warmer areas represent more simulated attention in this interface example. The
            exact colors do not tell us why someone looked there.
          </p>
        </div>

        <div className="patient-map-grid">
          <AttentionMap result={demoAnalysis.imageA} showHeatmap opacity={58} showRegion={false} watermark={demoAnalysis.watermark} />
          <AttentionMap result={demoAnalysis.imageB} showHeatmap opacity={58} showRegion={false} watermark={demoAnalysis.watermark} />
        </div>
        <p className="patient-caption">
          These images are unpaired. Comparing them demonstrates the website layout; it does
          not show a change in one person.
        </p>
      </section>

      <section className="patient-meaning-band">
        <div className="page-shell patient-meaning-grid">
          <div>
            <Eye aria-hidden="true" />
            <p className="eyebrow">Attention can describe</p>
            <h2>Where and when someone looked.</h2>
            <ul>
              <li>How much gaze time entered a defined region</li>
              <li>How soon the region received a first fixation</li>
              <li>How many and how long fixations lasted</li>
            </ul>
          </div>
          <div>
            <MessageCircleQuestion aria-hidden="true" />
            <p className="eyebrow">Attention cannot explain</p>
            <h2>What that person thought.</h2>
            <ul>
              <li>Looking does not tell us what someone thinks or feels.</li>
              <li>It does not measure attractiveness, stigma, or social judgment.</li>
              <li>It does not show whether a procedure worked.</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="page-shell patient-questions" aria-labelledby="questions-title">
        <p className="eyebrow">Bring the focus back to your goals</p>
        <h2 id="questions-title">Questions you may want to ask your care team</h2>
        <ol>
          <li><span>01</span><p>What changes are realistic for my own anatomy and procedure?</p></li>
          <li><span>02</span><p>How will we evaluate healing, function, and what matters to me?</p></li>
          <li><span>03</span><p>Which parts of this research are measured, and which remain uncertain?</p></li>
        </ol>
        <div className="patient-final-note">
          <strong>Remember</strong>
          <p>
            Your experience cannot be reduced to a heatmap. This research interface is meant
            to support questions—not replace your voice or your clinical team.
          </p>
        </div>
      </section>
    </article>
  )
}
