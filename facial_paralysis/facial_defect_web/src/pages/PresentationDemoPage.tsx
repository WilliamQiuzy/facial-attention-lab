import { useState } from 'react'
import { AttentionColorLegend } from '../components/AttentionColorLegend'
import {
  PRESENTATION_BOUNDARY,
  presentationSubjectOptions,
  type PresentationSubjectId,
  type PresentationTimepoint,
} from '../data/presentationDemoAssets'
import {
  PresentationAttentionStage,
  type PresentationViewMode,
} from '../presentation/PresentationAttentionStage'
import { PresentationBeforeAfterSlider } from '../presentation/PresentationBeforeAfterSlider'

type TimepointSelection = PresentationTimepoint | 'both'
type ComparisonLayout = 'side-by-side' | 'slider'

type PatientComparisonWorkspaceProps = {
  readonly initialSubjectId?: PresentationSubjectId
  readonly embedded?: boolean
}

const timepointOptions: readonly Readonly<{
  value: TimepointSelection
  label: string
}>[] = [
  { value: 'preoperative', label: 'Pre-operative' },
  { value: 'postoperative', label: 'Post-operative' },
  { value: 'both', label: 'Both' },
]

const viewOptions: readonly Readonly<{
  value: PresentationViewMode
  label: string
}>[] = [
  { value: 'photo', label: 'Photo' },
  { value: 'outline', label: 'Outline' },
  { value: 'composite', label: 'Photo + outline' },
]

const comparisonOptions: readonly Readonly<{
  value: ComparisonLayout
  label: string
}>[] = [
  { value: 'side-by-side', label: 'Side by side' },
  { value: 'slider', label: 'Drag slider' },
]

function SegmentedOption({
  checked,
  label,
  name,
  onChange,
}: {
  readonly checked: boolean
  readonly label: string
  readonly name: string
  readonly onChange: () => void
}) {
  return (
    <label className="presentation-segment">
      <input type="radio" name={name} checked={checked} onChange={onChange} />
      <span>{label}</span>
    </label>
  )
}

export function PatientComparisonWorkspace({
  initialSubjectId = 'subject-a',
  embedded = false,
}: PatientComparisonWorkspaceProps) {
  const [subjectId, setSubjectId] =
    useState<PresentationSubjectId>(initialSubjectId)
  const [timepoint, setTimepoint] = useState<TimepointSelection>('both')
  const [viewMode, setViewMode] = useState<PresentationViewMode>('photo')
  const [comparisonLayout, setComparisonLayout] =
    useState<ComparisonLayout>('side-by-side')
  const [showAttention, setShowAttention] = useState(true)
  const visibleTimepoints: readonly PresentationTimepoint[] =
    timepoint === 'both' ? ['preoperative', 'postoperative'] : [timepoint]
  const activeSubject = presentationSubjectOptions.find(
    (subject) => subject.id === subjectId,
  )!

  const workspace = (
    <>
      <section className="presentation-controls" aria-label="Comparison controls">
        {!embedded ? (
          <fieldset className="presentation-control-group presentation-control-group--subject">
            <legend>Patient</legend>
            <div className="presentation-segments">
              {presentationSubjectOptions.map((option) => (
                <SegmentedOption
                  key={option.id}
                  checked={subjectId === option.id}
                  label={option.label}
                  name="presentation-subject"
                  onChange={() => setSubjectId(option.id)}
                />
              ))}
            </div>
          </fieldset>
        ) : null}

        <fieldset className="presentation-control-group">
          <legend>Time point</legend>
          <div className="presentation-segments">
            {timepointOptions.map((option) => (
              <SegmentedOption
                key={option.value}
                checked={timepoint === option.value}
                label={option.label}
                name="presentation-timepoint"
                onChange={() => setTimepoint(option.value)}
              />
            ))}
          </div>
        </fieldset>

        <fieldset className="presentation-control-group">
          <legend>Display</legend>
          <div className="presentation-segments">
            {viewOptions.map((option) => (
              <SegmentedOption
                key={option.value}
                checked={viewMode === option.value}
                label={option.label}
                name="presentation-view"
                onChange={() => setViewMode(option.value)}
              />
            ))}
          </div>
        </fieldset>

        {timepoint === 'both' ? (
          <fieldset className="presentation-control-group presentation-control-group--compare">
            <legend>Compare</legend>
            <div className="presentation-segments">
              {comparisonOptions.map((option) => (
                <SegmentedOption
                  key={option.value}
                  checked={comparisonLayout === option.value}
                  label={option.label}
                  name="presentation-comparison"
                  onChange={() => setComparisonLayout(option.value)}
                />
              ))}
            </div>
          </fieldset>
        ) : null}

        <label className="presentation-toggle">
          <input
            type="checkbox"
            checked={showAttention}
            onChange={(event) => setShowAttention(event.target.checked)}
          />
          <span>Show attention layer</span>
        </label>
      </section>

      {showAttention ? (
        <div className="presentation-legend">
          <span>Relative attention</span>
          <AttentionColorLegend />
        </div>
      ) : null}

      <div className="presentation-subject-note">
        <div>
          <strong>Same patient</strong>
          {!embedded ? (
            <span className="presentation-subject-note__id">{activeSubject.label}</span>
          ) : null}
          <span>{activeSubject.description}</span>
        </div>
        <span>Frontal, non-mirrored view · patient left is viewer right</span>
      </div>

      {timepoint === 'both' && comparisonLayout === 'slider' ? (
        <PresentationBeforeAfterSlider
          subjectId={subjectId}
          viewMode={viewMode}
          showAttention={showAttention}
        />
      ) : (
        <section
          className={`presentation-comparison${visibleTimepoints.length === 1 ? ' presentation-comparison--single' : ''}`}
          aria-label="Before and after comparison"
        >
          {visibleTimepoints.map((visibleTimepoint) => (
            <PresentationAttentionStage
              key={`${subjectId}-${visibleTimepoint}`}
              subjectId={subjectId}
              timepoint={visibleTimepoint}
              viewMode={viewMode}
              showAttention={showAttention}
            />
          ))}
        </section>
      )}
    </>
  )

  if (!embedded) return workspace

  return (
    <section
      className="patient-comparison-workspace"
      aria-label="Patient before and after comparison"
    >
      <header className="patient-comparison-workspace__header">
        <div>
          <p>Patient comparison</p>
          <h2>Pre- and postoperative comparison</h2>
        </div>
        <p>
          Review either time point alone, or compare both views for the same patient.
        </p>
      </header>
      {workspace}
    </section>
  )
}

export function PresentationDemoPage() {
  return (
    <article
      className="presentation-page page-shell"
      data-provenance={PRESENTATION_BOUNDARY}
    >
      <header className="presentation-hero">
        <div>
          <p className="presentation-hero__kicker">Patient comparison</p>
          <h1>Patient before and after</h1>
          <p className="presentation-hero__lede">
            Review the same patient before and after facial excision, with photo
            and outline views in one workspace.
          </p>
        </div>
        <aside className="presentation-hero__note" aria-label="Comparison explanation">
          <strong>What changes</strong>
          <p>
            The cheek lesion becomes a closed, healing surgical incision. The
            illustrative attention signal remains present, but is noticeably
            lower at the surgical site.
          </p>
        </aside>
      </header>

      <PatientComparisonWorkspace />
    </article>
  )
}
