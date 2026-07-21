import { FlaskConical } from 'lucide-react'

export function ResearchNotice() {
  return (
    <div className="research-notice" role="status" aria-label="Research use status">
      <div className="research-notice__inner">
        <FlaskConical aria-hidden="true" size={17} strokeWidth={2} />
        <strong>Research prototype</strong>
        <span>Not for diagnosis, treatment, or patient-care decisions.</span>
      </div>
    </div>
  )
}
