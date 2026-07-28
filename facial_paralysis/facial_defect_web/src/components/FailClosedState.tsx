import { ShieldX } from 'lucide-react'
import { Link } from 'react-router-dom'

type FailClosedStateProps = {
  readonly eyebrow: string
  readonly title: string
  readonly requestedId?: string
  readonly description: string
  readonly backTo: string
  readonly backLabel: string
}

export function FailClosedState({
  eyebrow,
  title,
  requestedId,
  description,
  backTo,
  backLabel,
}: FailClosedStateProps) {
  return (
    <section className="workspace-page fail-closed-page">
      <div className="fail-closed-state page-shell">
        <ShieldX aria-hidden="true" />
        <p className="workspace-kicker">{eyebrow}</p>
        <h1>{title}</h1>
        {requestedId ? <code>{requestedId}</code> : null}
        <p>{description}</p>
        <Link className="workspace-button workspace-button--secondary" to={backTo}>
          {backLabel}
        </Link>
      </div>
    </section>
  )
}
