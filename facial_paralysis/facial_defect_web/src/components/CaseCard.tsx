import { Link } from 'react-router-dom'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'

const CATEGORY_LABELS: Readonly<Record<string, string>> = {
  mohs: 'Mohs',
  hn_cancer: 'Head & neck',
  trauma: 'Trauma',
  rhinophyma: 'Rhinophyma',
  burns: 'Burns',
  vascular: 'Vascular',
  nevus: 'Nevus',
}

export function humanCategory(category: string): string {
  return CATEGORY_LABELS[category] ?? category.replaceAll('_', ' ')
}

type CaseCardProps = {
  readonly asset: WorkbenchCatalogEntry
}

export function CaseCard({ asset }: CaseCardProps) {
  const title = asset.label.replace(/^Standalone synthetic case — /, '')

  return (
    <article className="case-card" data-testid="case-card">
      <figure className="case-card__preview">
        <img
          src={asset.url}
          alt={`${title} synthetic preview`}
          width="1024"
          height="1024"
          loading="lazy"
          decoding="async"
        />
      </figure>

      <div className="case-card__body">
        <h2>{title}</h2>

        <div className="case-card__actions">
          <Link
            className="workspace-button workspace-button--primary"
            to={`/analysis?case=${asset.id}`}
            aria-label={`Run simulation for ${title}`}
          >
            Run simulation
          </Link>
        </div>
      </div>
    </article>
  )
}
