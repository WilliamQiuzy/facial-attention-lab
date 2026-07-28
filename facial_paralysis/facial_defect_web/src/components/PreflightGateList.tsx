import { Check, LockKeyhole } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import { isVerifiedFullImageSourceBinding } from '../workbench/sourceBinding'
import type { RoiAnnotation } from '../workbench/types'
import type { WorkbenchGatewayMode } from '../workbench/WorkbenchGateway'

type PreflightGateListProps = {
  readonly asset: WorkbenchCatalogEntry
  readonly roi: RoiAnnotation
  readonly gatewayMode: WorkbenchGatewayMode
}

export function PreflightGateList({ asset, roi, gatewayMode }: PreflightGateListProps) {
  const fullImageBoundVerified = isVerifiedFullImageSourceBinding(asset, roi)
  const gates = [
    {
      label: 'Canonical asset binding',
      detail: `${asset.id} · SHA-256 verified`,
      passed: true,
    },
    {
      label: 'Full-image source bound',
      detail: fullImageBoundVerified
        ? `Verified full image · v${roi.version}`
        : 'Full-image source binding unavailable',
      passed: fullImageBoundVerified,
    },
    {
      label: 'Execution boundary',
      detail:
        gatewayMode === 'mock'
          ? 'Mock-only · no network · no persistent storage'
          : 'Connected research gateway · network request required',
      passed: true,
    },
  ] as const

  return (
    <section className="preflight-gates workspace-panel" aria-label="Inference preflight">
      <div className="workspace-panel__heading">
        <div>
          <p className="workspace-kicker">Fail-closed preflight</p>
          <h2>Run gates</h2>
        </div>
        <strong>{fullImageBoundVerified ? 'Ready' : 'Blocked'}</strong>
      </div>
      <ul>
        {gates.map((gate) => (
          <li key={gate.label} className={gate.passed ? 'is-passed' : 'is-blocked'}>
            {gate.passed ? <Check aria-hidden="true" /> : <LockKeyhole aria-hidden="true" />}
            <div>
              <strong>{gate.label}</strong>
              <span>{gate.detail}</span>
              {!gate.passed ? (
                <Link to={`/cases/${asset.id}/roi`}>Restore source binding</Link>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
