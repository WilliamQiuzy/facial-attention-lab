import { Check } from 'lucide-react'

interface WorkflowRailProps {
  readonly current: 1 | 2 | 3 | 4
}

const steps = [
  ['Prepare', 'Position and consent'],
  ['Capture', 'Upload or record'],
  ['Analyze', 'Validate and run'],
  ['Review', 'Regional outputs'],
] as const

export function WorkflowRail({ current }: WorkflowRailProps) {
  return (
    <ol className="workflow-rail" aria-label="Assessment workflow">
      {steps.map(([title, description], index) => {
        const number = (index + 1) as 1 | 2 | 3 | 4
        const completed = number < current
        const active = number === current
        return (
          <li className={active ? 'is-active' : completed ? 'is-complete' : ''} key={title}>
            <span className="workflow-number">{completed ? <Check aria-hidden="true" size={15} /> : number}</span>
            <span><strong>{title}</strong><small>{description}</small></span>
          </li>
        )
      })}
    </ol>
  )
}
