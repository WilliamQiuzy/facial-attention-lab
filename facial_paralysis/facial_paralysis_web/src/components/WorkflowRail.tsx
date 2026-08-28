import { Check } from 'lucide-react'

interface WorkflowRailProps {
  readonly current: 1 | 2 | 3 | 4 | 5
}

const steps = [
  ['Prepare', 'Review movements'],
  ['Set up', 'Camera and framing'],
  ['Record', 'Automatic sequence'],
  ['Analyze', 'Review and run'],
  ['Report', 'Open the result'],
] as const

export function WorkflowRail({ current }: WorkflowRailProps) {
  return (
    <nav className="journey-progress" aria-label="Assessment journey">
    <ol className="workflow-rail">
      {steps.map(([title, description], index) => {
        const number = (index + 1) as 1 | 2 | 3 | 4 | 5
        const completed = number < current
        const active = number === current
        return (
          <li
            className={active ? 'is-active' : completed ? 'is-complete' : ''}
            key={title}
            aria-current={active ? 'step' : undefined}
          >
            <span className="workflow-number">{completed ? <Check aria-hidden="true" size={15} /> : number}</span>
            <span><strong>{title}</strong><small>{description}</small></span>
          </li>
        )
      })}
    </ol>
    </nav>
  )
}
