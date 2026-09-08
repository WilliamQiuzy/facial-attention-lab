import { Check } from 'lucide-react'

interface WorkflowRailProps {
  readonly current: 1 | 2 | 3 | 4 | 5
  readonly onNavigate?: (step: 1 | 2 | 3 | 4 | 5) => void
  readonly availableSteps?: readonly number[]
  readonly locked?: boolean
}

const steps = [
  ['Prepare', 'Review movements'],
  ['Set up', 'Camera and framing'],
  ['Record', 'Automatic sequence'],
  ['Analyze', 'Review and run'],
  ['Report', 'Open the result'],
] as const

export function WorkflowRail({ current, onNavigate, availableSteps = [], locked = false }: WorkflowRailProps) {
  const [currentTitle, currentDescription] = steps[current - 1]

  return (
    <nav className="journey-progress" aria-label="Assessment journey">
      <ol className="workflow-rail">
        {steps.map(([title, description], index) => {
          const number = (index + 1) as 1 | 2 | 3 | 4 | 5
          const completed = number < current
          const active = number === current
          const status = active ? 'current step' : completed ? 'completed' : 'upcoming'
          return (
            <li
              className={active ? 'is-active' : completed ? 'is-complete' : ''}
              key={title}
              aria-current={active ? 'step' : undefined}
              aria-label={`Step ${number} of 5, ${title}, ${status}`}
            >
              {onNavigate && availableSteps.includes(number) && number !== current ? <button className="workflow-step-link" type="button" disabled={locked} aria-label={`Return to ${title}`} onClick={() => onNavigate(number)} /> : null}
              <span className="workflow-number">{completed ? <Check aria-hidden="true" size={17} /> : number}</span>
              <span><strong>{title}</strong><small>{description}</small></span>
            </li>
          )
        })}
      </ol>
      <div className="workflow-current-summary" aria-hidden="true">
        <span>Step {current} of 5</span>
        <strong>{currentTitle}</strong>
        <small>{currentDescription}</small>
      </div>
      <details className="workflow-compact-menu">
        <summary><strong>{current}/5 · {currentTitle}</strong><span>All steps</span></summary>
        <div>{steps.map(([title], index) => <button type="button" key={title} disabled={locked || !availableSteps.includes(index + 1) || current === index + 1} aria-current={current === index + 1 ? 'step' : undefined} onClick={(event) => { onNavigate?.((index + 1) as 1 | 2 | 3 | 4 | 5); event.currentTarget.closest('details')?.removeAttribute('open') }}>{index + 1}. {title}{current === index + 1 ? ' · Current' : ''}</button>)}</div>
      </details>
    </nav>
  )
}
