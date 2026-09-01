import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { WorkflowRail } from './WorkflowRail'

describe('WorkflowRail', () => {
  it('announces the total, label, and status of every workflow step', () => {
    render(<WorkflowRail current={2} />)

    const journey = screen.getByRole('navigation', { name: 'Assessment journey' })
    const items = within(journey).getAllByRole('listitem')
    expect(items).toHaveLength(5)
    expect(items[0]).toHaveAccessibleName('Step 1 of 5, Prepare, completed')
    expect(items[1]).toHaveAccessibleName('Step 2 of 5, Set up, current step')
    expect(items[2]).toHaveAccessibleName('Step 3 of 5, Record, upcoming')
    expect(items[1]).toHaveAttribute('aria-current', 'step')
  })

  it('provides a compact current-step summary for zoomed and narrow layouts', () => {
    const { container } = render(<WorkflowRail current={4} />)

    const summary = container.querySelector('.workflow-current-summary')
    expect(summary).not.toBeNull()
    expect(summary).toHaveAttribute('aria-hidden', 'true')
    expect(summary).toHaveTextContent('Step 4 of 5')
    expect(summary).toHaveTextContent('Analyze')
    expect(summary).toHaveTextContent('Review and run')
  })
})
