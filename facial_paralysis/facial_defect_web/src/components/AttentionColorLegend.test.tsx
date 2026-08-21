import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AttentionColorLegend } from './AttentionColorLegend'

describe('AttentionColorLegend', () => {
  it('explains the scale in patient-friendly language', () => {
    render(<AttentionColorLegend />)

    expect(screen.getByText('Less attention')).toBeVisible()
    expect(screen.getByText('More attention')).toBeVisible()
    expect(screen.queryByText('Low')).not.toBeInTheDocument()
    expect(screen.queryByText('Peak')).not.toBeInTheDocument()
  })
})
