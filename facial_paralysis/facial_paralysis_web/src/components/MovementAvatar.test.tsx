import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { MovementAvatar } from './MovementAvatar'

describe('MovementAvatar', () => {
  it('exposes the demonstrated movement to assistive technology', () => {
    render(
      <MovementAvatar
        action="eyebrow_raise"
        title="Eyebrow Raise"
        active
      />,
    )

    const avatar = screen.getByRole('img', {
      name: 'Eyebrow Raise movement demonstration',
    })
    expect(avatar).toHaveAttribute('data-action', 'eyebrow_raise')
    expect(avatar).toHaveClass('is-active')
  })

  it('renders a distinct visual state for the reanimated smile', () => {
    render(
      <MovementAvatar
        action="reanimated_smile"
        title="Reanimated Smile (if applicable)"
      />,
    )

    expect(
      screen.getByRole('img', {
        name: 'Reanimated Smile (if applicable) movement demonstration',
      }),
    ).toHaveAttribute('data-action', 'reanimated_smile')
  })
})
