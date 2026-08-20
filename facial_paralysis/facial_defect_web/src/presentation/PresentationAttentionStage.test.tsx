import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PRESENTATION_BOUNDARY } from '../data/presentationDemoAssets'
import { presentationAttentionByTimepoint } from './presentationAttention'
import { PresentationAttentionStage } from './PresentationAttentionStage'

describe('PresentationAttentionStage', () => {
  it('shows the exact synthetic pre-operative photo with its simulated signal', () => {
    const { container } = render(
      <PresentationAttentionStage
        timepoint="preoperative"
        viewMode="photo"
        showAttention
      />,
    )

    expect(
      screen.getByRole('img', {
        name: /synthetic pre-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(
      presentationAttentionByTimepoint.preoperative.length,
    )
    expect(screen.getByText(PRESENTATION_BOUNDARY)).toBeVisible()
  })

  it('replaces the photo with a hash-bound MediaPipe contour in outline mode', () => {
    const { container } = render(
      <PresentationAttentionStage
        timepoint="postoperative"
        viewMode="outline"
        showAttention
      />,
    )

    expect(screen.queryByRole('img', { name: /photograph/i })).not.toBeInTheDocument()
    expect(
      screen.getByRole('img', {
        name: /abstract facial outline with simulated post-operative attention/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.patient-face-contour__path')).toHaveLength(13)
    expect(
      container.querySelector('.patient-face-contour'),
    ).toHaveAttribute('data-geometry-source', 'on_device_face_landmarks')
  })

  it('keeps a smaller but non-zero post-operative cheek signal', () => {
    const pre = render(
      <PresentationAttentionStage
        timepoint="preoperative"
        viewMode="photo"
        showAttention
      />,
    ).container.querySelector('[data-total-signal]')
    const post = render(
      <PresentationAttentionStage
        timepoint="postoperative"
        viewMode="photo"
        showAttention
      />,
    ).container.querySelector('[data-total-signal]')

    expect(Number(post?.getAttribute('data-total-signal'))).toBeGreaterThan(0)
    expect(Number(post?.getAttribute('data-total-signal'))).toBeLessThan(
      Number(pre?.getAttribute('data-total-signal')),
    )
  })

  it('can show either source image without a simulated signal layer', () => {
    const { container } = render(
      <PresentationAttentionStage
        timepoint="postoperative"
        viewMode="photo"
        showAttention={false}
      />,
    )

    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(0)
    expect(screen.getByText('Attention layer hidden')).toBeVisible()
  })
})
