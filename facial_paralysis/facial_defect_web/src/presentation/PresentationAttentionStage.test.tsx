import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { presentationAttentionBySubject } from './presentationAttention'
import { PresentationAttentionStage } from './PresentationAttentionStage'

describe('PresentationAttentionStage', () => {
  it('shows the exact sample pre-operative photo with its illustrative signal', () => {
    const { container } = render(
      <PresentationAttentionStage
        subjectId="subject-a"
        timepoint="preoperative"
        viewMode="photo"
        showAttention
      />,
    )

    expect(
      screen.getByRole('img', {
        name: /sample pre-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(
      presentationAttentionBySubject['subject-a'].preoperative.length,
    )
    expect(screen.queryByText(/clinical use blocked/i)).not.toBeInTheDocument()
  })

  it('replaces the photo with a hash-bound MediaPipe contour in outline mode', () => {
    const { container } = render(
      <PresentationAttentionStage
        subjectId="subject-a"
        timepoint="postoperative"
        viewMode="outline"
        showAttention
      />,
    )

    expect(screen.queryByRole('img', { name: /photograph/i })).not.toBeInTheDocument()
    expect(
      screen.getByRole('img', {
        name: /abstract facial outline with illustrative post-operative attention/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.patient-face-contour__path')).toHaveLength(13)
    expect(
      container.querySelector('.patient-face-contour'),
    ).toHaveAttribute('data-geometry-source', 'on_device_face_landmarks')
  })

  it('can combine the exact photo and its own outline in one uncluttered view', () => {
    const { container } = render(
      <PresentationAttentionStage
        subjectId="subject-b"
        timepoint="postoperative"
        viewMode="composite"
        showAttention
      />,
    )

    expect(
      screen.getByRole('img', {
        name: /subject b sample post-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.patient-face-contour__path')).toHaveLength(13)
    expect(screen.getByText('Photo + outline')).toBeVisible()
  })

  it('keeps a smaller but non-zero post-operative cheek signal', () => {
    const pre = render(
      <PresentationAttentionStage
        subjectId="subject-a"
        timepoint="preoperative"
        viewMode="photo"
        showAttention
      />,
    ).container.querySelector('[data-total-signal]')
    const post = render(
      <PresentationAttentionStage
        subjectId="subject-a"
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
        subjectId="subject-b"
        timepoint="postoperative"
        viewMode="photo"
        showAttention={false}
      />,
    )

    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(0)
    expect(screen.getByText('Attention off')).toBeVisible()
    expect(screen.getByText('Healing surgical incision')).toBeVisible()
  })
})
