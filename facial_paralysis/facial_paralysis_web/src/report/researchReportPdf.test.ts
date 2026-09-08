import { waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { downloadResearchReportPdf, type ResearchReportPdfData } from './researchReportPdf'
import { getSessionFilename } from './sessionIdentity'

const data: ResearchReportPdfData = {
  score: '48 / 100', scoreMeaning: 'MEEI development-group similarity.',
  outputClass: 'Below MEEI research cutpoint', cutpointSummary: '2 points below 50.',
  recordingCoverage: ['Neutral baseline and 6 movements.'], clinicalReviewNote: 'Review with recorded video.',
  actions: [{
    title: 'Eyebrow raise', region: 'Brow region', contextSeconds: '6.0 seconds', tracking: '32 of 32',
    imageDataUrl: 'data:image/jpeg;base64,invalid-image',
    measurements: [{ label: 'Brow-height change from rest', kind: 'Change from neutral', primaryValue: '1.0%', normalizedValue: '0.010', explanation: 'Movement from neutral.' }],
    influence: { status: 'stable', strength: 'Strong relative influence', direction: 'Moved upward.', relative: '100% relative magnitude.' },
    stability: ['3 of 3 model members', 'Mirror view: passed', '2 of 2 timing shifts'],
  }],
}

describe('research report PDF download', () => {
  it('uses the recording-linked filename and still builds a PDF when an evidence image fails', async () => {
    const recording = new File(['video'], 'patient-name.webm', { type: 'video/webm', lastModified: 1_700_000_000_000 })
    const create = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:pdf')
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    let filename = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { filename = this.download })
    await downloadResearchReportPdf(data, recording)
    expect(filename).toBe(getSessionFilename(recording, 'report'))
    expect(create.mock.calls[0][0]).toHaveProperty('type', 'application/pdf')
    expect((create.mock.calls[0][0] as Blob).size).toBeGreaterThan(1_000)
    await waitFor(() => expect(revoke).toHaveBeenCalledWith('blob:pdf'))
    expect(document.querySelector('a[download]')).not.toBeInTheDocument()
  })

  it('rejects a failed browser download request while releasing the anchor and URL', async () => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:failed-pdf')
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => { throw new Error('browser request failed') })
    await expect(downloadResearchReportPdf(data, new File(['video'], 'capture.webm'))).rejects.toThrow('browser request failed')
    await waitFor(() => expect(revoke).toHaveBeenCalledWith('blob:failed-pdf'))
    expect(document.querySelector('a[download]')).not.toBeInTheDocument()
  })
})
