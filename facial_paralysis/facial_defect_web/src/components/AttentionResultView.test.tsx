import { readFileSync } from 'node:fs'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { listWorkbenchAssets } from '../workbench/catalog'
import { deriveClinicalAoiPresentation } from '../workbench/clinicalAoiPresentation'
import { createInferenceBinding, runMockEngine } from '../workbench/mockEngine'
import {
  createInitialWorkspaceState,
  getCaseRoi,
} from '../workbench/reducer'
import { createSucceededReviewTarget } from '../workbench/reviewTestFixtures'
import type { ApprovedRoiAnnotation, InferenceOutput } from '../workbench/types'
import { AttentionResultView } from './AttentionResultView'

function renderResultView(
  layout: 'clinician-stack' | 'patient-compact' = 'patient-compact',
  updateOutput?: (output: InferenceOutput) => InferenceOutput,
) {
  const target = createSucceededReviewTarget()
  const output = runMockEngine(target.binding)
  const view = render(
    <AttentionResultView
      asset={target.asset}
      output={updateOutput ? updateOutput(output) : output}
      roi={target.roi}
      layout={layout}
    />,
  )
  return { target, output, ...view }
}

function asConnected(output: InferenceOutput): InferenceOutput {
  return {
    ...output,
    attentionSemantics: {
      ...output.attentionSemantics,
      clinicalAoi: {
        ...output.attentionSemantics.clinicalAoi,
        registration: 'registration_geometry_unavailable_v1',
      },
    },
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark:
      'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED',
    modelIdentity: {
      modelId: 'observer-attention-test',
      modelVersion: 'test-v1',
      artifactSha256: 'a'.repeat(64),
      preprocessingVersion: 'preprocess-v1',
      calibrationVersion: 'calibration-v1',
      displayScaleId: 'display-scale-v1',
    },
    provenance: {
      engine: 'connected_model_gateway',
      engineVersion: 'test-spatial-contract',
      canonicalSyntheticAsset: true,
      deterministic: true,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

function renderCanonicalResult(assetId: string) {
  const state = createInitialWorkspaceState()
  const asset = listWorkbenchAssets().find((candidate) => candidate.id === assetId)
  if (!asset) throw new Error(`Unknown canonical asset ${assetId}`)
  const roi = getCaseRoi(state, asset.id) as ApprovedRoiAnnotation
  const binding = createInferenceBinding({
    clientRunId: `run-${asset.id}`,
    attemptToken: `attempt-${asset.id}`,
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
  const output = runMockEngine(binding)
  render(
    <AttentionResultView
      asset={asset}
      output={output}
      roi={roi}
      layout="clinician-stack"
    />,
  )
  return { output }
}

function displayedPercentages(region: HTMLElement): number[] {
  return within(region)
    .getAllByRole('listitem')
    .map((row) => {
      const value = within(row).getByText(/\d+%/).textContent
      if (!value) throw new Error('Missing displayed percentage')
      return Number.parseInt(value, 10)
    })
}

describe('AttentionResultView clinical AOI story', () => {
  it('renders the mock clinician story in source, density, overlay, AOI order without the 3x3 result', () => {
    const { container } = renderResultView('clinician-stack')

    expect(screen.queryAllByRole('radio')).toHaveLength(0)
    const source = screen.getByRole('heading', { name: 'Source image' })
    const density = screen.getByRole('heading', {
      name: 'Simulated attention-density field',
    })
    const overlay = screen.getByRole('heading', { name: 'Density overlay' })
    const summary = screen.getByRole('heading', { name: 'Clinical AOI summary' })

    expect(
      source.compareDocumentPosition(density) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      density.compareDocumentPosition(overlay) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      overlay.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    expect(screen.queryByText('Regional summary')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('list', { name: 'Result field summary' }),
    ).not.toBeInTheDocument()
    expect(container.querySelectorAll('.attention-summary__cell')).toHaveLength(0)
    expect(container).not.toHaveTextContent(/nine image-relative areas/i)
    expect(
      container.querySelector('.face-reference-outline'),
    ).not.toBeInTheDocument()
    expect(container).toHaveTextContent(
      /face outline unavailable.*registered contour or landmarks were not supplied/i,
    )
    expect(
      screen.getByRole('region', {
        name: 'Simulated attention-density field',
      }),
    ).toHaveAccessibleDescription(
      'Face outline unavailable: registered contour or landmarks were not supplied with this result.',
    )
  })

  it('separates the non-additive mock AOI readouts and states the point-weight basis', () => {
    renderResultView('clinician-stack')

    const summary = screen
      .getByRole('heading', { name: 'Clinical AOI summary' })
      .closest('section')
    expect(summary).not.toBeNull()
    const aoi = within(summary as HTMLElement)

    const orientation = aoi.getByRole('group', { name: 'Patient orientation' })
    const patientRight = within(orientation).getByText(
      'Patient right (viewer left)',
    )
    const patientLeft = within(orientation).getByText(
      'Patient left (viewer right)',
    )
    expect(
      patientRight.compareDocumentPosition(patientLeft) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    const templateShares = aoi.getByRole('region', {
      name: 'Anatomical template shares',
    })
    for (const label of [
      'Brow / forehead',
      'Orbital / eyes',
      'Nasal / midface',
      'Perioral / mouth',
      'Outside template',
    ]) {
      const row = within(templateShares).getByText(label).closest('li')
      expect(row).not.toBeNull()
      expect(within(row as HTMLElement).getByText(/\d+%/)).toBeVisible()
    }
    expect(templateShares).toHaveTextContent(
      'Subsite template shares + outside template = 100%.',
    )

    const hemifaces = aoi.getByRole('region', { name: 'Hemiface shares' })
    for (const label of ['Patient left', 'Patient right']) {
      const row = within(hemifaces).getByText(label).closest('li')
      expect(row).not.toBeNull()
      expect(within(row as HTMLElement).getByText(/\d+%/)).toBeVisible()
    }
    expect(hemifaces).toHaveTextContent('Hemifaces = 100%.')

    const centralTriangle = aoi.getByRole('region', {
      name: 'Central triangle share',
    })
    expect(within(centralTriangle).getByText(/\d+%/)).toBeVisible()
    expect(centralTriangle).toHaveTextContent(
      'Overlapping AOI · not additive to either group.',
    )

    expect(aoi.getByText('Registration: Synthetic template v1')).toBeVisible()
    expect(
      aoi.getByText(
        'Fixed face-relative areas summarize this simulated field. They do not change the result.',
      ),
    ).toBeVisible()
    const methodDetails = aoi
      .getByText('How this summary is calculated')
      .closest('details')
    expect(methodDetails).not.toBeNull()
    expect(methodDetails).not.toHaveAttribute('open')
    expect(
      aoi.getByText(
        /simulated point-center intensity weights assigned to the fixed template; display radius and boundary overlap are not integrated/i,
      ),
    ).not.toBeVisible()
    expect(
      aoi.getByText('Fixed anatomical template — simulation; no landmarks detected.'),
    ).toBeVisible()
    expect(
      aoi.getByText(/frontal, non-mirrored synthetic display/i),
    ).toBeVisible()
  })

  it.each([
    [
      'SYN-HNC-CHEEK-FREEFLAP',
      99,
      [22, 27, 28, 23, 0],
    ],
    [
      'SYN-TRAUMA-FRACTURE-RECON',
      101,
      [22, 22, 31, 25, 0],
    ],
  ] as const)(
    'uses deterministic largest-remainder percentages for canonical %s instead of the naive %i%% total',
    (assetId, naiveTotal, expectedTemplatePercentages) => {
      const { output } = renderCanonicalResult(assetId)
      const rawPresentation = deriveClinicalAoiPresentation(
        output.heatmap,
        output.binding.roiGeometry,
      )
      expect(rawPresentation.ok).toBe(true)
      if (!rawPresentation.ok) throw new Error('Expected canonical AOI presentation')
      const naivePercentages = [
        ...rawPresentation.subsites.map((subsite) =>
          Math.round(subsite.share * 100),
        ),
        Math.round(rawPresentation.outsideTemplateShare * 100),
      ]
      expect(naivePercentages.reduce((sum, value) => sum + value, 0)).toBe(
        naiveTotal,
      )

      const templateRegion = screen.getByRole('region', {
        name: 'Anatomical template shares',
      })
      const templatePercentages = displayedPercentages(templateRegion)
      expect(templatePercentages).toEqual(expectedTemplatePercentages)
      expect(
        templatePercentages.reduce((sum, value) => sum + value, 0),
      ).toBe(100)

      const hemifacePercentages = displayedPercentages(
        screen.getByRole('region', { name: 'Hemiface shares' }),
      )
      expect(hemifacePercentages.reduce((sum, value) => sum + value, 0)).toBe(
        100,
      )
    },
  )

  it('uses the same largest-remainder percentage in the dominant finding and its row', () => {
    renderCanonicalResult('SYN-HNC-CHEEK-FREEFLAP')

    const templateRegion = screen.getByRole('region', {
      name: 'Anatomical template shares',
    })
    const nasalRow = within(templateRegion).getByText('Nasal / midface').closest('li')
    expect(nasalRow).not.toBeNull()
    expect(within(nasalRow as HTMLElement).getByText('28%')).toBeVisible()
    expect(
      screen.getByText('Largest anatomical AOI share: Nasal / midface (28%).'),
    ).toBeVisible()
    expect(
      screen.queryByText('Largest anatomical AOI share: Nasal / midface (27%).'),
    ).not.toBeInTheDocument()
  })

  it('keeps a future surgical-site annotation explicitly absent and non-interactive', async () => {
    const user = userEvent.setup()
    const { container } = renderResultView('clinician-stack')

    expect(screen.getByText('Surgical-site mask: not set')).not.toBeVisible()
    await user.click(screen.getByText('How this summary is calculated'))
    expect(screen.getByText('Surgical-site mask: not set')).toBeVisible()
    expect(
      screen.getByText(
        /future, separately versioned contextual annotation.*absent.*not the immutable image bound.*does not alter the result/i,
      ),
    ).toBeVisible()
    expect(container.querySelector('.surgical-site-mask')).not.toBeInTheDocument()
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.queryByText(/drag|coordinate/i)).not.toBeInTheDocument()
  })

  it('uses mock-only language and keeps the interpretation boundaries permanent', () => {
    const { container } = renderResultView('clinician-stack')

    expect(container).toHaveTextContent(/simulated attention-density field/i)
    expect(container).toHaveTextContent(/interface structure/i)
    expect(container).toHaveTextContent(/not observed or human gaze/i)
    expect(container).toHaveTextContent(/not a patient prediction or result/i)
    expect(container).toHaveTextContent(
      /colors do not mean severity, defect location, healing, emotion, attractiveness, or surgical success/i,
    )
  })

  it('fails connected AOI closed without registration geometry while keeping the image story', () => {
    const { container } = renderResultView('clinician-stack', asConnected)

    expect(
      screen.getByRole('heading', {
        name: 'Predicted observer-attention density',
      }),
    ).toBeVisible()
    expect(container).toHaveTextContent(/research-unvalidated/i)
    expect(container).toHaveTextContent(/not observed gaze/i)
    expect(container).not.toHaveTextContent(/\bsimulated\b/i)
    expect(
      screen.getByRole('heading', { name: 'AOI summary unavailable' }),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Registration geometry was not supplied with this connected result.',
      ),
    ).toBeVisible()
    expect(container).toHaveTextContent(
      /landmarks or polygons.*orientation.*quality control/i,
    )
    expect(
      screen.queryByRole('heading', { name: 'Clinical AOI summary' }),
    ).not.toBeInTheDocument()
    expect(container).not.toHaveTextContent(/model-supplied registration/i)
    expect(container).not.toHaveTextContent(/central triangle|dominant anatomical AOI/i)

    expect(
      screen.getByRole('heading', { name: 'Source image' }),
    ).toBeVisible()
    expect(
      screen.getByRole('heading', { name: 'Density overlay' }),
    ).toBeVisible()
    const orientations = screen.getAllByRole('group', {
      name: 'Viewer orientation',
    })
    expect(orientations).toHaveLength(2)
    for (const orientation of orientations) {
      expect(within(orientation).getByText('Viewer left')).toBeVisible()
      expect(within(orientation).getByText('Viewer right')).toBeVisible()
    }
    expect(
      screen.queryByRole('group', { name: 'Patient orientation' }),
    ).not.toBeInTheDocument()
    expect(container).not.toHaveTextContent(/patient left|patient right/i)
    expect(
      container.querySelector('.face-reference-outline'),
    ).not.toBeInTheDocument()
    expect(container).toHaveTextContent(
      /face outline unavailable.*registered contour or landmarks were not supplied/i,
    )
    expect(
      screen.getByRole('region', {
        name: 'Predicted observer-attention density field',
      }),
    ).toHaveAccessibleDescription(
      'Face outline unavailable: registered contour or landmarks were not supplied with this result.',
    )
  })

  it('maps connected model intensities to the shared high-contrast thermal scale', () => {
    const { container } = renderResultView('clinician-stack', (output) => {
      const connected = asConnected(output)
      return {
        ...connected,
        heatmap: connected.heatmap.map((point, index) => ({
          ...point,
          intensity: index === 0 ? 1 : point.intensity,
        })),
      }
    })

    const densityPeak = container.querySelector(
      '.attention-signal-field__point',
    ) as HTMLElement | null
    const overlayPeak = container.querySelector(
      '.heatmap-point',
    ) as HTMLElement | null

    expect(densityPeak?.style.getPropertyValue('--attention-color-rgb')).toBe(
      '207 16 32',
    )
    expect(overlayPeak?.style.getPropertyValue('--attention-color-rgb')).toBe(
      '207 16 32',
    )
    expect(
      screen.getAllByRole('group', {
        name: 'Relative density color scale: blue low, cyan, yellow, orange, red peak',
      }),
    ).toHaveLength(2)
  })

  it('does not invent a dominant AOI for an all-zero field', () => {
    renderResultView('clinician-stack', (output) => ({
      ...output,
      heatmap: output.heatmap.map((point) => ({ ...point, intensity: 0 })),
    }))

    expect(
      screen.getByText(
        'No dominant anatomical AOI is available from this simulated interface result.',
      ),
    ).toBeVisible()
    const summary = screen
      .getByRole('heading', { name: 'Clinical AOI summary' })
      .closest('section')
    expect(summary).not.toBeNull()
    expect(within(summary as HTMLElement).getAllByText('0%')).toHaveLength(8)
    expect(
      within(summary as HTMLElement).getAllByText(
        'No non-zero point weight to distribute.',
      ),
    ).toHaveLength(2)
    expect(summary).not.toHaveTextContent(/=\s*100%/)
    expect(summary).not.toHaveTextContent(/strongest|dominant:/i)
  })

  it.each([
    [
      'empty field',
      (output: InferenceOutput): InferenceOutput => ({ ...output, heatmap: [] }),
      'No attention-density points are available.',
    ],
    [
      'invalid point',
      (output: InferenceOutput): InferenceOutput => ({
        ...output,
        heatmap: [{ ...output.heatmap[0], intensity: Number.NaN }],
      }),
      'The attention-density point data could not be verified.',
    ],
    [
      'point outside image boundary',
      (output: InferenceOutput): InferenceOutput => ({
        ...output,
        heatmap: [{ ...output.heatmap[0], x: 1 }],
        binding: {
          ...output.binding,
          roiGeometry: { x: 0, y: 0, width: 0.5, height: 1 },
        },
      }),
      'The attention-density field does not match the verified image boundary.',
    ],
  ])('fails closed before any visual for an %s', (_label, update, reason) => {
    renderResultView('clinician-stack', update)

    expect(
      screen.getByRole('heading', { name: 'Result view unavailable' }),
    ).toBeVisible()
    expect(screen.getByText(reason)).toBeVisible()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('heading', { name: 'Clinical AOI summary' }),
    ).not.toBeInTheDocument()
    expect(screen.queryByText('Surgical-site mask: not set')).not.toBeInTheDocument()
  })

  it('defaults the compact patient view to AOI summary and keeps three simple modes', async () => {
    const user = userEvent.setup()
    const { container } = renderResultView('patient-compact')

    expect(screen.getAllByRole('radio')).toHaveLength(3)
    expect(screen.getByRole('radio', { name: 'AOI summary' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Density field' })).not.toBeChecked()
    expect(screen.getByRole('radio', { name: 'Overlay' })).not.toBeChecked()
    expect(
      screen.getByRole('heading', { name: 'Clinical AOI summary' }),
    ).toBeVisible()
    expect(screen.getByRole('img', { name: /AOI template/i })).toBeVisible()
    expect(container.querySelector('.roi-box')).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Density field' }))
    expect(
      screen.getByRole('heading', {
        name: 'Simulated attention-density field',
      }),
    ).toBeVisible()
    expect(screen.queryByText('Display options')).not.toBeInTheDocument()
    expect(
      container.querySelector('.face-reference-outline'),
    ).not.toBeInTheDocument()
    expect(container).toHaveTextContent(
      /face outline unavailable.*registered contour or landmarks were not supplied/i,
    )

    await user.click(screen.getByRole('radio', { name: 'Overlay' }))
    expect(
      screen.getByRole('region', { name: 'Density overlay' }),
    ).toBeVisible()
    const options = screen.getByText('Display options').closest('details')
    expect(options).not.toHaveAttribute('open')
    expect(
      screen.getByRole('slider', { name: 'Overlay opacity' }),
    ).toHaveAttribute('name', 'overlayOpacity')
    expect(screen.getByRole('slider', { name: 'Overlay opacity' })).not.toBeVisible()
  })

  it('keeps the compact connected AOI unavailable and uses viewer-only orientation', async () => {
    const user = userEvent.setup()
    const { container } = renderResultView('patient-compact', asConnected)

    expect(screen.getByRole('radio', { name: 'AOI summary' })).toBeChecked()
    expect(
      screen.getByRole('heading', { name: 'AOI summary unavailable' }),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Registration geometry was not supplied with this connected result.',
      ),
    ).toBeVisible()
    expect(
      screen.queryByRole('heading', { name: 'Clinical AOI summary' }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Density field' }))
    expect(
      screen.getByRole('heading', {
        name: 'Predicted observer-attention density',
      }),
    ).toBeVisible()
    const sourceOrientation = screen.getByRole('group', {
      name: 'Viewer orientation',
    })
    expect(within(sourceOrientation).getByText('Viewer left')).toBeVisible()
    expect(within(sourceOrientation).getByText('Viewer right')).toBeVisible()
    expect(
      container.querySelector('.face-reference-outline'),
    ).not.toBeInTheDocument()
    expect(container).toHaveTextContent(
      /face outline unavailable.*registered contour or landmarks were not supplied/i,
    )

    await user.click(screen.getByRole('radio', { name: 'Overlay' }))
    const overlayOrientation = screen.getByRole('group', {
      name: 'Viewer orientation',
    })
    expect(within(overlayOrientation).getByText('Viewer left')).toBeVisible()
    expect(within(overlayOrientation).getByText('Viewer right')).toBeVisible()
    expect(container).not.toHaveTextContent(/patient left|patient right/i)
  })

  it('uses unique accessible IDs when two compact summaries share a page', () => {
    const target = createSucceededReviewTarget()
    const output = runMockEngine(target.binding)
    render(
      <>
        <AttentionResultView
          asset={target.asset}
          output={output}
          roi={target.roi}
          layout="patient-compact"
        />
        <AttentionResultView
          asset={target.asset}
          output={output}
          roi={target.roi}
          layout="patient-compact"
        />
      </>,
    )

    const headingIds = screen
      .getAllByRole('heading', { name: 'Clinical AOI summary' })
      .map((heading) => heading.id)
    const radioNames = screen
      .getAllByRole('radio', { name: 'AOI summary' })
      .map((radio) => radio.getAttribute('name'))
    expect(new Set(headingIds).size).toBe(2)
    expect(new Set(radioNames).size).toBe(2)
  })

  it('keeps the AOI layout fluid and share bars readable at narrow widths', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')
    const densityFieldRules = [
      ...css.matchAll(/\.attention-signal-field\s*\{([^}]+)\}/g),
    ]
      .map((match) => match[1])
      .join('\n')

    expect(css).toMatch(
      /\.clinical-aoi-summary\s*\{[^}]*min-width:\s*0;[^}]*overflow:\s*hidden/s,
    )
    expect(css).toMatch(
      /\.clinical-aoi-summary__bar\s*\{[^}]*min-width:\s*64px/s,
    )
    expect(css).toMatch(/@media \(max-width: 460px\)[\s\S]*?\.clinical-aoi-summary/)
    expect(css).toMatch(
      /@media \(max-width: 360px\)[\s\S]*?\.clinical-aoi-summary__shares li\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s+auto/s,
    )
    expect(css).toMatch(
      /@media \(max-width: 360px\)[\s\S]*?\.clinical-aoi-summary__bar\s*\{[^}]*grid-column:\s*1\s*\/\s*-1;[^}]*grid-row:\s*2;/s,
    )
    expect(densityFieldRules).not.toMatch(
      /33\.333%|repeating-(?:linear|radial)-gradient/,
    )
  })
})
