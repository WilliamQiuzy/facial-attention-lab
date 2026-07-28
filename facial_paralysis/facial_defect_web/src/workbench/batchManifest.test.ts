import { describe, expect, it, vi } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import {
  auditBatchManifest,
  createBatchManifest,
  type BatchManifest,
  type BatchManifestInput,
} from './batchManifest'
import { createInitialWorkspaceState, workspaceReducer } from './reducer'
import {
  WorkbenchError,
  type RoiAnnotation,
  type WorkspaceState,
} from './types'

const catalog = listWorkbenchAssets()

function createExplicitBlockedState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const firstDefault = state.roisByCase[catalog[0].id]!
  const secondDefault = state.roisByCase[catalog[1].id]!
  const { reviewerId: _firstReviewer, ...firstBase } = firstDefault
  const { reviewerId: _secondReviewer, ...secondBase } = secondDefault
  const draftRoi: RoiAnnotation = { ...firstBase, status: 'draft' }
  const inReviewRoi: RoiAnnotation = { ...secondBase, status: 'in_review' }
  return {
    ...state,
    roisByCase: {
      ...state.roisByCase,
      [catalog[0].id]: draftRoi,
      [catalog[1].id]: inReviewRoi,
    },
  }
}

function allCaseInput(): BatchManifestInput {
  return {
    workspaceState: createInitialWorkspaceState(),
    selectedCaseIds: catalog.map((asset) => asset.id),
    modelVersion: 'mock-salience-v0.4',
    config: { threshold: 0.44, smoothing: 0.31 },
  }
}

describe('immutable batch manifest', () => {
  it('blocks an approved partial rectangle with the exact source-binding blocker', () => {
    const initial = createInitialWorkspaceState()
    const partialCase = catalog[2]
    const current = initial.roisByCase[partialCase.id]!
    const partialState: WorkspaceState = {
      ...initial,
      roisByCase: {
        ...initial.roisByCase,
        [partialCase.id]: {
          ...current,
          geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
        },
      },
    }

    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: partialState,
      selectedCaseIds: [partialCase.id],
    })

    expect(manifest.items).toEqual([
      expect.objectContaining({
        caseId: partialCase.id,
        preflight: 'blocked',
        blocker: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
      }),
    ])
    expect(auditBatchManifest(manifest, partialState)).toEqual({
      valid: true,
      staleCaseIds: [],
    })
  })

  it('keeps every selected canonical case visible as ready or blocked', () => {
    const manifest = createBatchManifest(allCaseInput())

    expect(manifest.items).toHaveLength(10)
    expect(manifest.items.map((item) => item.caseId)).toEqual(
      catalog.map((asset) => asset.id),
    )
    expect(manifest.items.filter((item) => item.preflight === 'ready')).toHaveLength(10)
    expect(manifest.items.filter((item) => item.preflight === 'blocked')).toEqual([])
    expect(manifest.hash).toMatch(/^manifest_[a-f0-9]{16}$/)
    expect(manifest.modelMode).toBe('mock_only')
    expect(manifest.origin).toBe('synthetic_catalog_session')
    expect(manifest.persistence).toBe('memory_only')
  })

  it('deep-freezes the manifest, configuration, ROI snapshots, and item list', () => {
    const manifest = createBatchManifest(allCaseInput())
    const ready = manifest.items.find((item) => item.preflight === 'ready')!

    expect(Object.isFrozen(manifest)).toBe(true)
    expect(Object.isFrozen(manifest.config)).toBe(true)
    expect(Object.isFrozen(manifest.items)).toBe(true)
    expect(Object.isFrozen(manifest.excludedItems)).toBe(true)
    expect(Object.isFrozen(ready)).toBe(true)
    expect(Object.isFrozen(ready.roiGeometry)).toBe(true)
  })

  it('binds blocked exclusions into the immutable manifest and audits their ROI state', () => {
    const initial = createExplicitBlockedState()
    const readyCaseIds = catalog.slice(2).map((asset) => asset.id)
    const excludedCaseIds = catalog.slice(0, 2).map((asset) => asset.id)
    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: initial,
      selectedCaseIds: readyCaseIds,
      excludedCaseIds,
    })

    expect(manifest.items.map((item) => item.caseId)).toEqual(readyCaseIds)
    expect(manifest.excludedItems).toEqual([
      expect.objectContaining({ caseId: catalog[0].id, preflight: 'blocked' }),
      expect.objectContaining({ caseId: catalog[1].id, preflight: 'blocked' }),
    ])
    expect(Object.isFrozen(manifest.excludedItems[0])).toBe(true)
    expect(auditBatchManifest(manifest, initial)).toEqual({
      valid: true,
      staleCaseIds: [],
    })

    const changed = workspaceReducer(initial, {
      type: 'roi/submitReview',
      caseId: catalog[0].id,
      actorId: 'demo_author',
    })
    expect(auditBatchManifest(manifest, changed)).toEqual({
      valid: false,
      staleCaseIds: [catalog[0].id],
    })

    const withoutExclusions = createBatchManifest({
      ...allCaseInput(),
      workspaceState: initial,
      selectedCaseIds: readyCaseIds,
    })
    expect(withoutExclusions.hash).not.toBe(manifest.hash)
  })

  it('rejects duplicate, overlapping, unknown, or newly ready exclusions', () => {
    const readyCaseIds = catalog.slice(2).map((asset) => asset.id)
    const invalidExcludedCaseIds = [
      [catalog[0].id, catalog[0].id],
      [catalog[0].id, readyCaseIds[0]],
      ['UNKNOWN-CASE'],
      [readyCaseIds[0]],
    ]

    for (const excludedCaseIds of invalidExcludedCaseIds) {
      expect(() =>
        createBatchManifest({
          ...allCaseInput(),
          selectedCaseIds: readyCaseIds,
          excludedCaseIds,
        }),
      ).toThrow(WorkbenchError)
    }
  })

  it('is deterministic and changes its hash when scientific input changes', () => {
    const first = createBatchManifest(allCaseInput())
    const replay = createBatchManifest(allCaseInput())
    const changed = createBatchManifest({
      ...allCaseInput(),
      config: { threshold: 0.45, smoothing: 0.31 },
    })

    expect(replay).toEqual(first)
    expect(replay).not.toBe(first)
    expect(changed.hash).not.toBe(first.hash)
  })

  it('fails closed for an empty, duplicate, or unknown selection and invalid config', () => {
    const invalidSelections = [
      [],
      [catalog[2].id, catalog[2].id],
      [catalog[2].id, 'UNKNOWN-CASE'],
    ]

    for (const selectedCaseIds of invalidSelections) {
      expect(() =>
        createBatchManifest({ ...allCaseInput(), selectedCaseIds }),
      ).toThrow(WorkbenchError)
    }
    expect(() =>
      createBatchManifest({
        ...allCaseInput(),
        config: { threshold: Number.NaN, smoothing: 0.31 },
      }),
    ).toThrow(WorkbenchError)
  })

  it('detects any post-preflight ROI change without mutating or partially accepting the manifest', () => {
    const initial = createInitialWorkspaceState()
    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: initial,
      selectedCaseIds: [catalog[2].id, catalog[3].id],
    })
    const changed = workspaceReducer(initial, {
      type: 'roi/supersede',
      caseId: catalog[2].id,
      actorId: 'demo_author',
    })

    expect(auditBatchManifest(manifest, initial)).toEqual({
      valid: true,
      staleCaseIds: [],
    })
    expect(auditBatchManifest(manifest, changed)).toEqual({
      valid: false,
      staleCaseIds: [catalog[2].id],
    })
    expect(manifest.items[0].roiStatus).toBe('approved')
  })

  it('fails closed without throwing when the manifest hash or exact schema is tampered', () => {
    const state = createInitialWorkspaceState()
    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: state,
      selectedCaseIds: [catalog[2].id],
    })
    const tamperedHash = { ...manifest, hash: 'manifest_0000000000000000' }
    const extraField = { ...manifest, unexpected: true } as unknown as BatchManifest

    expect(() => auditBatchManifest(tamperedHash, state)).not.toThrow()
    expect(auditBatchManifest(tamperedHash, state).valid).toBe(false)
    expect(() => auditBatchManifest(extraField, state)).not.toThrow()
    expect(auditBatchManifest(extraField, state).valid).toBe(false)
  })

  it('rejects hidden and symbol schema keys instead of omitting them from the audit', () => {
    const state = createInitialWorkspaceState()
    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: state,
      selectedCaseIds: [catalog[2].id],
    })
    const hiddenTopLevel = { ...manifest }
    Object.defineProperty(hiddenTopLevel, 'hidden', { value: true })
    const symbolConfig = {
      ...manifest,
      config: { ...manifest.config, [Symbol('hidden')]: true },
    }
    const hiddenItem = {
      ...manifest,
      items: [{ ...manifest.items[0] }],
    }
    Object.defineProperty(hiddenItem.items[0], 'hidden', { value: true })
    const symbolItems = {
      ...manifest,
      items: [...manifest.items],
    }
    Object.defineProperty(symbolItems.items, Symbol('hidden'), { value: true })
    const hiddenGeometry = {
      ...manifest,
      items: [
        {
          ...manifest.items[0],
          roiGeometry: { ...manifest.items[0].roiGeometry! },
        },
      ],
    }
    Object.defineProperty(hiddenGeometry.items[0].roiGeometry!, 'hidden', {
      value: true,
    })

    for (const candidate of [
      hiddenTopLevel,
      symbolConfig,
      hiddenItem,
      symbolItems,
      hiddenGeometry,
    ]) {
      expect(() =>
        auditBatchManifest(candidate as BatchManifest, state),
      ).not.toThrow()
      expect(auditBatchManifest(candidate as BatchManifest, state).valid).toBe(false)
    }
  })

  it('rejects manifest records with custom prototypes', () => {
    const state = createInitialWorkspaceState()
    const manifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: state,
      selectedCaseIds: [catalog[2].id],
    })
    const customManifest = { ...manifest }
    Object.setPrototypeOf(customManifest, { inherited: true })
    const customConfig = { ...manifest, config: { ...manifest.config } }
    Object.setPrototypeOf(customConfig.config, { inherited: true })
    const customItem = { ...manifest, items: [{ ...manifest.items[0] }] }
    Object.setPrototypeOf(customItem.items[0], { inherited: true })

    for (const candidate of [customManifest, customConfig, customItem]) {
      expect(() => auditBatchManifest(candidate, state)).not.toThrow()
      expect(auditBatchManifest(candidate, state).valid).toBe(false)
    }
  })

  it('rejects forged readiness, duplicate items, and selected/excluded overlap without throwing', () => {
    const state = createExplicitBlockedState()
    const blockedManifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: state,
      selectedCaseIds: [catalog[0].id],
    })
    const forgedReady = {
      ...blockedManifest,
      items: [
        {
          ...blockedManifest.items[0],
          preflight: 'ready',
          blocker: undefined,
        },
      ],
    } as unknown as BatchManifest

    const readyManifest = createBatchManifest({
      ...allCaseInput(),
      workspaceState: state,
      selectedCaseIds: [catalog[2].id],
    })
    const duplicateItems = {
      ...readyManifest,
      items: [readyManifest.items[0], readyManifest.items[0]],
    } as unknown as BatchManifest
    const overlap = {
      ...readyManifest,
      excludedItems: [readyManifest.items[0]],
    } as unknown as BatchManifest

    for (const candidate of [forgedReady, duplicateItems, overlap]) {
      expect(() => auditBatchManifest(candidate, state)).not.toThrow()
      expect(auditBatchManifest(candidate, state).valid).toBe(false)
    }
  })

  it('returns invalid instead of throwing for malformed runtime manifest values', () => {
    const state = createInitialWorkspaceState()
    const malformed = [
      null,
      {},
      { items: null, excludedItems: [] },
      { items: [], excludedItems: 'not-an-array' },
    ]

    for (const candidate of malformed) {
      expect(() =>
        auditBatchManifest(candidate as unknown as BatchManifest, state),
      ).not.toThrow()
      expect(
        auditBatchManifest(candidate as unknown as BatchManifest, state).valid,
      ).toBe(false)
    }
  })

  it('does not access fetch or browser storage while preflighting', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const storageSpy = vi.spyOn(Storage.prototype, 'setItem')

    createBatchManifest(allCaseInput())

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
    storageSpy.mockRestore()
  })
})
