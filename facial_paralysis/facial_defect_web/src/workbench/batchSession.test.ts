import { describe, expect, it } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import { createBatchManifest } from './batchManifest'
import {
  batchSessionReducer,
  createInitialBatchSessionState,
  manifestMatchesBatchDraft,
} from './batchSession'
import { createInitialWorkspaceState } from './reducer'
import type { RoiAnnotation, WorkspaceState } from './types'

const catalog = listWorkbenchAssets()

function createExplicitBlockedState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const defaultRoi = state.roisByCase[catalog[0].id]!
  const { reviewerId: _reviewerId, ...draftBase } = defaultRoi
  const draftRoi: RoiAnnotation = { ...draftBase, status: 'draft' }
  return {
    ...state,
    roisByCase: {
      ...state.roisByCase,
      [catalog[0].id]: draftRoi,
    },
  }
}

describe('batch session state machine', () => {
  it('manages canonical multi-selection without duplicates', () => {
    const initial = createInitialBatchSessionState()
    const first = batchSessionReducer(initial, {
      type: 'selection/toggle',
      caseId: catalog[2].id,
    })
    const duplicateToggle = batchSessionReducer(first, {
      type: 'selection/toggle',
      caseId: catalog[2].id,
    })
    const all = batchSessionReducer(duplicateToggle, {
      type: 'selection/selectAll',
      caseIds: catalog.map((asset) => asset.id),
    })

    expect(first.selectedCaseIds).toEqual([catalog[2].id])
    expect(duplicateToggle.selectedCaseIds).toEqual([])
    expect(all.selectedCaseIds).toHaveLength(10)
    expect(new Set(all.selectedCaseIds).size).toBe(10)
  })

  it('pins a preflight manifest and detects any later draft change', () => {
    let state = batchSessionReducer(createInitialBatchSessionState(), {
      type: 'selection/selectAll',
      caseIds: [catalog[2].id, catalog[3].id],
    })
    const manifest = createBatchManifest({
      workspaceState: createInitialWorkspaceState(),
      selectedCaseIds: state.selectedCaseIds,
      modelVersion: state.modelVersion,
      config: state.config,
    })
    state = batchSessionReducer(state, { type: 'manifest/set', manifest })

    expect(manifestMatchesBatchDraft(state)).toBe(true)

    const changed = batchSessionReducer(state, {
      type: 'config/update',
      field: 'threshold',
      value: 0.61,
    })
    expect(changed.manifest).toBe(manifest)
    expect(manifestMatchesBatchDraft(changed)).toBe(false)
  })

  it('freezes submitted run mappings and locks the scientific draft', () => {
    let state = batchSessionReducer(createInitialBatchSessionState(), {
      type: 'selection/selectAll',
      caseIds: [catalog[2].id],
    })
    const manifest = createBatchManifest({
      workspaceState: createInitialWorkspaceState(),
      selectedCaseIds: state.selectedCaseIds,
      modelVersion: state.modelVersion,
      config: state.config,
    })
    state = batchSessionReducer(state, { type: 'manifest/set', manifest })
    state = batchSessionReducer(state, {
      type: 'job/submit',
      jobId: 'batch-job-1',
      runIdsByCase: { [catalog[2].id]: 'run-1' },
    })

    expect(state.job).toEqual({
      id: 'batch-job-1',
      manifestHash: manifest.hash,
      runIdsByCase: { [catalog[2].id]: 'run-1' },
    })
    expect(Object.isFrozen(state.job)).toBe(true)
    expect(Object.isFrozen(state.job?.runIdsByCase)).toBe(true)
    expect(
      batchSessionReducer(state, {
        type: 'config/update',
        field: 'smoothing',
        value: 0.7,
      }),
    ).toBe(state)
  })

  it('rejects malformed IDs, missing ready cases, blocked cases, and duplicate run IDs', () => {
    let state = batchSessionReducer(createInitialBatchSessionState(), {
      type: 'selection/selectAll',
      caseIds: [catalog[0].id, catalog[2].id, catalog[3].id],
    })
    const manifest = createBatchManifest({
      workspaceState: createExplicitBlockedState(),
      selectedCaseIds: state.selectedCaseIds,
      modelVersion: state.modelVersion,
      config: state.config,
    })
    state = batchSessionReducer(state, { type: 'manifest/set', manifest })

    const invalidSubmissions = [
      { jobId: '', runIdsByCase: { [catalog[2].id]: 'run-1', [catalog[3].id]: 'run-2' } },
      { jobId: '   ', runIdsByCase: { [catalog[2].id]: 'run-1', [catalog[3].id]: 'run-2' } },
      { jobId: 'batch-job-1', runIdsByCase: { [catalog[2].id]: 'run-1' } },
      {
        jobId: 'batch-job-1',
        runIdsByCase: {
          [catalog[0].id]: 'run-blocked',
          [catalog[2].id]: 'run-1',
          [catalog[3].id]: 'run-2',
        },
      },
      {
        jobId: 'batch-job-1',
        runIdsByCase: {
          [catalog[2].id]: 'same-run',
          [catalog[3].id]: 'same-run',
        },
      },
      {
        jobId: 'batch-job-1',
        runIdsByCase: {
          [catalog[2].id]: 'run-1',
          [catalog[3].id]: '',
        },
      },
    ] as const

    for (const submission of invalidSubmissions) {
      expect(
        batchSessionReducer(state, { type: 'job/submit', ...submission }),
      ).toBe(state)
    }
    expect(state.job).toBeUndefined()
  })

  it('makes exact submit replay idempotent and prohibits conflicting replacement', () => {
    let state = batchSessionReducer(createInitialBatchSessionState(), {
      type: 'selection/selectAll',
      caseIds: [catalog[2].id, catalog[3].id],
    })
    const manifest = createBatchManifest({
      workspaceState: createInitialWorkspaceState(),
      selectedCaseIds: state.selectedCaseIds,
      modelVersion: state.modelVersion,
      config: state.config,
    })
    state = batchSessionReducer(state, { type: 'manifest/set', manifest })
    const runIdsByCase = {
      [catalog[2].id]: 'run-1',
      [catalog[3].id]: 'run-2',
    }
    const submitted = batchSessionReducer(state, {
      type: 'job/submit',
      jobId: 'batch-job-1',
      runIdsByCase,
    })

    expect(
      batchSessionReducer(submitted, {
        type: 'job/submit',
        jobId: 'batch-job-1',
        runIdsByCase: { ...runIdsByCase },
      }),
    ).toBe(submitted)
    expect(
      batchSessionReducer(submitted, {
        type: 'job/submit',
        jobId: 'batch-job-2',
        runIdsByCase,
      }),
    ).toBe(submitted)
    expect(
      batchSessionReducer(submitted, {
        type: 'job/submit',
        jobId: 'batch-job-1',
        runIdsByCase: {
          [catalog[2].id]: 'run-3',
          [catalog[3].id]: 'run-4',
        },
      }),
    ).toBe(submitted)
    expect(submitted.job).toEqual({
      id: 'batch-job-1',
      manifestHash: manifest.hash,
      runIdsByCase,
    })
  })
})
