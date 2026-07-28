import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'
import { getWorkbenchAsset } from './catalog'
import type { BatchManifest } from './batchManifest'
import type { InferenceConfiguration, MockModelVersion } from './types'

export type BatchSessionJob = {
  readonly id: string
  readonly manifestHash: string
  readonly runIdsByCase: Readonly<Partial<Record<WorkbenchAssetId, string>>>
}

export type BatchSessionState = {
  readonly selectedCaseIds: readonly WorkbenchAssetId[]
  readonly modelVersion: MockModelVersion
  readonly config: Readonly<InferenceConfiguration>
  readonly manifest?: BatchManifest
  readonly job?: BatchSessionJob
}

export type BatchSessionAction =
  | { readonly type: 'session/reset' }
  | { readonly type: 'selection/toggle'; readonly caseId: string }
  | { readonly type: 'selection/selectAll'; readonly caseIds: readonly string[] }
  | { readonly type: 'selection/clear' }
  | {
      readonly type: 'config/update'
      readonly field: keyof InferenceConfiguration
      readonly value: number
    }
  | { readonly type: 'model/update'; readonly modelVersion: MockModelVersion }
  | { readonly type: 'manifest/set'; readonly manifest: BatchManifest }
  | {
      readonly type: 'job/submit'
      readonly jobId: string
      readonly runIdsByCase: Readonly<Record<string, string>>
    }

const INITIAL_CONFIG = Object.freeze({ threshold: 0.45, smoothing: 0.3 })

export function createInitialBatchSessionState(): BatchSessionState {
  return {
    selectedCaseIds: Object.freeze([]),
    modelVersion: 'mock-salience-v0.4',
    config: INITIAL_CONFIG,
  }
}

function exactCanonicalIds(caseIds: readonly string[]): readonly WorkbenchAssetId[] | undefined {
  if (new Set(caseIds).size !== caseIds.length) return undefined
  const canonical = caseIds.map((caseId) => getWorkbenchAsset(caseId)?.id)
  return canonical.every((caseId) => caseId !== undefined)
    ? Object.freeze(canonical as WorkbenchAssetId[])
    : undefined
}

export function manifestMatchesBatchDraft(state: BatchSessionState): boolean {
  const manifest = state.manifest
  if (!manifest) return false
  return (
    manifest.modelVersion === state.modelVersion &&
    manifest.config.threshold === state.config.threshold &&
    manifest.config.smoothing === state.config.smoothing &&
    manifest.items.length === state.selectedCaseIds.length &&
    manifest.items.every((item, index) => item.caseId === state.selectedCaseIds[index])
  )
}

function sameRunMapping(
  first: Readonly<Record<string, string>>,
  second: Readonly<Record<string, string>>,
): boolean {
  const firstKeys = Object.keys(first).sort()
  const secondKeys = Object.keys(second).sort()
  return (
    firstKeys.length === secondKeys.length &&
    firstKeys.every(
      (caseId, index) =>
        caseId === secondKeys[index] && first[caseId] === second[caseId],
    )
  )
}

export function isExactBatchJobReplay(
  state: BatchSessionState,
  jobId: string,
  runIdsByCase: Readonly<Record<string, string>>,
): boolean {
  return Boolean(
    state.job &&
      state.job.id === jobId &&
      state.job.manifestHash === state.manifest?.hash &&
      sameRunMapping(state.job.runIdsByCase, runIdsByCase),
  )
}

export function isValidBatchJobSubmission(
  state: BatchSessionState,
  jobId: string,
  runIdsByCase: Readonly<Record<string, string>>,
): boolean {
  if (
    !state.manifest ||
    !manifestMatchesBatchDraft(state) ||
    !/^batch-job-[1-9]\d*$/.test(jobId)
  ) {
    return false
  }

  const readyCaseIds = state.manifest.items
    .filter((item) => item.preflight === 'ready')
    .map((item) => item.caseId)
    .sort()
  const submittedCaseIds = Object.keys(runIdsByCase).sort()
  if (
    readyCaseIds.length === 0 ||
    readyCaseIds.length !== submittedCaseIds.length ||
    !readyCaseIds.every((caseId, index) => caseId === submittedCaseIds[index])
  ) {
    return false
  }

  const runIds = submittedCaseIds.map((caseId) => runIdsByCase[caseId])
  return (
    runIds.every(
      (runId): runId is string =>
        typeof runId === 'string' &&
        runId.trim().length > 0 &&
        runId !== jobId,
    ) && new Set(runIds).size === runIds.length
  )
}

export function batchSessionReducer(
  state: BatchSessionState,
  action: BatchSessionAction,
): BatchSessionState {
  if (
    state.job &&
    action.type !== 'job/submit' &&
    action.type !== 'session/reset'
  ) {
    return state
  }

  switch (action.type) {
    case 'session/reset':
      return createInitialBatchSessionState()
    case 'selection/toggle': {
      const canonical = getWorkbenchAsset(action.caseId)?.id
      if (!canonical) return state
      const selected = state.selectedCaseIds.includes(canonical)
        ? state.selectedCaseIds.filter((caseId) => caseId !== canonical)
        : [...state.selectedCaseIds, canonical]
      return { ...state, selectedCaseIds: Object.freeze(selected) }
    }
    case 'selection/selectAll': {
      const caseIds = exactCanonicalIds(action.caseIds)
      return caseIds ? { ...state, selectedCaseIds: caseIds } : state
    }
    case 'selection/clear':
      return { ...state, selectedCaseIds: Object.freeze([]) }
    case 'config/update':
      return {
        ...state,
        config: Object.freeze({ ...state.config, [action.field]: action.value }),
      }
    case 'model/update':
      return { ...state, modelVersion: action.modelVersion }
    case 'manifest/set':
      return { ...state, manifest: action.manifest }
    case 'job/submit': {
      if (isExactBatchJobReplay(state, action.jobId, action.runIdsByCase)) return state
      if (state.job) return state
      const manifest = state.manifest
      if (
        !manifest ||
        !isValidBatchJobSubmission(state, action.jobId, action.runIdsByCase)
      ) {
        return state
      }
      const runIdsByCase = Object.freeze({ ...action.runIdsByCase })
      return {
        ...state,
        job: Object.freeze({
          id: action.jobId,
          manifestHash: manifest.hash,
          runIdsByCase,
        }),
      }
    }
  }
}
