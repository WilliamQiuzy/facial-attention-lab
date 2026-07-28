import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'
import { getWorkbenchAsset } from './catalog'
import { hashConfiguration, validateNormalizedRoi } from './mockEngine'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import type {
  InferenceConfiguration,
  MockModelVersion,
  NormalizedRoi,
  RoiStatus,
  WorkspaceState,
} from './types'
import { WorkbenchError } from './types'

export type BatchManifestInput = {
  readonly workspaceState: WorkspaceState
  readonly selectedCaseIds: readonly string[]
  readonly excludedCaseIds?: readonly string[]
  readonly modelVersion: MockModelVersion
  readonly config: InferenceConfiguration
}

export type BatchManifestBlocker =
  | 'ROI_NOT_APPROVED'
  | 'ROI_BINDING_MISMATCH'
  | 'FULL_IMAGE_SOURCE_BINDING_REQUIRED'

export type BatchManifestItem = {
  readonly caseId: WorkbenchAssetId
  readonly assetId: WorkbenchAssetId
  readonly assetSha256: string
  readonly label: string
  readonly relationship: 'unpaired_demo'
  readonly preflight: 'ready' | 'blocked'
  readonly blocker?: BatchManifestBlocker
  readonly roiStatus: RoiStatus | 'missing'
  readonly roiId?: string
  readonly roiVersion?: number
  readonly roiGeometry?: Readonly<NormalizedRoi>
  readonly roiAuthorId?: string
  readonly roiReviewerId?: string
}

export type BatchManifest = {
  readonly hash: string
  readonly origin: 'synthetic_catalog_session'
  readonly persistence: 'memory_only'
  readonly modelMode: 'mock_only'
  readonly modelVersion: MockModelVersion
  readonly config: Readonly<InferenceConfiguration>
  readonly items: readonly BatchManifestItem[]
  readonly excludedItems: readonly BatchManifestItem[]
}

export type BatchManifestAudit = {
  readonly valid: boolean
  readonly staleCaseIds: readonly WorkbenchAssetId[]
}

const MOCK_MODELS = new Set<MockModelVersion>([
  'mock-salience-v0.3',
  'mock-salience-v0.4',
])

function fail(
  reason: ConstructorParameters<typeof WorkbenchError>[0]['reason'],
  message: string,
  field?: string,
): never {
  throw new WorkbenchError({ reason, message, ...(field ? { field } : {}) })
}

function stableSerialize(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(',')}]`

  const record = value as Record<string, unknown>
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
    .join(',')}}`
}

function hashText(value: string): string {
  let first = 0x811c9dc5
  let second = 0x9e3779b9

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    first ^= code
    first = Math.imul(first, 0x01000193)
    second ^= code + index
    second = Math.imul(second, 0x85ebca6b)
  }

  return `${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0)
    .toString(16)
    .padStart(8, '0')}`
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype
  )
}

function hasExactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
): boolean {
  const actual = Reflect.ownKeys(value)
  const expected = [...keys].sort()
  return (
    actual.length === expected.length &&
    actual.every((key) => typeof key === 'string') &&
    (actual as string[]).sort().every((key, index) => key === expected[index]) &&
    actual.every((key) => {
      const descriptor = Reflect.getOwnPropertyDescriptor(value, key)
      return descriptor?.enumerable === true && 'value' in descriptor
    })
  )
}

function hasOnlyEnumerableDataKeys(value: Record<string, unknown>): boolean {
  return Reflect.ownKeys(value).every((key) => {
    if (typeof key !== 'string') return false
    const descriptor = Reflect.getOwnPropertyDescriptor(value, key)
    return descriptor?.enumerable === true && 'value' in descriptor
  })
}

function hasExactArrayShape(value: readonly unknown[]): boolean {
  if (Object.getPrototypeOf(value) !== Array.prototype) return false
  const actual = Reflect.ownKeys(value)
  const expected = [
    ...value.map((_, index) => String(index)),
    'length',
  ]
  if (
    actual.length !== expected.length ||
    !actual.every((key) => typeof key === 'string' && expected.includes(key))
  ) {
    return false
  }
  return actual.every((key) => {
    const descriptor = Reflect.getOwnPropertyDescriptor(value, key)
    return (
      descriptor !== undefined &&
      'value' in descriptor &&
      (key === 'length' ? descriptor.enumerable === false : descriptor.enumerable === true)
    )
  })
}

function hasStrictNestedShape(value: Record<string, unknown>): boolean {
  if (!isPlainRecord(value) || !hasOnlyEnumerableDataKeys(value)) return false
  const geometry = value.roiGeometry
  return (
    geometry === undefined ||
    (isPlainRecord(geometry) &&
      hasExactKeys(geometry, ['x', 'y', 'width', 'height']))
  )
}

function invalidAudit(caseIds: readonly string[] = []): BatchManifestAudit {
  const canonicalIds = caseIds.flatMap((caseId) => {
    const asset = getWorkbenchAsset(caseId)
    return asset ? [asset.id] : []
  })
  return Object.freeze({
    valid: false,
    staleCaseIds: Object.freeze([...new Set(canonicalIds)]),
  })
}

function freezeGeometry(geometry: NormalizedRoi | undefined) {
  return geometry
    ? Object.freeze({
        x: geometry.x,
        y: geometry.y,
        width: geometry.width,
        height: geometry.height,
      })
    : undefined
}

function createManifestItem(
  state: WorkspaceState,
  caseId: string,
): BatchManifestItem {
  const asset = getWorkbenchAsset(caseId)
  if (!asset) fail('UNKNOWN_CASE', `Unknown workbench case: ${caseId}.`, 'caseId')
  const roi = state.roisByCase[asset.id]
  const bound =
    roi !== undefined &&
    roi.caseId === asset.id &&
    roi.assetId === asset.id &&
    validateNormalizedRoi(roi.geometry)
  const ready = isVerifiedFullImageSourceBinding(asset, roi)

  const item: BatchManifestItem = {
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    label: asset.label,
    relationship: 'unpaired_demo',
    preflight: ready ? 'ready' : 'blocked',
    ...(!ready
      ? {
          blocker: bound
            ? ('FULL_IMAGE_SOURCE_BINDING_REQUIRED' as const)
            : ('ROI_BINDING_MISMATCH' as const),
        }
      : {}),
    roiStatus: roi?.status ?? 'missing',
    ...(roi
      ? {
          roiId: roi.id,
          roiVersion: roi.version,
          roiGeometry: freezeGeometry(roi.geometry),
          roiAuthorId: roi.authorId,
          ...(roi.reviewerId ? { roiReviewerId: roi.reviewerId } : {}),
        }
      : {}),
  }
  return Object.freeze(item)
}

export function createBatchManifest(input: BatchManifestInput): BatchManifest {
  if (input.selectedCaseIds.length === 0) {
    fail('INVALID_OPERATIONAL_ID', 'Select at least one canonical synthetic case.', 'caseIds')
  }
  if (new Set(input.selectedCaseIds).size !== input.selectedCaseIds.length) {
    fail('INVALID_OPERATIONAL_ID', 'Batch selections cannot contain duplicate case IDs.', 'caseIds')
  }
  const excludedCaseIds = input.excludedCaseIds ?? []
  if (new Set(excludedCaseIds).size !== excludedCaseIds.length) {
    fail('INVALID_OPERATIONAL_ID', 'Batch exclusions cannot contain duplicate case IDs.', 'excludedCaseIds')
  }
  const selectedCaseIds = new Set(input.selectedCaseIds)
  if (excludedCaseIds.some((caseId) => selectedCaseIds.has(caseId))) {
    fail(
      'INVALID_OPERATIONAL_ID',
      'A case cannot be both selected and excluded in one batch review.',
      'excludedCaseIds',
    )
  }
  if (!MOCK_MODELS.has(input.modelVersion)) {
    fail('UNKNOWN_MODEL', `Unknown mock model: ${String(input.modelVersion)}.`, 'modelVersion')
  }

  hashConfiguration(input.config)
  const config = Object.freeze({
    threshold: input.config.threshold,
    smoothing: input.config.smoothing,
  })
  const items = Object.freeze(
    input.selectedCaseIds.map((caseId) => createManifestItem(input.workspaceState, caseId)),
  )
  const excludedItems = Object.freeze(
    excludedCaseIds.map((caseId) => createManifestItem(input.workspaceState, caseId)),
  )
  if (excludedItems.some((item) => item.preflight !== 'blocked')) {
    fail(
      'INVALID_OPERATIONAL_ID',
      'Only cases blocked by the current ROI gate can be recorded as exclusions.',
      'excludedCaseIds',
    )
  }
  const content = {
    origin: 'synthetic_catalog_session' as const,
    persistence: 'memory_only' as const,
    modelMode: 'mock_only' as const,
    modelVersion: input.modelVersion,
    config,
    items,
    excludedItems,
  }

  return Object.freeze({
    hash: `manifest_${hashText(stableSerialize(content))}`,
    ...content,
  })
}

export function auditBatchManifest(
  manifest: BatchManifest,
  state: WorkspaceState,
): BatchManifestAudit {
  try {
    if (!isPlainRecord(manifest)) return invalidAudit()
    if (
      !hasExactKeys(manifest, [
        'hash',
        'origin',
        'persistence',
        'modelMode',
        'modelVersion',
        'config',
        'items',
        'excludedItems',
      ]) ||
      !isPlainRecord(manifest.config) ||
      !hasExactKeys(manifest.config, ['threshold', 'smoothing']) ||
      !Array.isArray(manifest.items) ||
      !Array.isArray(manifest.excludedItems) ||
      !hasExactArrayShape(manifest.items) ||
      !hasExactArrayShape(manifest.excludedItems)
    ) {
      return invalidAudit()
    }

    const reviewedItems = [...manifest.items, ...manifest.excludedItems]
    if (
      reviewedItems.some(
        (item) => !hasStrictNestedShape(item) || typeof item.caseId !== 'string',
      )
    ) {
      return invalidAudit()
    }
    const selectedCaseIds = manifest.items.map((item) => item.caseId)
    const excludedCaseIds = manifest.excludedItems.map((item) => item.caseId)
    const reviewedCaseIds = [...selectedCaseIds, ...excludedCaseIds]
    if (
      selectedCaseIds.length === 0 ||
      new Set(selectedCaseIds).size !== selectedCaseIds.length ||
      new Set(excludedCaseIds).size !== excludedCaseIds.length ||
      excludedCaseIds.some((caseId) => selectedCaseIds.includes(caseId))
    ) {
      return invalidAudit(reviewedCaseIds)
    }

    const canonical = createBatchManifest({
      workspaceState: state,
      selectedCaseIds,
      excludedCaseIds,
      modelVersion: manifest.modelVersion,
      config: manifest.config,
    })
    const staleCaseIds = reviewedItems.flatMap((item, index) => {
      const canonicalItem =
        index < manifest.items.length
          ? canonical.items[index]
          : canonical.excludedItems[index - manifest.items.length]
      return stableSerialize(item) === stableSerialize(canonicalItem)
        ? []
        : [item.caseId]
    })
    const exact = stableSerialize(manifest) === stableSerialize(canonical)
    if (!exact && staleCaseIds.length === 0) {
      return invalidAudit(reviewedCaseIds)
    }
    return Object.freeze({
      valid: exact,
      staleCaseIds: Object.freeze([...new Set(staleCaseIds)]),
    })
  } catch {
    let caseIds: string[] = []
    try {
      if (isPlainRecord(manifest)) {
        const items = Array.isArray(manifest.items) ? manifest.items : []
        const excludedItems = Array.isArray(manifest.excludedItems)
          ? manifest.excludedItems
          : []
        caseIds = [...items, ...excludedItems].flatMap((item) =>
          isPlainRecord(item) && typeof item.caseId === 'string'
            ? [item.caseId]
            : [],
        )
      }
    } catch {
      caseIds = []
    }
    return invalidAudit(caseIds)
  }
}
