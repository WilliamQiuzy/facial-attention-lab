import type { SessionMediaHandle } from './types'
import { isSessionMediaHandle } from './validation'

export type SessionMediaVaultEntry = Readonly<{
  readonly blob: Blob
  readonly previewUrl: string
}>

export type SessionMediaVaultObjectUrlApi = {
  readonly createObjectURL: (blob: Blob) => string
  readonly revokeObjectURL: (url: string) => void
}

const browserObjectUrlApi: SessionMediaVaultObjectUrlApi = {
  createObjectURL: (blob) => URL.createObjectURL(blob),
  revokeObjectURL: (url) => URL.revokeObjectURL(url),
}

function requireMediaHandle(
  handle: SessionMediaHandle,
): SessionMediaHandle {
  if (!isSessionMediaHandle(handle)) {
    throw new TypeError('A valid session media handle is required.')
  }
  return handle
}

export class SessionMediaVault {
  readonly #entries = new Map<
    SessionMediaHandle,
    SessionMediaVaultEntry
  >()

  constructor(
    private readonly objectUrls: SessionMediaVaultObjectUrlApi =
      browserObjectUrlApi,
  ) {}

  set(
    handle: SessionMediaHandle,
    blob: Blob,
  ): SessionMediaVaultEntry {
    const validHandle = requireMediaHandle(handle)
    const previous = this.#entries.get(validHandle)
    const entry = Object.freeze({
      blob,
      previewUrl: this.objectUrls.createObjectURL(blob),
    })

    this.#entries.set(validHandle, entry)
    if (previous) {
      this.objectUrls.revokeObjectURL(previous.previewUrl)
    }
    return entry
  }

  has(handle: SessionMediaHandle): boolean {
    return this.#entries.has(requireMediaHandle(handle))
  }

  get(
    handle: SessionMediaHandle,
  ): SessionMediaVaultEntry | undefined {
    return this.#entries.get(requireMediaHandle(handle))
  }

  delete(handle: SessionMediaHandle): boolean {
    const validHandle = requireMediaHandle(handle)
    const entry = this.#entries.get(validHandle)
    if (!entry) return false

    this.#entries.delete(validHandle)
    this.objectUrls.revokeObjectURL(entry.previewUrl)
    return true
  }

  clear(): void {
    const entries = [...this.#entries.values()]
    this.#entries.clear()
    for (const entry of entries) {
      this.objectUrls.revokeObjectURL(entry.previewUrl)
    }
  }
}
