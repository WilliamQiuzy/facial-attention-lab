import { afterEach, describe, expect, it, vi } from 'vitest'
import { SessionMediaVault } from './SessionMediaVault'
import { createSessionMediaHandle } from './validation'

function createObjectUrlApi() {
  let nextUrl = 1
  return {
    createObjectURL: vi.fn(
      () => `blob:patient-workflow-${nextUrl++}`,
    ),
    revokeObjectURL: vi.fn(),
  }
}

function assertRequiresBrandedHandle(
  vault: SessionMediaVault,
  media: Blob,
): void {
  // @ts-expect-error Raw strings must not be accepted as media handles.
  vault.set('session-media:unbranded', media)
}

void assertRequiresBrandedHandle

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('SessionMediaVault', () => {
  it('stores media and an owned preview URL behind a branded handle', () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const handle = createSessionMediaHandle('capture_0001')
    const media = new Blob(['jpeg bytes'], { type: 'image/jpeg' })

    const stored = vault.set(handle, media)

    expect(objectUrls.createObjectURL).toHaveBeenCalledOnce()
    expect(objectUrls.createObjectURL).toHaveBeenCalledWith(media)
    expect(stored).toEqual({
      blob: media,
      previewUrl: 'blob:patient-workflow-1',
    })
    expect(Object.isFrozen(stored)).toBe(true)
    expect(vault.has(handle)).toBe(true)
    expect(vault.get(handle)).toBe(stored)
    expect(objectUrls.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('rejects a forged handle before creating or storing a URL', () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const media = new Blob(['image'], { type: 'image/png' })

    expect(() =>
      vault.set('not-a-session-handle' as never, media),
    ).toThrowError('A valid session media handle is required.')
    expect(objectUrls.createObjectURL).not.toHaveBeenCalled()
    expect(objectUrls.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('revokes each URL exactly once when media is replaced and deleted', () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const handle = createSessionMediaHandle('capture_0002')
    const original = new Blob(['original'], { type: 'image/jpeg' })
    const replacement = new Blob(['replacement'], {
      type: 'image/webp',
    })

    vault.set(handle, original)
    const current = vault.set(handle, replacement)

    expect(current.previewUrl).toBe('blob:patient-workflow-2')
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledTimes(1)
    expect(objectUrls.revokeObjectURL).toHaveBeenNthCalledWith(
      1,
      'blob:patient-workflow-1',
    )

    expect(vault.delete(handle)).toBe(true)
    expect(vault.delete(handle)).toBe(false)
    expect(vault.has(handle)).toBe(false)
    expect(vault.get(handle)).toBeUndefined()
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledTimes(2)
    expect(objectUrls.revokeObjectURL).toHaveBeenNthCalledWith(
      2,
      'blob:patient-workflow-2',
    )
  })

  it('clears all active media without revoking stale or unknown URLs', () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const firstHandle = createSessionMediaHandle('capture_0003')
    const secondHandle = createSessionMediaHandle('capture_0004')

    vault.set(
      firstHandle,
      new Blob(['first'], { type: 'image/jpeg' }),
    )
    vault.set(
      secondHandle,
      new Blob(['second'], { type: 'image/png' }),
    )

    vault.clear()
    vault.clear()
    expect(vault.delete(createSessionMediaHandle('capture_unknown'))).toBe(
      false,
    )

    expect(vault.has(firstHandle)).toBe(false)
    expect(vault.has(secondHandle)).toBe(false)
    expect(objectUrls.revokeObjectURL.mock.calls).toEqual([
      ['blob:patient-workflow-1'],
      ['blob:patient-workflow-2'],
    ])
  })

  it('does not use browser persistence, IndexedDB, or the network', () => {
    const storageGet = vi.spyOn(Storage.prototype, 'getItem')
    const storageSet = vi.spyOn(Storage.prototype, 'setItem')
    const storageRemove = vi.spyOn(Storage.prototype, 'removeItem')
    const storageClear = vi.spyOn(Storage.prototype, 'clear')
    const indexedDbOpen = vi.fn()
    const fetchRequest = vi.fn()
    vi.stubGlobal('indexedDB', { open: indexedDbOpen })
    vi.stubGlobal('fetch', fetchRequest)

    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const handle = createSessionMediaHandle('capture_0005')
    vault.set(
      handle,
      new Blob(['session only'], { type: 'image/jpeg' }),
    )
    vault.get(handle)
    vault.clear()

    expect(storageGet).not.toHaveBeenCalled()
    expect(storageSet).not.toHaveBeenCalled()
    expect(storageRemove).not.toHaveBeenCalled()
    expect(storageClear).not.toHaveBeenCalled()
    expect(indexedDbOpen).not.toHaveBeenCalled()
    expect(fetchRequest).not.toHaveBeenCalled()
  })
})
