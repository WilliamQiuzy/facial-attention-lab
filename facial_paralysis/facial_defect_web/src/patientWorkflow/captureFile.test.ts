import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  CAPTURE_FILE_MAX_BYTES,
  CAPTURE_FILE_MIN_DIMENSION,
  CaptureFileValidationError,
  validateCaptureFile,
  type CaptureFileDependencies,
} from './captureFile'

const TEST_SHA256 = 'ab'.repeat(32)
const IMAGE_SIGNATURES = {
  'image/jpeg': Uint8Array.of(0xff, 0xd8, 0xff),
  'image/png': Uint8Array.of(
    0x89,
    0x50,
    0x4e,
    0x47,
    0x0d,
    0x0a,
    0x1a,
    0x0a,
  ),
  'image/webp': Uint8Array.of(
    0x52,
    0x49,
    0x46,
    0x46,
    0x00,
    0x00,
    0x00,
    0x00,
    0x57,
    0x45,
    0x42,
    0x50,
  ),
} as const

type SupportedTestMimeType = keyof typeof IMAGE_SIGNATURES

function validImageBytes(
  mimeType: SupportedTestMimeType,
  size = IMAGE_SIGNATURES[mimeType].length + 4,
): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(size))
  bytes.set(IMAGE_SIGNATURES[mimeType])
  return bytes
}

function copyBytes(
  bytes: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(new ArrayBuffer(bytes.byteLength))
  copy.set(bytes)
  return copy
}

function makeFile(
  contents: BlobPart | Uint8Array<ArrayBufferLike> =
    validImageBytes('image/jpeg'),
  type = 'image/jpeg',
): File {
  const fileContents =
    contents instanceof Uint8Array ? copyBytes(contents) : contents
  return new File([fileContents], 'patient-name-must-not-escape.jpg', {
    type,
  })
}

function createDependencies(
  dimensions = { width: 1_200, height: 900 },
): {
  readonly dependencies: CaptureFileDependencies
  readonly decodeDimensions: ReturnType<typeof vi.fn>
  readonly sha256Hex: ReturnType<typeof vi.fn>
} {
  const decodeDimensions = vi.fn(async () => dimensions)
  const sha256Hex = vi.fn(async () => TEST_SHA256)
  return {
    dependencies: { decodeDimensions, sha256Hex },
    decodeDimensions,
    sha256Hex,
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('validateCaptureFile', () => {
  it.each(['image/jpeg', 'image/png', 'image/webp'] as const)(
    'prepares supported %s media with exact metadata and no filename',
    async (mimeType) => {
      const contents = validImageBytes(mimeType)
      const file = makeFile(contents, mimeType)
      const { dependencies, decodeDimensions, sha256Hex } =
        createDependencies()

      const result = await validateCaptureFile(file, dependencies)

      expect(result.ok).toBe(true)
      if (!result.ok) return
      expect(result.value.metadata).toEqual({
        mimeType,
        sizeBytes: file.size,
        width: 1_200,
        height: 900,
        sha256: TEST_SHA256,
      })
      expect(Object.keys(result.value)).toEqual([
        'metadata',
        'vaultMedia',
      ])
      expect(Object.keys(result.value.metadata).sort()).toEqual([
        'height',
        'mimeType',
        'sha256',
        'sizeBytes',
        'width',
      ])
      expect(result.value.vaultMedia).toBeInstanceOf(Blob)
      expect(result.value.vaultMedia).not.toBeInstanceOf(File)
      expect(result.value.vaultMedia.type).toBe(mimeType)
      expect(result.value.vaultMedia.size).toBe(file.size)
      expect('name' in result.value.vaultMedia).toBe(false)
      expect(Object.isFrozen(result.value)).toBe(true)
      expect(Object.isFrozen(result.value.metadata)).toBe(true)
      expect(decodeDimensions).toHaveBeenCalledOnce()
      expect(decodeDimensions).toHaveBeenCalledWith(file)
      expect(sha256Hex).toHaveBeenCalledOnce()
      const hashedBytes = sha256Hex.mock.calls[0]?.[0] as ArrayBuffer
      expect([...new Uint8Array(hashedBytes)]).toEqual([...contents])
    },
  )

  it('rejects unsupported media types before decoding or hashing', async () => {
    const file = makeFile('gif bytes', 'image/gif')
    const { dependencies, decodeDimensions, sha256Hex } =
      createDependencies()

    const result = await validateCaptureFile(file, dependencies)

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error).toBeInstanceOf(CaptureFileValidationError)
    expect(result.error).toMatchObject({
      code: 'UNSUPPORTED_TYPE',
      message:
        'Choose a JPEG, PNG, or WebP image. Other file types are not supported.',
    })
    expect(decodeDimensions).not.toHaveBeenCalled()
    expect(sha256Hex).not.toHaveBeenCalled()
  })

  it.each([
    {
      label: 'GIF bytes declared as JPEG',
      mimeType: 'image/jpeg',
      bytes: new TextEncoder().encode('GIF89a'),
    },
    {
      label: 'AVIF bytes declared as WebP',
      mimeType: 'image/webp',
      bytes: Uint8Array.of(
        0x00,
        0x00,
        0x00,
        0x18,
        0x66,
        0x74,
        0x79,
        0x70,
        0x61,
        0x76,
        0x69,
        0x66,
      ),
    },
    {
      label: 'arbitrary bytes declared as PNG',
      mimeType: 'image/png',
      bytes: Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8),
    },
  ] as const)(
    'rejects $label before decoding or hashing',
    async ({ mimeType, bytes }) => {
      const { dependencies, decodeDimensions, sha256Hex } =
        createDependencies()

      const result = await validateCaptureFile(
        makeFile(bytes, mimeType),
        dependencies,
      )

      expect(result.ok).toBe(false)
      if (result.ok) return
      expect(result.error).toBeInstanceOf(CaptureFileValidationError)
      expect(result.error).toMatchObject({
        code: 'CONTENT_TYPE_MISMATCH',
        message:
          'The file contents do not match the selected image type. Choose an original JPEG, PNG, or WebP image.',
      })
      expect(decodeDimensions).not.toHaveBeenCalled()
      expect(sha256Hex).not.toHaveBeenCalled()
    },
  )

  it('rejects PNG bytes declared as JPEG', async () => {
    const { dependencies, decodeDimensions, sha256Hex } =
      createDependencies()

    const result = await validateCaptureFile(
      makeFile(validImageBytes('image/png'), 'image/jpeg'),
      dependencies,
    )

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error.code).toBe('CONTENT_TYPE_MISMATCH')
    expect(decodeDimensions).not.toHaveBeenCalled()
    expect(sha256Hex).not.toHaveBeenCalled()
  })

  it.each([
    {
      mimeType: 'image/jpeg',
      bytes: IMAGE_SIGNATURES['image/jpeg'].slice(0, 2),
    },
    {
      mimeType: 'image/png',
      bytes: IMAGE_SIGNATURES['image/png'].slice(0, 7),
    },
    {
      mimeType: 'image/webp',
      bytes: IMAGE_SIGNATURES['image/webp'].slice(0, 11),
    },
  ] as const)(
    'rejects a too-short $mimeType file',
    async ({ mimeType, bytes }) => {
      const { dependencies, decodeDimensions, sha256Hex } =
        createDependencies()

      const result = await validateCaptureFile(
        makeFile(bytes, mimeType),
        dependencies,
      )

      expect(result.ok).toBe(false)
      if (result.ok) return
      expect(result.error).toMatchObject({
        code: 'CONTENT_TYPE_MISMATCH',
        message:
          'The file contents do not match the selected image type. Choose an original JPEG, PNG, or WebP image.',
      })
      expect(decodeDimensions).not.toHaveBeenCalled()
      expect(sha256Hex).not.toHaveBeenCalled()
    },
  )

  it('rejects empty files and files larger than 12 MiB', async () => {
    const { dependencies, decodeDimensions, sha256Hex } =
      createDependencies()

    const empty = await validateCaptureFile(
      makeFile(new Uint8Array(0)),
      dependencies,
    )
    const oversized = await validateCaptureFile(
      makeFile(new Uint8Array(CAPTURE_FILE_MAX_BYTES + 1)),
      dependencies,
    )

    expect(empty.ok).toBe(false)
    expect(oversized.ok).toBe(false)
    if (!empty.ok) {
      expect(empty.error).toMatchObject({
        code: 'EMPTY_FILE',
        message:
          'The selected image is empty. Choose a JPEG, PNG, or WebP image with visible content.',
      })
    }
    if (!oversized.ok) {
      expect(oversized.error).toMatchObject({
        code: 'FILE_TOO_LARGE',
        message:
          'The selected image is larger than 12 MiB. Choose a smaller image.',
      })
    }
    expect(decodeDimensions).not.toHaveBeenCalled()
    expect(sha256Hex).not.toHaveBeenCalled()
  })

  it('accepts a nonempty file at the exact 12 MiB limit', async () => {
    const file = makeFile(
      validImageBytes('image/jpeg', CAPTURE_FILE_MAX_BYTES),
    )
    const { dependencies } = createDependencies({
      width: CAPTURE_FILE_MIN_DIMENSION,
      height: CAPTURE_FILE_MIN_DIMENSION,
    })

    const result = await validateCaptureFile(file, dependencies)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.metadata.sizeBytes).toBe(
      CAPTURE_FILE_MAX_BYTES,
    )
  })

  it('returns a typed decode error for unreadable image bytes', async () => {
    const decodeDimensions = vi.fn(async () => {
      throw new DOMException('Decoder rejected the bytes.')
    })
    const sha256Hex = vi.fn(async () => TEST_SHA256)

    const result = await validateCaptureFile(makeFile(), {
      decodeDimensions,
      sha256Hex,
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error).toBeInstanceOf(CaptureFileValidationError)
    expect(result.error).toMatchObject({
      code: 'DECODE_FAILED',
      message:
        'The image could not be decoded. Choose an uncorrupted JPEG, PNG, or WebP image.',
    })
    expect(sha256Hex).not.toHaveBeenCalled()
  })

  it('returns a typed, non-leaking error when Blob.arrayBuffer rejects', async () => {
    const file = makeFile()
    vi.spyOn(file, 'arrayBuffer').mockRejectedValueOnce(
      new Error('raw local filename and decoder details'),
    )
    const { dependencies, sha256Hex } = createDependencies()

    const result = await validateCaptureFile(file, dependencies)

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error).toBeInstanceOf(CaptureFileValidationError)
    expect(result.error).toMatchObject({
      code: 'FILE_READ_FAILED',
      message:
        'The image could not be read. Choose the file again or select another image.',
    })
    expect(result.error.message).not.toContain('raw local filename')
    expect(sha256Hex).not.toHaveBeenCalled()
  })

  it.each([
    new Error('raw cryptography implementation detail'),
    { patientName: 'must never escape' },
  ])(
    'returns a typed, non-leaking error when the hash dependency rejects',
    async (reason) => {
      const decodeDimensions = vi.fn(async () => ({
        width: 1_200,
        height: 900,
      }))
      const sha256Hex = vi.fn(async () => {
        throw reason
      })

      const result = await validateCaptureFile(makeFile(), {
        decodeDimensions,
        sha256Hex,
      })

      expect(result.ok).toBe(false)
      if (result.ok) return
      expect(result.error).toBeInstanceOf(CaptureFileValidationError)
      expect(result.error).toMatchObject({
        code: 'HASH_FAILED',
        message:
          'The image could not be verified securely. Choose the file again or select another image.',
      })
      expect(result.error.message).not.toMatch(
        /cryptography implementation detail|patientName|must never escape/,
      )
    },
  )

  it('returns a typed hash error when Web Crypto is unavailable', async () => {
    vi.stubGlobal('crypto', {})
    const decodeDimensions = vi.fn(async () => ({
      width: 1_200,
      height: 900,
    }))

    const result = await validateCaptureFile(makeFile(), {
      decodeDimensions,
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error).toMatchObject({
      code: 'HASH_FAILED',
      message:
        'The image could not be verified securely. Choose the file again or select another image.',
    })
    expect(result.error.message).not.toContain('Web Crypto')
  })

  it('fails closed when an injected hash is not SHA-256 hex', async () => {
    const decodeDimensions = vi.fn(async () => ({
      width: 1_200,
      height: 900,
    }))
    const sha256Hex = vi.fn(async () => 'not-a-sha256-digest')

    const result = await validateCaptureFile(makeFile(), {
      decodeDimensions,
      sha256Hex,
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error.code).toBe('HASH_FAILED')
  })

  it.each([
    { width: 639, height: 640 },
    { width: 640, height: 639 },
  ])(
    'rejects decoded dimensions below 640 by 640: $width by $height',
    async (dimensions) => {
      const { dependencies, sha256Hex } =
        createDependencies(dimensions)

      const result = await validateCaptureFile(
        makeFile(),
        dependencies,
      )

      expect(result.ok).toBe(false)
      if (result.ok) return
      expect(result.error).toBeInstanceOf(CaptureFileValidationError)
      expect(result.error).toMatchObject({
        code: 'RESOLUTION_TOO_LOW',
        message: `The image is ${dimensions.width} × ${dimensions.height} pixels. Choose an image at least 640 × 640 pixels.`,
      })
      expect(sha256Hex).not.toHaveBeenCalled()
    },
  )

  it.each([
    { width: 0, height: 640 },
    { width: Number.NaN, height: 640 },
    { width: 640, height: Number.POSITIVE_INFINITY },
  ])(
    'treats invalid decoded dimensions as a decode failure',
    async (dimensions) => {
      const { dependencies, sha256Hex } =
        createDependencies(dimensions)

      const result = await validateCaptureFile(
        makeFile(),
        dependencies,
      )

      expect(result.ok).toBe(false)
      if (result.ok) return
      expect(result.error.code).toBe('DECODE_FAILED')
      expect(sha256Hex).not.toHaveBeenCalled()
    },
  )
})
