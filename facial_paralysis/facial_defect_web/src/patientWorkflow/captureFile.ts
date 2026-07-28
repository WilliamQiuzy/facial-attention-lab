import type { CaptureAsset } from './types'

export const CAPTURE_FILE_MAX_BYTES = 12 * 1024 * 1024
export const CAPTURE_FILE_MIN_DIMENSION = 640

const SUPPORTED_CAPTURE_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
] as const satisfies readonly CaptureAsset['mimeType'][]

export type CaptureFileMetadata = Readonly<
  Pick<
    CaptureAsset,
    'sha256' | 'mimeType' | 'sizeBytes' | 'width' | 'height'
  >
>

export type PreparedCaptureFile = Readonly<{
  readonly metadata: CaptureFileMetadata
  /**
   * A filename-free Blob prepared for SessionMediaVault. The caller must
   * retain only its opaque vault handle in reducer state.
   */
  readonly vaultMedia: Blob
}>

export type CaptureFileErrorCode =
  | 'UNSUPPORTED_TYPE'
  | 'EMPTY_FILE'
  | 'FILE_TOO_LARGE'
  | 'CONTENT_TYPE_MISMATCH'
  | 'DECODE_FAILED'
  | 'RESOLUTION_TOO_LOW'
  | 'FILE_READ_FAILED'
  | 'HASH_FAILED'
  | 'PROCESSING_FAILED'

export class CaptureFileValidationError extends Error {
  readonly name = 'CaptureFileValidationError'

  constructor(
    readonly code: CaptureFileErrorCode,
    message: string,
  ) {
    super(message)
  }
}

export type CaptureFileValidationResult =
  | {
      readonly ok: true
      readonly value: PreparedCaptureFile
    }
  | {
      readonly ok: false
      readonly error: CaptureFileValidationError
    }

export type DecodedImageDimensions = Readonly<{
  readonly width: number
  readonly height: number
}>

export type CaptureFileDependencies = {
  readonly decodeDimensions: (
    media: Blob,
  ) => Promise<DecodedImageDimensions>
  readonly sha256Hex: (bytes: ArrayBuffer) => Promise<string>
}

async function decodeDimensionsWithBrowser(
  media: Blob,
): Promise<DecodedImageDimensions> {
  if (typeof createImageBitmap !== 'function') {
    throw new Error('Browser image decoding is unavailable.')
  }

  const bitmap = await createImageBitmap(media)
  try {
    return { width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}

async function sha256HexWithWebCrypto(
  bytes: ArrayBuffer,
): Promise<string> {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) {
    throw new Error('Web Crypto SHA-256 is unavailable.')
  }

  const digest = await subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('')
}

const browserDependencies: CaptureFileDependencies = {
  decodeDimensions: decodeDimensionsWithBrowser,
  sha256Hex: sha256HexWithWebCrypto,
}

function failure(
  code: CaptureFileErrorCode,
  message: string,
): CaptureFileValidationResult {
  return {
    ok: false,
    error: new CaptureFileValidationError(code, message),
  }
}

function isSupportedMimeType(
  type: string,
): type is CaptureAsset['mimeType'] {
  return SUPPORTED_CAPTURE_MIME_TYPES.some(
    (supported) => supported === type,
  )
}

function areDecodedDimensionsValid(
  dimensions: DecodedImageDimensions,
): boolean {
  return (
    Number.isInteger(dimensions.width) &&
    Number.isInteger(dimensions.height) &&
    dimensions.width > 0 &&
    dimensions.height > 0
  )
}

function hasExpectedContentSignature(
  bytes: ArrayBuffer,
  mimeType: CaptureAsset['mimeType'],
): boolean {
  const view = new Uint8Array(bytes)

  if (mimeType === 'image/jpeg') {
    return (
      view.length >= 3 &&
      view[0] === 0xff &&
      view[1] === 0xd8 &&
      view[2] === 0xff
    )
  }

  if (mimeType === 'image/png') {
    const pngSignature = [
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ]
    return (
      view.length >= pngSignature.length &&
      pngSignature.every((byte, index) => view[index] === byte)
    )
  }

  return (
    view.length >= 12 &&
    view[0] === 0x52 &&
    view[1] === 0x49 &&
    view[2] === 0x46 &&
    view[3] === 0x46 &&
    view[8] === 0x57 &&
    view[9] === 0x45 &&
    view[10] === 0x42 &&
    view[11] === 0x50
  )
}

export async function validateCaptureFile(
  file: Blob,
  dependencyOverrides: Partial<CaptureFileDependencies> = {},
): Promise<CaptureFileValidationResult> {
  try {
    const dependencies: CaptureFileDependencies = {
      ...browserDependencies,
      ...dependencyOverrides,
    }

    if (!isSupportedMimeType(file.type)) {
      return failure(
        'UNSUPPORTED_TYPE',
        'Choose a JPEG, PNG, or WebP image. Other file types are not supported.',
      )
    }
    if (file.size === 0) {
      return failure(
        'EMPTY_FILE',
        'The selected image is empty. Choose a JPEG, PNG, or WebP image with visible content.',
      )
    }
    if (file.size > CAPTURE_FILE_MAX_BYTES) {
      return failure(
        'FILE_TOO_LARGE',
        'The selected image is larger than 12 MiB. Choose a smaller image.',
      )
    }

    let bytes: ArrayBuffer
    try {
      bytes = await file.arrayBuffer()
    } catch {
      return failure(
        'FILE_READ_FAILED',
        'The image could not be read. Choose the file again or select another image.',
      )
    }
    if (!hasExpectedContentSignature(bytes, file.type)) {
      return failure(
        'CONTENT_TYPE_MISMATCH',
        'The file contents do not match the selected image type. Choose an original JPEG, PNG, or WebP image.',
      )
    }

    let dimensions: DecodedImageDimensions
    try {
      dimensions = await dependencies.decodeDimensions(file)
    } catch {
      return failure(
        'DECODE_FAILED',
        'The image could not be decoded. Choose an uncorrupted JPEG, PNG, or WebP image.',
      )
    }

    if (!areDecodedDimensionsValid(dimensions)) {
      return failure(
        'DECODE_FAILED',
        'The image could not be decoded. Choose an uncorrupted JPEG, PNG, or WebP image.',
      )
    }
    if (
      dimensions.width < CAPTURE_FILE_MIN_DIMENSION ||
      dimensions.height < CAPTURE_FILE_MIN_DIMENSION
    ) {
      return failure(
        'RESOLUTION_TOO_LOW',
        `The image is ${dimensions.width} × ${dimensions.height} pixels. Choose an image at least 640 × 640 pixels.`,
      )
    }

    const vaultMedia = new Blob([bytes], { type: file.type })
    let sha256: string
    try {
      sha256 = await dependencies.sha256Hex(bytes)
    } catch {
      return failure(
        'HASH_FAILED',
        'The image could not be verified securely. Choose the file again or select another image.',
      )
    }
    if (
      typeof sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/i.test(sha256)
    ) {
      return failure(
        'HASH_FAILED',
        'The image could not be verified securely. Choose the file again or select another image.',
      )
    }

    const metadata: CaptureFileMetadata = Object.freeze({
      mimeType: file.type,
      sizeBytes: file.size,
      width: dimensions.width,
      height: dimensions.height,
      sha256: sha256.toLowerCase(),
    })

    return {
      ok: true,
      value: Object.freeze({ metadata, vaultMedia }),
    }
  } catch {
    return failure(
      'PROCESSING_FAILED',
      'The image could not be prepared. Choose the file again or select another image.',
    )
  }
}
