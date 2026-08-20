import { presentationDemoAssets } from './data/presentationDemoAssets'
import { validateCaptureFile } from './patientWorkflow/captureFile'
import { detectPatientFaceRegistration } from './patientWorkflow/onDeviceFaceRegistration'
import type {
  PatientFaceRegistration,
  PatientTimepoint,
} from './patientWorkflow/types'

const output = document.querySelector<HTMLPreElement>(
  '[data-testid="presentation-registrations"]',
)

async function registrationFor(
  timepoint: Extract<
    PatientTimepoint,
    'preoperative' | 'postoperative'
  >,
): Promise<PatientFaceRegistration> {
  const asset = presentationDemoAssets[timepoint]
  const response = await fetch(asset.url)
  if (!response.ok) {
    throw new Error(`IMAGE_LOAD_FAILED:${timepoint}`)
  }
  const media = await response.blob()
  const prepared = await validateCaptureFile(media)
  if (!prepared.ok) {
    throw new Error(
      `IMAGE_VALIDATION_FAILED:${timepoint}:${prepared.error.code}`,
    )
  }
  if (prepared.value.metadata.sha256 !== asset.sha256) {
    throw new Error(`IMAGE_HASH_MISMATCH:${timepoint}`)
  }

  return detectPatientFaceRegistration({
    media: prepared.value.vaultMedia,
    captureSha256: prepared.value.metadata.sha256,
    sourceWidth: prepared.value.metadata.width,
    sourceHeight: prepared.value.metadata.height,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
  })
}

async function extract() {
  if (!output) throw new Error('OUTPUT_UNAVAILABLE')
  try {
    const [preoperative, postoperative] = await Promise.all([
      registrationFor('preoperative'),
      registrationFor('postoperative'),
    ])
    output.textContent = JSON.stringify(
      { preoperative, postoperative },
      null,
      2,
    )
    output.dataset.status = 'ready'
  } catch (error) {
    output.textContent =
      error instanceof Error ? error.message : 'EXTRACTION_FAILED'
    output.dataset.status = 'failed'
  }
}

void extract()
