import {
  presentationDemoAssets,
  presentationSubjectIds,
  type PresentationSubjectId,
  type PresentationTimepoint,
} from './data/presentationDemoAssets'
import { validateCaptureFile } from './patientWorkflow/captureFile'
import { detectPatientFaceRegistration } from './patientWorkflow/onDeviceFaceRegistration'
import type { PatientFaceRegistration } from './patientWorkflow/types'

const output = document.querySelector<HTMLPreElement>(
  '[data-testid="presentation-registrations"]',
)

async function registrationFor(
  subjectId: PresentationSubjectId,
  timepoint: PresentationTimepoint,
): Promise<PatientFaceRegistration> {
  const asset = presentationDemoAssets[subjectId][timepoint]
  const response = await fetch(asset.url)
  if (!response.ok) {
    throw new Error(`IMAGE_LOAD_FAILED:${subjectId}:${timepoint}`)
  }
  const media = await response.blob()
  const prepared = await validateCaptureFile(media)
  if (!prepared.ok) {
    throw new Error(
      `IMAGE_VALIDATION_FAILED:${subjectId}:${timepoint}:${prepared.error.code}`,
    )
  }
  if (prepared.value.metadata.sha256 !== asset.sha256) {
    throw new Error(`IMAGE_HASH_MISMATCH:${subjectId}:${timepoint}`)
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
    const pairs = await Promise.all(
      presentationSubjectIds.map(async (subjectId) => {
        const [preoperative, postoperative] = await Promise.all([
          registrationFor(subjectId, 'preoperative'),
          registrationFor(subjectId, 'postoperative'),
        ])
        return [subjectId, { preoperative, postoperative }] as const
      }),
    )
    output.textContent = JSON.stringify(
      Object.fromEntries(pairs),
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
