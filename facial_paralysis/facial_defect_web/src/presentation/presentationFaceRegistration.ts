import type { PatientFaceRegistration } from '../patientWorkflow/types'
import type {
  PresentationSubjectId,
  PresentationTimepoint,
} from '../data/presentationDemoAssets'
import registrationData from './presentationFaceRegistrationData.json'

type RegistrationBySubject = Readonly<
  Record<
    PresentationSubjectId,
    Readonly<Record<PresentationTimepoint, Readonly<PatientFaceRegistration>>>
  >
>

// Generated once, on-device, from each exact hash-bound synthetic image.
// Keeping these snapshots in the presentation payload avoids loading
// MediaPipe, its model, or camera permissions in the shareable demo.
export const registrationBySubject =
  registrationData as RegistrationBySubject
