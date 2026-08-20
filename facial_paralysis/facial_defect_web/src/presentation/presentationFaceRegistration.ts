import type { PatientFaceRegistration } from '../patientWorkflow/types'
import registrationData from './presentationFaceRegistrationData.json'

type RegistrationByTimepoint = Readonly<{
  preoperative: Readonly<PatientFaceRegistration>
  postoperative: Readonly<PatientFaceRegistration>
}>

// Generated once, on-device, from each exact hash-bound synthetic image.
// Keeping these snapshots in the presentation payload avoids loading
// MediaPipe, its model, or camera permissions in the shareable demo.
export const registrationByTimepoint =
  registrationData as RegistrationByTimepoint
