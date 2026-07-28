import type { PatientRecord } from '../patientWorkflow/types'
import { createPatientId } from '../patientWorkflow/validation'

/**
 * Standalone fictional identities for the session-only clinician workflow.
 * They intentionally contain no visit, capture, asset, or pairing metadata.
 */
export const DEMO_PATIENT_RECORDS: readonly PatientRecord[] = Object.freeze([
  Object.freeze({
    id: createPatientId('patient-demo-001'),
    displayName: 'Synthetic Demo — Facial Paralysis',
    recordNumber: 'DEMO-1001',
    dateOfBirth: '1962-03-14',
    carePathway: 'Facial paralysis',
    recordKind: 'synthetic_demo',
    createdAt: '2026-07-27T12:00:00.000Z',
  }),
  Object.freeze({
    id: createPatientId('patient-demo-002'),
    displayName: 'Synthetic Demo — Facial Reconstruction',
    recordNumber: 'DEMO-1002',
    dateOfBirth: '1975-09-08',
    carePathway: 'Facial reconstruction',
    recordKind: 'synthetic_demo',
    createdAt: '2026-07-27T12:00:01.000Z',
  }),
  Object.freeze({
    id: createPatientId('patient-demo-003'),
    displayName: 'Synthetic Demo — Follow-up Clinic',
    recordNumber: 'DEMO-1003',
    dateOfBirth: '1988-11-22',
    carePathway: 'Follow-up clinic',
    recordKind: 'synthetic_demo',
    createdAt: '2026-07-27T12:00:02.000Z',
  }),
] satisfies readonly PatientRecord[])
