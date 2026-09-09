export const FACES_PROTOCOL_VERSION = 'FACES-v0.01' as const

export type FacesActionId =
  | 'repose'
  | 'eyebrow_raise'
  | 'gentle_eye_closure'
  | 'tight_eye_squeeze'
  | 'relaxed_smile'
  | 'lip_pucker'
  | 'lower_teeth_show'
  | 'reanimated_smile'

export interface FacesProtocolStep {
  readonly id: FacesActionId
  readonly title: string
  readonly shortLabel: string
  readonly instruction: string
  readonly holdSeconds: 3
  readonly optional: boolean
}

export const FACES_PREPARATION = [
  'Welcome and thank you for participating in our study. This recording helps us understand how your facial movements change over time. Please follow the instructions carefully. If you’re at home, try to replicate the same conditions each time you record.',
  'Use an iPad or iPhone with a front-facing camera. Place the device around arm’s length away from you. This minimizes facial distortion caused by wide-angle lenses. Use a tripod or prop the device at eye level.',
  'Your entire face and neck should be clearly visible. Look directly forward, and keep your head and body still. Sit in a well-lit room. Avoid bright windows or lights behind you. Choose a neutral background - plain wall, if possible.',
  'We’ll now go through a series of facial movements. Start with a relaxed face and follow along as each instruction is read aloud. Try not to move your head.',
] as const

export const FACES_PROTOCOL: readonly FacesProtocolStep[] = [
  {
    id: 'repose',
    title: 'Neutral Expression (Repose)',
    shortLabel: 'Repose',
    instruction:
      'Look straight ahead. Keep your face relaxed. Don’t smile or frown. Hold this position for 3 seconds.',
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'eyebrow_raise',
    title: 'Eyebrow Raise',
    shortLabel: 'Brows',
    instruction:
      'Raise your eyebrows as high as you can. Hold for 3 seconds, then relax.',
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'gentle_eye_closure',
    title: 'Gentle Eye Closure',
    shortLabel: 'Gentle close',
    instruction:
      "Close your eyes gently, like you're falling asleep. Hold for 3 seconds, then open your eyes.",
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'tight_eye_squeeze',
    title: 'Tight Eye Squeeze',
    shortLabel: 'Tight close',
    instruction:
      'Close your eyes as tightly as possible, using your facial muscles. Hold for 3 seconds, then open.',
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'relaxed_smile',
    title: 'Relaxed Smile',
    shortLabel: 'Smile',
    instruction:
      'Smile gently without showing teeth. Hold for 3 seconds, then relax.',
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'lip_pucker',
    title: 'Lip Pucker',
    shortLabel: 'Pucker',
    instruction:
      "Purse your lips like you're going to whistle or give a kiss. Hold for 3 seconds, then relax.",
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'lower_teeth_show',
    title: 'Lower Teeth Show',
    shortLabel: 'Lower teeth',
    instruction:
      'Open your mouth and pull your lower lip down to show your bottom teeth. Hold for 3 seconds, then relax.',
    holdSeconds: 3,
    optional: false,
  },
  {
    id: 'reanimated_smile',
    title: 'Reanimated Smile (if applicable)',
    shortLabel: 'Reanimated smile',
    instruction:
      'If you’ve had facial reanimation surgery, please attempt your reanimated smile now. Hold for 3 seconds, then relax.',
    holdSeconds: 3,
    optional: true,
  },
] as const

export const FACES_COMPLETION = [
  'You’ve completed all the steps—great work!',
  'Thank you for participating. These recordings are an important part of tracking your recovery and improving your care.',
] as const
