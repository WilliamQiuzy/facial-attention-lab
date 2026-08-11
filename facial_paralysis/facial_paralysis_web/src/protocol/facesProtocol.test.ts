import { describe, expect, it } from 'vitest'

import {
  FACES_COMPLETION,
  FACES_PREPARATION,
  FACES_PROTOCOL,
} from './facesProtocol'

describe('FACES protocol source fidelity', () => {
  it('preserves the four source preparation paragraphs', () => {
    expect(FACES_PREPARATION).toEqual([
      'Welcome and thank you for participating in our study. This recording helps us understand how your facial movements change over time. Please follow the instructions carefully. If you’re at home, try to replicate the same conditions each time you record.',
      'Use an iPad or iPhone with a front-facing camera. Place the device around arm’s length away from you. This minimizes facial distortion caused by wide-angle lenses. Use a tripod or prop the device at eye level.',
      'Your entire face and neck should be clearly visible. Look directly forward, and keep your head and body still. Sit in a well-lit room. Avoid bright windows or lights behind you. Choose a neutral background - plain wall, if possible.',
      'We’ll now go through a series of facial movements. Start with a relaxed face and follow along as each instruction is read aloud. Try not to move your head.',
    ])
  })

  it('preserves all eight ordered voice instructions', () => {
    expect(FACES_PROTOCOL.map(({ id, instruction }) => ({ id, instruction }))).toEqual([
      {
        id: 'repose',
        instruction:
          'Look straight ahead. Keep your face relaxed. Don’t smile or frown. Hold this position for 3 seconds.',
      },
      {
        id: 'eyebrow_raise',
        instruction:
          'Raise your eyebrows as high as you can. Hold for 3 seconds, then relax.',
      },
      {
        id: 'gentle_eye_closure',
        instruction:
          "Close your eyes gently, like you're falling asleep. Hold for 3 seconds, then open your eyes.",
      },
      {
        id: 'tight_eye_squeeze',
        instruction:
          'Close your eyes as tightly as possible, using your facial muscles. Hold for 3 seconds, then open.',
      },
      {
        id: 'relaxed_smile',
        instruction:
          'Smile gently without showing teeth. Hold for 3 seconds, then relax.',
      },
      {
        id: 'lip_pucker',
        instruction:
          "Purse your lips like you're going to whistle or give a kiss. Hold for 3 seconds, then relax.",
      },
      {
        id: 'lower_teeth_show',
        instruction:
          'Open your mouth and pull your lower lip down to show your bottom teeth. Hold for 3 seconds, then relax.',
      },
      {
        id: 'reanimated_smile',
        instruction:
          'If you’ve had facial reanimation surgery, please attempt your reanimated smile now. Hold for 3 seconds, then relax.',
      },
    ])
  })

  it('marks three-second holds and only the conditional final step as optional', () => {
    expect(FACES_PROTOCOL.map((step) => step.holdSeconds)).toEqual(Array(8).fill(3))
    expect(FACES_PROTOCOL.map((step) => step.optional)).toEqual([
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
    ])
  })

  it('preserves both source completion messages', () => {
    expect(FACES_COMPLETION).toEqual([
      'You’ve completed all the steps—great work!',
      'Thank you for participating. These recordings are an important part of tracking your recovery and improving your care.',
    ])
  })
})
