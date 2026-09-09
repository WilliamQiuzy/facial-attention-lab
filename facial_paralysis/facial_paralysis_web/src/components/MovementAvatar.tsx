import type { FacesActionId } from '../protocol/facesProtocol'

interface MovementAvatarProps {
  readonly action: FacesActionId
  readonly title: string
  readonly active?: boolean
}

function Mouth({ action }: { readonly action: FacesActionId }) {
  if (action === 'lip_pucker') {
    return <ellipse className="avatar-mouth avatar-mouth-pucker" cx="110" cy="146" rx="10" ry="8" />
  }

  if (action === 'lower_teeth_show') {
    return (
      <g className="avatar-mouth avatar-mouth-open">
        <path d="M78 137 Q110 151 142 137 Q138 174 110 177 Q82 174 78 137Z" />
        <path className="avatar-teeth" d="M86 146 Q110 153 134 146 L132 155 Q110 161 88 155Z" />
      </g>
    )
  }

  if (action === 'relaxed_smile') {
    return <path className="avatar-mouth avatar-mouth-smile" d="M76 139 Q110 169 144 139" />
  }

  if (action === 'reanimated_smile') {
    return <path className="avatar-mouth avatar-mouth-smile avatar-mouth-reanimated" d="M76 143 Q107 166 144 135" />
  }

  return <path className="avatar-mouth" d="M88 147 Q110 151 132 147" />
}

export function MovementAvatar({ action, title, active = false }: MovementAvatarProps) {
  const eyesClosed = action === 'gentle_eye_closure' || action === 'tight_eye_squeeze'
  const tightClosure = action === 'tight_eye_squeeze'

  return (
    <div
      className={`movement-avatar ${active ? 'is-active' : ''}`}
      role="img"
      aria-label={`${title} movement demonstration`}
      data-action={action}
    >
      <svg viewBox="0 0 220 220" aria-hidden="true" focusable="false">
        <circle className="avatar-halo" cx="110" cy="110" r="101" />
        <path
          className="avatar-head"
          d="M110 27 C68 27 45 60 48 108 C50 155 73 190 110 192 C147 190 170 155 172 108 C175 60 152 27 110 27Z"
        />
        <path className="avatar-ear" d="M48 97 C36 94 35 126 51 132" />
        <path className="avatar-ear" d="M172 97 C184 94 185 126 169 132" />

        <g className="avatar-brows">
          <path className="avatar-brow" d="M67 82 Q82 72 96 81" />
          <path className="avatar-brow" d="M124 81 Q138 72 153 82" />
        </g>

        {eyesClosed ? (
          <g className={`avatar-eyes avatar-eyes-closed ${tightClosure ? 'is-tight' : ''}`}>
            <path d="M68 105 Q82 114 96 105" />
            <path d="M124 105 Q138 114 152 105" />
            {tightClosure ? (
              <>
                <path className="avatar-strain" d="M69 116 Q82 120 95 116" />
                <path className="avatar-strain" d="M125 116 Q138 120 151 116" />
              </>
            ) : null}
          </g>
        ) : (
          <g className="avatar-eyes avatar-eyes-open">
            <ellipse cx="82" cy="104" rx="13" ry="7" />
            <ellipse cx="138" cy="104" rx="13" ry="7" />
            <circle cx="82" cy="104" r="3.2" />
            <circle cx="138" cy="104" r="3.2" />
          </g>
        )}

        <path className="avatar-nose" d="M110 103 C109 116 108 127 102 130 Q110 135 118 130" />
        <Mouth action={action} />

        {action === 'eyebrow_raise' ? (
          <g className="avatar-motion-cues avatar-motion-cues-up">
            <path d="M82 66 V49 M75 56 L82 49 L89 56" />
            <path d="M138 66 V49 M131 56 L138 49 L145 56" />
          </g>
        ) : null}
        {action === 'lower_teeth_show' ? (
          <path className="avatar-motion-cues" d="M110 183 V199 M103 192 L110 199 L117 192" />
        ) : null}
        {action === 'reanimated_smile' ? (
          <path className="avatar-motion-cues" d="M151 150 L164 136 M154 137 L164 136 L163 146" />
        ) : null}
      </svg>
    </div>
  )
}
