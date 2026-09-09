import type {
  PatientFacePath,
  PatientFaceRegistration,
} from '../patientWorkflow/types'

type PatientFaceContourProps = {
  readonly registration: PatientFaceRegistration
}

function viewBoxCoordinate(value: number): string {
  return Number((value * 100).toFixed(3)).toString()
}

function pathPoints(path: PatientFacePath): string {
  return path.points
    .map(
      (point) =>
        `${viewBoxCoordinate(point.x)},${viewBoxCoordinate(point.y)}`,
    )
    .join(' ')
}

export function PatientFaceContour({
  registration,
}: PatientFaceContourProps) {
  return (
    <svg
      aria-hidden="true"
      className="patient-face-contour"
      data-geometry-source={registration.source}
      focusable="false"
      preserveAspectRatio="none"
      viewBox="0 0 100 100"
    >
      {registration.paths.map((path, index) => {
        const commonProps = {
          className: `patient-face-contour__path patient-face-contour__path--${path.feature.replaceAll('_', '-')}`,
          'data-feature': path.feature,
          points: pathPoints(path),
        }

        return path.closed ? (
          <polygon key={`${path.feature}-${index}`} {...commonProps} />
        ) : (
          <polyline key={`${path.feature}-${index}`} {...commonProps} />
        )
      })}
    </svg>
  )
}
