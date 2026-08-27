import type { jsPDF as JsPdfDocument } from 'jspdf'

import sourceSansRegular from '../assets/fonts/SourceSans3-Regular.ttf?inline'
import sourceSansSemibold from '../assets/fonts/SourceSans3-Semibold.ttf?inline'

export const RESEARCH_REPORT_PDF_FILENAME = 'faces-research-movement-report.pdf'

export interface PdfMeasurement {
  readonly label: string
  readonly kind: string
  readonly primaryValue: string
  readonly normalizedValue: string
  readonly explanation: string
}

export interface PdfActionEvidence {
  readonly title: string
  readonly contextSeconds: string
  readonly tracking: string
  readonly imageDataUrl: string | null
  readonly measurements: readonly PdfMeasurement[]
}

export interface ResearchReportPdfData {
  readonly score: string
  readonly scoreMeaning: string
  readonly outputClass: string
  readonly cutpointSummary: string
  readonly recordingCoverage: readonly string[]
  readonly actions: readonly PdfActionEvidence[]
  readonly clinicalReviewNote: string
}

type Rgb = readonly [number, number, number]

const COLORS = {
  ink: [12, 31, 52] as Rgb,
  muted: [73, 91, 109] as Rgb,
  blue: [0, 87, 184] as Rgb,
  blueDark: [1, 45, 97] as Rgb,
  pale: [241, 247, 253] as Rgb,
  line: [207, 219, 230] as Rgb,
} as const

const PDF_FONT = 'SourceSans3'
const ACTION_CARD_BASE_HEIGHT = 37
const ACTION_MEASUREMENT_HEIGHT = 23.5
const ACTION_CARD_MIN_HEIGHT = 88

function fontBase64(dataUrl: string): string {
  const marker = ';base64,'
  const markerIndex = dataUrl.indexOf(marker)
  if (markerIndex < 0) throw new Error('Embedded PDF font is not base64 encoded')
  return dataUrl.slice(markerIndex + marker.length)
}

function registerFonts(document: JsPdfDocument): void {
  document.addFileToVFS('SourceSans3-Regular.ttf', fontBase64(sourceSansRegular))
  document.addFont('SourceSans3-Regular.ttf', PDF_FONT, 'normal', 400)
  document.addFileToVFS('SourceSans3-Semibold.ttf', fontBase64(sourceSansSemibold))
  document.addFont('SourceSans3-Semibold.ttf', PDF_FONT, 'bold', 600)
}

function setTextColor(document: JsPdfDocument, color: Rgb): void {
  document.setTextColor(color[0], color[1], color[2])
}

function wrappedText(
  document: JsPdfDocument,
  text: string,
  x: number,
  y: number,
  width: number,
  size: number,
  color: Rgb,
  style: 'normal' | 'bold' = 'normal',
  lineHeight = 1.25,
): number {
  document.setFont(PDF_FONT, style, style === 'bold' ? 600 : 400)
  document.setFontSize(size)
  setTextColor(document, color)
  const lines = document.splitTextToSize(text, width) as string[]
  document.text(lines, x, y, { lineHeightFactor: lineHeight })
  return y + lines.length * size * 0.3528 * lineHeight
}

function addPageHeading(document: JsPdfDocument, title: string, subtitle: string): number {
  document.setFillColor(...COLORS.blueDark)
  document.rect(0, 0, 210, 34, 'F')
  wrappedText(document, title, 16, 17, 178, 22, [255, 255, 255], 'bold')
  wrappedText(document, subtitle, 16, 27, 178, 9.5, [218, 234, 250])
  return 46
}

function drawAction(
  document: JsPdfDocument,
  action: PdfActionEvidence,
  y: number,
): number {
  const cardHeight = Math.max(
    ACTION_CARD_MIN_HEIGHT,
    ACTION_CARD_BASE_HEIGHT + action.measurements.length * ACTION_MEASUREMENT_HEIGHT,
  )
  document.setDrawColor(...COLORS.line)
  document.setFillColor(249, 251, 253)
  document.roundedRect(14, y, 182, cardHeight, 2.5, 2.5, 'FD')
  wrappedText(document, action.title, 20, y + 10, 92, 12, COLORS.blueDark, 'bold')
  wrappedText(document, `Tracking: ${action.tracking}`, 120, y + 9.5, 68, 10, COLORS.muted)

  if (action.imageDataUrl) {
    try {
      document.addImage(action.imageDataUrl, 'JPEG', 20, y + 17, 54, 38, undefined, 'FAST')
    } catch {
      document.setFillColor(232, 239, 245)
      document.rect(20, y + 17, 54, 38, 'F')
      wrappedText(document, 'Recorded context image unavailable', 25, y + 34, 44, 9, COLORS.muted)
    }
  } else {
    document.setFillColor(232, 239, 245)
    document.rect(20, y + 17, 54, 38, 'F')
    wrappedText(document, 'Recorded context image unavailable', 25, y + 34, 44, 9, COLORS.muted)
  }
  wrappedText(document, `Registered hold midpoint: ${action.contextSeconds}`, 20, y + 62, 54, 9.2, COLORS.muted)

  let measurementY = y + 21
  for (const measurement of action.measurements) {
    wrappedText(document, measurement.kind, 82, measurementY, 63, 9.5, [64, 86, 108], 'bold')
    wrappedText(document, measurement.label, 82, measurementY + 6.7, 62, 10, COLORS.ink)
    wrappedText(document, measurement.primaryValue, 147, measurementY + 6.7, 41, 10.5, COLORS.blueDark, 'bold')
    wrappedText(document, measurement.normalizedValue, 147, measurementY + 13, 41, 9.3, COLORS.muted)
    wrappedText(document, measurement.explanation, 82, measurementY + 16, 106, 9.3, COLORS.muted, 'normal', 1.35)
    measurementY += ACTION_MEASUREMENT_HEIGHT
  }
  return y + cardHeight + 7
}

export async function buildResearchReportPdf(data: ResearchReportPdfData): Promise<Blob> {
  const { jsPDF } = await import('jspdf')
  const document = new jsPDF({ unit: 'mm', format: 'a4', compress: true })
  registerFonts(document)
  document.setProperties({
    title: 'FACES Research Movement Report',
    subject: 'Facial movement research report with recorded context evidence',
    creator: 'FACES Research Capture',
  })

  let y = addPageHeading(
    document,
    'Research Movement Report',
    'Facial movement classification and recorded action evidence',
  )
  document.setFillColor(...COLORS.pale)
  document.setDrawColor(...COLORS.line)
  document.roundedRect(14, y, 182, 49, 3, 3, 'FD')
  wrappedText(document, 'MEEI FACIAL-MOVEMENT CLASSIFICATION SCORE', 20, y + 10, 95, 8, COLORS.blue, 'bold')
  wrappedText(document, data.score, 20, y + 29, 55, 28, COLORS.blueDark, 'bold')
  wrappedText(document, data.outputClass, 82, y + 22, 106, 14, COLORS.blueDark, 'bold')
  wrappedText(document, data.cutpointSummary, 82, y + 31, 106, 9, COLORS.muted)
  y += 61

  y = wrappedText(document, 'What this number represents', 14, y, 182, 14, COLORS.blueDark, 'bold') + 2
  y = wrappedText(document, data.scoreMeaning, 14, y, 182, 10, COLORS.ink, 'normal', 1.45) + 7

  y = wrappedText(document, 'Recording coverage', 14, y, 182, 13, COLORS.blueDark, 'bold') + 1
  for (const line of data.recordingCoverage) {
    y = wrappedText(document, `- ${line}`, 18, y, 174, 9, COLORS.ink, 'normal', 1.35) + 1
  }

  y += 5
  document.setFillColor(244, 248, 252)
  document.setDrawColor(82, 126, 167)
  document.roundedRect(14, y, 182, 27, 2.5, 2.5, 'FD')
  document.setFillColor(82, 126, 167)
  document.rect(14, y, 2, 27, 'F')
  wrappedText(document, 'Clinical review note', 21, y + 9, 167, 11, COLORS.blueDark, 'bold')
  wrappedText(document, data.clinicalReviewNote, 21, y + 17, 167, 9.5, [36, 54, 75], 'normal', 1.4)

  document.addPage()
  y = addPageHeading(
    document,
    'Recorded action evidence',
    'Context images and descriptive geometry from each registered three-second hold',
  )
  y = wrappedText(
    document,
    'Measurements are scaled to the same eye-to-eye reference width so values are comparable within this recording. They do not have a clinical normal range or severity cutpoint.',
    14,
    y,
    182,
    9,
    COLORS.muted,
    'normal',
    1.4,
  ) + 5

  for (const action of data.actions) {
    const needed = Math.max(
      ACTION_CARD_MIN_HEIGHT,
      ACTION_CARD_BASE_HEIGHT + action.measurements.length * ACTION_MEASUREMENT_HEIGHT,
    )
    if (y + needed > 282) {
      document.addPage()
      y = addPageHeading(document, 'Recorded action evidence', 'Continued')
    }
    y = drawAction(document, action, y)
  }

  const pageCount = document.getNumberOfPages()
  for (let page = 1; page <= pageCount; page += 1) {
    document.setPage(page)
    wrappedText(document, `Page ${page} of ${pageCount}`, 174, 291, 22, 8.5, COLORS.muted)
    wrappedText(document, 'Contains identifiable facial images - handle under the approved protocol.', 14, 291, 150, 8.5, COLORS.muted)
  }

  const bytes = document.output('arraybuffer')
  return new Blob([bytes], { type: 'application/pdf' })
}

export async function downloadResearchReportPdf(data: ResearchReportPdfData): Promise<void> {
  const blob = await buildResearchReportPdf(data)
  const objectUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = RESEARCH_REPORT_PDF_FILENAME
  anchor.rel = 'noopener'
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
}
