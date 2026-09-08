/** A local matching label, not a patient identifier or globally unique session ID. */
export function getSessionFilename(recording: File, artifact: 'report' | 'recording'): string {
  const timestamp = new Date(recording.lastModified).toISOString().replace(/[-:.]/g, '')
  // Deliberately exclude the source filename: it may contain patient identifiers.
  const metadata = `${recording.lastModified}:${recording.size}:${recording.type.toLowerCase()}`
  let hash = 2166136261
  for (let index = 0; index < metadata.length; index += 1) {
    hash = Math.imul(hash ^ metadata.charCodeAt(index), 16777619)
  }
  const session = `faces-research-${timestamp}-${(hash >>> 0).toString(36).padStart(7, '0')}`
  return `${session}-${artifact}.${artifact === 'report' ? 'pdf' : recordingExtension(recording)}`
}

function recordingExtension(recording: File): string {
  const mime = recording.type.toLowerCase().split(';')[0].trim()
  const containers: Readonly<Record<string, string>> = {
    'video/webm': 'webm',
    'video/mp4': 'mp4',
    'video/quicktime': 'mov',
    'video/ogg': 'ogg',
  }
  if (containers[mime]) return containers[mime]
  const extension = recording.name.split('.').pop()?.toLowerCase()
  return extension && ['webm', 'mp4', 'mov', 'ogg', 'm4v'].includes(extension) ? extension : 'video'
}
