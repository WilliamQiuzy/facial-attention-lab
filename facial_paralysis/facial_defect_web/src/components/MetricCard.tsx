type MetricCardProps = {
  label: string
  imageA: string
  imageB: string
  note: string
}

export function MetricCard({ label, imageA, imageB, note }: MetricCardProps) {
  return (
    <article className="metric-card">
      <p className="metric-card__label">{label}</p>
      <div className="metric-card__values">
        <div>
          <span>A</span>
          <strong>{imageA}</strong>
        </div>
        <div>
          <span>B</span>
          <strong>{imageB}</strong>
        </div>
      </div>
      <p>{note}</p>
      <span className="metric-card__badge">Simulated values</span>
    </article>
  )
}
