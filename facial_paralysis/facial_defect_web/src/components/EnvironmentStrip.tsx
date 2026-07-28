export function EnvironmentStrip() {
  return (
    <div className="environment-strip" role="status" aria-label="Workspace environment">
      <div className="environment-strip__inner">
        <span className="environment-strip__item environment-strip__item--data">
          Research prototype · synthetic/test records only · session data resets on
          refresh · clinical use blocked
        </span>
      </div>
    </div>
  )
}
