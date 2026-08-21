export function EnvironmentStrip() {
  return (
    <div className="environment-strip" role="status" aria-label="Workspace environment">
      <div className="environment-strip__inner">
        <span className="environment-strip__item environment-strip__item--data">
          <span className="environment-strip__copy--full">
            Research prototype · sample data only · session data resets on refresh
          </span>
          <span className="environment-strip__copy--compact">
            Research prototype · sample data only
          </span>
        </span>
      </div>
    </div>
  )
}
