
import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Activity, Clock3, TrendingUp, Zap } from 'lucide-react'
import './ForecastPage.css'

type ForecastPoint = {
  forecast_origin: string
  target_timestamp: string
  hours_ahead: number
  predicted_demand: number
}

type ForecastResponse = {
  tool: string
  source: string
  coverage: {
    forecast_origin: string
    forecast_start: string
    forecast_end: string
    forecast_horizon_hours: number
    timezone: string
    timestamp_semantics: string
  }
  data: ForecastPoint[]
}

function toDateTimeLocal(value: string) {
  return value.replace(' ', 'T').slice(0, 16)
}

function ForecastPage() {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [horizon, setHorizon] = useState(24)
  const [selectedOrigin, setSelectedOrigin] = useState('')

  async function loadForecast(asOf?: string) {
    try {
      setLoading(true)
      setError('')

      const query = asOf
        ? `?as_of=${encodeURIComponent(asOf.replace('T', ' '))}`
        : ''
      const response = await fetch(`/api/v1/forecasts/citywide${query}`)

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `Forecast API returned ${response.status}`)
      }

      const result = await response.json() as ForecastResponse
      setForecast(result)
      setSelectedOrigin(toDateTimeLocal(result.coverage.forecast_origin))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load forecast.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const controller = new AbortController()

    fetch('/api/v1/forecasts/citywide', {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const message = await response.text()
          throw new Error(message || `Forecast API returned ${response.status}`)
        }

        return response.json() as Promise<ForecastResponse>
      })
      .then((result) => {
        if (controller.signal.aborted) return

        setForecast(result)
        setSelectedOrigin(toDateTimeLocal(result.coverage.forecast_origin))
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Unable to load forecast.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [])

  const points = (forecast?.data ?? []).slice(0, horizon)

  const peak = useMemo(
    () => points.reduce<ForecastPoint | null>(
      (best, point) => !best || point.predicted_demand > best.predicted_demand ? point : best,
      null,
    ),
    [points],
  )

  const total = points.reduce((sum, point) => sum + point.predicted_demand, 0)
  const average = points.length ? total / points.length : 0

  const chartOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 25, right: 25, bottom: 35, left: 65 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111b2a',
      borderColor: '#294156',
      textStyle: { color: '#e8f0fa' },
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params
        return `${point?.name}<br/>Predicted demand: <b>${Number(point?.value).toLocaleString('en-US', { maximumFractionDigits: 0 })}</b>`
      },
    },
    xAxis: {
      type: 'category',
      data: points.map(point => point.target_timestamp.slice(11, 16)),
      axisLabel: { color: '#8fa5b8', interval: 2 },
      axisLine: { lineStyle: { color: '#26394c' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'Predicted trips',
      nameTextStyle: { color: '#8fa5b8' },
      axisLabel: {
        color: '#8fa5b8',
        formatter: (value: number) => new Intl.NumberFormat('en-US', { notation: 'compact' }).format(value),
      },
      splitLine: { lineStyle: { color: 'rgba(130,160,185,.12)' } },
    },
    series: [{
      name: 'Predicted demand',
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 7,
      data: points.map(point => Math.round(point.predicted_demand)),
      lineStyle: { color: '#31d7f5', width: 3 },
      itemStyle: { color: '#31d7f5' },
      areaStyle: { color: 'rgba(49,215,245,.13)' },
    }],
  }

  const dateLabel = forecast?.coverage.forecast_start
    ? new Date(`${forecast.coverage.forecast_start.slice(0, 10)}T12:00:00`)
        .toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    : 'Forecast date unavailable'

  return (
    <div className="forecast-page">
      <div className="forecast-heading">
        <div>
          <div className="forecast-eyebrow"><span /> PREDICTIVE MOBILITY INTELLIGENCE</div>
          <h1>Anticipate the <span>city's next move.</span></h1>
          <p>Explore model-generated hourly taxi demand predictions across New York City.</p>
        </div>
        <button
          className="forecast-refresh"
          onClick={() => void loadForecast(selectedOrigin || undefined)}
          disabled={loading}
        >
          <Activity size={15} /> {loading ? 'Loading...' : 'Refresh forecast'}
        </button>
      </div>

      <div className="forecast-controls">
        <div className="forecast-control-group">
          <label htmlFor="forecast-horizon">FORECAST WINDOW</label>
          <div className="forecast-horizon-options">
            {[6, 12, 24].map(value => (
              <button
                key={value}
                type="button"
                className={horizon === value ? 'active' : ''}
                onClick={() => setHorizon(value)}
                disabled={loading}
              >
                {value} hours
              </button>
            ))}
          </div>
        </div>

        <div className="forecast-control-group forecast-origin-control">
          <label htmlFor="forecast-origin">FORECAST ORIGIN (NEW YORK TIME)</label>
          <div className="forecast-origin-input">
            <input
              id="forecast-origin"
              type="datetime-local"
              value={selectedOrigin}
              onChange={event => setSelectedOrigin(event.target.value)}
              disabled={loading}
            />
            <button
              type="button"
              onClick={() => void loadForecast(selectedOrigin)}
              disabled={loading || !selectedOrigin}
            >
              Apply
            </button>
          </div>
        </div>
      </div>

      <div className="forecast-history-note">
        <Clock3 size={16} />
        Forecasts use historical training data ending September 30, 2026. They are model predictions, not live demand measurements.
      </div>

      {error && <div className="forecast-error">{error}</div>}
      {loading && <div className="forecast-loading">Loading citywide forecast...</div>}

      {!loading && forecast && (
        <>
          <div className="forecast-notice">
            <Clock3 size={17} />
            <div>
              <strong>Forecast period: {dateLabel}</strong>
              <span>
                Origin: {forecast.coverage.forecast_origin} · {forecast.coverage.timezone}
                {' · '} {points.length}-hour display window
              </span>
            </div>
          </div>

          <div className="forecast-metrics">
            <article className="forecast-metric">
              <div><span>PEAK PREDICTED DEMAND</span><TrendingUp size={17} /></div>
              <strong>{peak ? Math.round(peak.predicted_demand).toLocaleString('en-US') : '--'}</strong>
              <small>{peak ? `${peak.target_timestamp.slice(11, 16)} local time` : 'No forecast data'}</small>
            </article>
            <article className="forecast-metric">
              <div><span>AVERAGE HOURLY DEMAND</span><Activity size={17} /></div>
              <strong>{Math.round(average).toLocaleString('en-US')}</strong>
              <small>Predicted trips per hour</small>
            </article>
            <article className="forecast-metric">
              <div><span>FORECAST HORIZON</span><Clock3 size={17} /></div>
              <strong>{points.length} hrs</strong>
              <small>Selected prediction window</small>
            </article>
            <article className="forecast-metric">
              <div><span>FORECAST POINTS</span><Zap size={17} /></div>
              <strong>{points.length}</strong>
              <small>Citywide hourly predictions</small>
            </article>
          </div>

          <section className="forecast-panel">
            <div className="forecast-panel-heading">
              <div>
                <span>01 / CITYWIDE DEMAND</span>
                <h2>How demand changes throughout the day</h2>
                <p>Model-predicted taxi trips by target hour</p>
              </div>
              <span className="forecast-unit">PREDICTED TRIPS / HOUR</span>
            </div>
            {points.length > 0
              ? <ReactECharts option={chartOption} style={{ height: 370 }} notMerge lazyUpdate />
              : <div className="forecast-loading">No forecast points returned.</div>}
            {peak && (
              <div className="forecast-observation">
                <TrendingUp size={17} />
                <p>
                  The model predicts the highest demand at <strong>{peak.target_timestamp.slice(11, 16)}</strong>,
                  with approximately <strong>{Math.round(peak.predicted_demand).toLocaleString('en-US')} trips</strong>.
                </p>
              </div>
            )}
          </section>

          <section className="forecast-panel">
            <div className="forecast-panel-heading">
              <div>
                <span>02 / HOURLY BREAKDOWN</span>
                <h2>Prediction details</h2>
                <p>Individual model outputs for the selected forecast window</p>
              </div>
            </div>
            <div className="forecast-table-wrap">
              <table className="forecast-table">
                <thead>
                  <tr><th>Target time</th><th>Hours ahead</th><th>Predicted demand</th></tr>
                </thead>
                <tbody>
                  {points.map(point => (
                    <tr key={point.target_timestamp}>
                      <td>{point.target_timestamp.replace('T', ' ')}</td>
                      <td>+{point.hours_ahead}h</td>
                      <td>{Math.round(point.predicted_demand).toLocaleString('en-US')} trips</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <footer className="forecast-footer">
            <span>{forecast.source}</span>
            <span>Times are New York local time, not UTC</span>
          </footer>
        </>
      )}
    </div>
  )
}

export default ForecastPage
