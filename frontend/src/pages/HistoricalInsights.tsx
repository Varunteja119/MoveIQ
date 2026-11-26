import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Activity, Clock3, DollarSign, MapPin, RefreshCw, TrendingUp } from 'lucide-react'
import './HistoricalInsights.css'

type ZoneProfile = {
  location_id: number
  zone_name: string
  borough: string
  first_date: string
  last_date: string
  total_trips: number
  valid_fare_trips: number
  fare_revenue: number
  average_fare: number | null
  valid_distance_trips: number
  average_distance_miles: number | null
  valid_duration_trips: number
  average_duration_minutes: number | null
}

type HourlyRecord = {
  pickup_hour: string
  pickup_date: string
  pickup_hour_of_day: number
  total_trips: number
  average_fare: number | null
}

type ComparisonRecord = {
  segment: string
  total_trips: number
  average_fare: number | null
  average_distance_miles: number | null
  average_duration_minutes: number | null
  fare_revenue: number
}

type Props = {
  initialZoneId: number
  onZoneChange: (id: number) => void
}

const compact = (value: number) =>
  new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)

const number = (value: number) =>
  new Intl.NumberFormat('en-US').format(Math.round(value))

const money = (value: number | null | undefined) =>
  value == null
    ? '--'
    : new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value)

function HistoricalInsights({ initialZoneId, onZoneChange }: Props) {
  const [zoneId, setZoneId] = useState(initialZoneId)
  const [role, setRole] = useState<'pickup' | 'dropoff'>('pickup')
  const [startDate, setStartDate] = useState('2025-01-01')
  const [endDate, setEndDate] = useState('2025-01-31')
  const [profile, setProfile] = useState<ZoneProfile | null>(null)
  const [hourly, setHourly] = useState<HourlyRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [compareAStart, setCompareAStart] = useState('2025-01-01')
  const [compareAEnd, setCompareAEnd] = useState('2025-01-31')
  const [compareBStart, setCompareBStart] = useState('2026-01-01')
  const [compareBEnd, setCompareBEnd] = useState('2026-01-31')
  const [comparison, setComparison] = useState<ComparisonRecord[]>([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState('')



  useEffect(() => {
    if (!Number.isInteger(zoneId) || zoneId < 1 || zoneId > 263) return

    const controller = new AbortController()

    async function loadZone() {
      setLoading(true)
      setError('')
      setProfile(null)
      setHourly([])

      const params = new URLSearchParams({
        location_role: role,
        start_date: startDate,
        end_date: endDate,
      })

      try {
        const [profileResponse, hourlyResponse] = await Promise.all([
          fetch(`/api/v1/historical/zones/${zoneId}?${params}`, {
            signal: controller.signal,
          }),
          fetch(`/api/v1/historical/zones/${zoneId}/hourly?${params}`, {
            signal: controller.signal,
          }),
        ])

        if (!profileResponse.ok || !hourlyResponse.ok) {
  const profileBody = await profileResponse.text()
  const hourlyBody = await hourlyResponse.text()

  throw new Error(
    `Profile: ${profileResponse.status} - ${profileBody.slice(0, 200)} | ` +
    `Hourly: ${hourlyResponse.status} - ${hourlyBody.slice(0, 200)}`
  )
}

        const [profileData, hourlyData] = await Promise.all([
          profileResponse.json() as Promise<ZoneProfile | null>,
          hourlyResponse.json() as Promise<{ records: HourlyRecord[] }>,
        ])

        if (controller.signal.aborted) return

        setProfile(profileData)
        setHourly(hourlyData.records ?? [])
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'Unable to load historical data.')
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    void loadZone()
    return () => controller.abort()
  }, [zoneId, role, startDate, endDate])

  const dailyData = useMemo(() => {
    const grouped = new Map<string, number>()

    hourly.forEach((item) => {
      const date = String(item.pickup_date).slice(0, 10)
      grouped.set(date, (grouped.get(date) ?? 0) + item.total_trips)
    })

    return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [hourly])

  const hourOfDayData = useMemo(() => {
    const grouped = Array.from({ length: 24 }, () => 0)

    hourly.forEach((item) => {
      const hour = Number(item.pickup_hour_of_day)
      if (hour >= 0 && hour <= 23) grouped[hour] += item.total_trips
    })

    return grouped
  }, [hourly])

  const peakHour = useMemo(() => {
    const max = Math.max(...hourOfDayData)
    return max > 0 ? hourOfDayData.indexOf(max) : null
  }, [hourOfDayData])

  const dailyOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 25, right: 22, bottom: 48, left: 58 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111b2a',
      borderColor: '#294156',
      textStyle: { color: '#e8f0fa' },
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params
        return `${point?.name}<br/>Trips: <b>${number(Number(point?.value))}</b>`
      },
    },
    xAxis: {
      type: 'category',
      data: dailyData.map(([date]) => date),
      axisLabel: { color: '#8fa5b8', rotate: 30, hideOverlap: true },
      axisLine: { lineStyle: { color: '#26394c' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#8fa5b8',
        formatter: (value: number) => compact(value),
      },
      splitLine: { lineStyle: { color: 'rgba(130,160,185,.12)' } },
    },
    series: [{
      type: 'line',
      data: dailyData.map(([, trips]) => trips),
      smooth: true,
      showSymbol: false,
      lineStyle: { color: '#31d7f5', width: 2.5 },
      areaStyle: { color: 'rgba(49,215,245,.13)' },
    }],
  }

  const hourlyOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 22, right: 18, bottom: 38, left: 55 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111b2a',
      borderColor: '#294156',
      textStyle: { color: '#e8f0fa' },
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params
        return `${point?.name}<br/>Trips: <b>${number(Number(point?.value))}</b>`
      },
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 24 }, (_, hour) => `${String(hour).padStart(2, '0')}:00`),
      axisLabel: { color: '#8fa5b8', interval: 2 },
      axisLine: { lineStyle: { color: '#26394c' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#8fa5b8',
        formatter: (value: number) => compact(value),
      },
      splitLine: { lineStyle: { color: 'rgba(130,160,185,.12)' } },
    },
    series: [{
      type: 'bar',
      data: hourOfDayData.map((value, hour) => ({
        value,
        itemStyle: {
          color: hour === peakHour ? '#31d7f5' : '#287fa8',
          borderRadius: [4, 4, 0, 0],
        },
      })),
      barMaxWidth: 22,
    }],
  }

  async function runComparison() {
    setCompareLoading(true)
    setCompareError('')
    setComparison([])

    try {
      const response = await fetch('/api/v1/historical/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location_id: zoneId,
          location_role: role,
          segment_a_start: compareAStart,
          segment_a_end: compareAEnd,
          segment_b_start: compareBStart,
          segment_b_end: compareBEnd,
        }),
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.detail ?? 'Unable to compare the selected periods.')
      }

      setComparison(payload as ComparisonRecord[])
    } catch (cause) {
      setCompareError(cause instanceof Error ? cause.message : 'Comparison failed.')
    } finally {
      setCompareLoading(false)
    }
  }

  const segmentA = comparison.find((item) => item.segment === 'segment_a')
  const segmentB = comparison.find((item) => item.segment === 'segment_b')
  const tripChange = segmentA && segmentB && segmentA.total_trips > 0
    ? ((segmentB.total_trips - segmentA.total_trips) / segmentA.total_trips) * 100
    : null

  return (
    <div className="historical-page">
      <div className="historical-heading">
        <div>
          <div className="historical-eyebrow"><span /> LONG-RANGE MOBILITY ANALYTICS</div>
          <h1>Explore a zone's <span>history.</span></h1>
          <p>Investigate demand, fares and trip characteristics across the NYC taxi network.</p>
        </div>
        <div className="historical-coverage"><Clock3 size={15} /> JAN 2024 — SEP 2026</div>
      </div>

      <section className="historical-controls">
        <label>
          <span>ZONE ID</span>
          <input
            type="number"
            min="1"
            max="263"
            value={zoneId}
            onChange={(event) => {
              const value = Number(event.target.value)
              setZoneId(value)
              if (Number.isInteger(value) && value >= 1 && value <= 263) onZoneChange(value)
            }}
          />
        </label>
        <label>
          <span>LOCATION ROLE</span>
          <select value={role} onChange={(event) => setRole(event.target.value as 'pickup' | 'dropoff')}>
            <option value="pickup">Pickup zone</option>
            <option value="dropoff">Drop-off zone</option>
          </select>
        </label>
        <label>
          <span>START DATE</span>
          <input type="date" min="2024-01-01" max="2026-09-30" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
        </label>
        <label>
          <span>END DATE</span>
          <input type="date" min="2024-01-01" max="2026-09-30" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
        </label>
        <button
          className="historical-reset"
          type="button"
          onClick={() => {
            setStartDate('2025-01-01')
            setEndDate('2025-01-31')
          }}
          title="Reset date range"
        >
          <RefreshCw size={15} />
        </button>
      </section>

      {error && <div className="historical-error">{error}</div>}
      {loading ? (
        <div className="historical-loading">Loading historical zone analytics...</div>
      ) : profile ? (
        <>
          <div className="historical-zone-title">
            <div>
              <span>SELECTED ZONE</span>
              <h2>{profile.zone_name}</h2>
              <p>{profile.borough} · Zone {profile.location_id} · {role === 'pickup' ? 'Pickup activity' : 'Drop-off activity'}</p>
            </div>
            <span className="historical-range">{startDate} — {endDate}</span>
          </div>

          <div className="historical-metrics">
            <article><div><span>TOTAL TRIPS</span><Activity size={17} /></div><strong>{compact(profile.total_trips)}</strong><small>{number(profile.total_trips)} recorded trips</small></article>
            <article><div><span>AVERAGE FARE</span><DollarSign size={17} /></div><strong>{money(profile.average_fare)}</strong><small>{number(profile.valid_fare_trips)} valid fare records</small></article>
            <article><div><span>AVERAGE DISTANCE</span><MapPin size={17} /></div><strong>{profile.average_distance_miles == null ? '--' : `${profile.average_distance_miles.toFixed(2)} mi`}</strong><small>Valid distance records</small></article>
            <article><div><span>AVERAGE DURATION</span><TrendingUp size={17} /></div><strong>{profile.average_duration_minutes == null ? '--' : `${profile.average_duration_minutes.toFixed(1)} min`}</strong><small>Valid duration records</small></article>
          </div>

          <section className="historical-panel">
            <div className="historical-panel-heading">
              <div><span>01 / DAILY ACTIVITY</span><h2>How does demand change over time?</h2><p>Trip volume by calendar day in the selected period</p></div>
              <span>TRIPS / DAY</span>
            </div>
            <ReactECharts option={dailyOption} style={{ height: 310 }} notMerge lazyUpdate />
          </section>

          <section className="historical-panel">
            <div className="historical-panel-heading">
              <div><span>02 / INTRADAY PROFILE</span><h2>When is this zone busiest?</h2><p>Trip totals grouped by hour of day across the selected dates</p></div>
              <span>TRIPS / HOUR</span>
            </div>
            <ReactECharts option={hourlyOption} style={{ height: 300 }} notMerge lazyUpdate />
            <div className="historical-observation">
              <Activity size={17} />
              {peakHour == null
                ? 'No hourly demand data is available for this selection.'
                : <>The busiest hour is <strong>{String(peakHour).padStart(2, '0')}:00</strong>, with <strong>{number(hourOfDayData[peakHour])}</strong> trips across the selected period.</>}
            </div>
            <p className="historical-note">
              Time basis: pickup hour. For drop-off zones, trips are grouped by the hour they were originally picked up, not their arrival time.
            </p>
          </section>

          <section className="historical-panel">
            <div className="historical-panel-heading">
              <div><span>03 / PERIOD COMPARISON</span><h2>Has activity changed?</h2><p>Compare two non-overlapping periods for this zone</p></div>
            </div>
            <div className="comparison-controls">
              <div><strong>PERIOD A</strong><label>From<input type="date" value={compareAStart} onChange={(event) => setCompareAStart(event.target.value)} /></label><label>To<input type="date" value={compareAEnd} onChange={(event) => setCompareAEnd(event.target.value)} /></label></div>
              <div><strong>PERIOD B</strong><label>From<input type="date" value={compareBStart} onChange={(event) => setCompareBStart(event.target.value)} /></label><label>To<input type="date" value={compareBEnd} onChange={(event) => setCompareBEnd(event.target.value)} /></label></div>
              <button type="button" onClick={runComparison} disabled={compareLoading}>{compareLoading ? 'Comparing...' : 'Compare periods'}</button>
            </div>
            {compareError && <div className="historical-error">{compareError}</div>}
            {segmentA && segmentB && (
              <div className="comparison-result">
                <div><span>PERIOD A TRIPS</span><strong>{number(segmentA.total_trips)}</strong></div>
                <div><span>PERIOD B TRIPS</span><strong>{number(segmentB.total_trips)}</strong></div>
                <div><span>TRIP VOLUME CHANGE</span><strong className={tripChange != null && tripChange < 0 ? 'negative' : 'positive'}>{tripChange == null ? '--' : `${tripChange > 0 ? '+' : ''}${tripChange.toFixed(1)}%`}</strong></div>
                <div><span>PERIOD B AVG FARE</span><strong>{money(segmentB.average_fare)}</strong></div>
              </div>
            )}
          </section>
        </>
      ) : !error ? (
        <div className="historical-loading">No records found for this zone and date range.</div>
      ) : null}

      <footer className="historical-footer">
        <span>NYC TLC YELLOW TAXI DATA</span>
        <span>HISTORICAL COVERAGE: JAN 2024 — SEP 2026</span>
      </footer>
    </div>
  )
}

export default HistoricalInsights
