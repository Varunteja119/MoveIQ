import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Activity, Clock3, DollarSign, MapPin, TrendingUp } from 'lucide-react'
import './PatternsPage.css'

type ApiResponse<T> = {
  data: T[]
  coverage?: {
    start_date: string
    end_date: string
    total_trips: number
  }
}

type DemandRow = {
  group_value: number
  trip_count: number
}

type RevenueRow = {
  group_value: string | null
  valid_fare_trips: number
  fare_revenue: number
  average_fare: number
}

type TripMetrics = {
  valid_fare_trips: number
  average_fare: number
  average_distance_miles: number
  average_duration_minutes: number
  median_fare: number
  median_distance_miles: number
}

const money = (value: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value)

const compact = (value: number) =>
  new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)

function PatternsPage() {
  const [demand, setDemand] = useState<DemandRow[]>([])
  const [revenue, setRevenue] = useState<RevenueRow[]>([])
  const [metrics, setMetrics] = useState<TripMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadPatterns() {
      try {
        setLoading(true)
        setError('')

        const [demandResponse, revenueResponse, metricsResponse] =
          await Promise.all([
            fetch('/api/v1/analytics/demand'),
            fetch('/api/v1/analytics/revenue'),
            fetch('/api/v1/analytics/trip-metrics'),
          ])

        if (!demandResponse.ok || !revenueResponse.ok || !metricsResponse.ok) {
          throw new Error('One or more analytics endpoints returned an error.')
        }

        const [demandData, revenueData, metricsData] = await Promise.all([
          demandResponse.json() as Promise<ApiResponse<DemandRow>>,
          revenueResponse.json() as Promise<ApiResponse<RevenueRow>>,
          metricsResponse.json() as Promise<ApiResponse<TripMetrics>>,
        ])

        if (!active) return

        setDemand(demandData.data)
        setRevenue(revenueData.data)
        setMetrics(metricsData.data[0] ?? null)
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : 'Unable to load analytics.')
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    void loadPatterns()
    return () => {
      active = false
    }
  }, [])

  const hourlyData = useMemo(
    () => [...demand].sort((a, b) => a.group_value - b.group_value),
    [demand],
  )

  const peakHour = useMemo(
    () => demand.reduce<DemandRow | null>(
      (best, item) => !best || item.trip_count > best.trip_count ? item : best,
      null,
    ),
    [demand],
  )

  const boroughRevenue = useMemo(
    () =>
      revenue
        .filter(
          (item): item is RevenueRow & { group_value: string } =>
            typeof item.group_value === 'string' && item.group_value !== 'Unknown',
        )
        .sort((a, b) => b.fare_revenue - a.fare_revenue),
    [revenue],
  )

  const demandOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 25, right: 18, bottom: 35, left: 55 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111b2a',
      borderColor: '#294156',
      textStyle: { color: '#e8f0fa' },
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params
        const hour = String(point?.name ?? '').split(':')[0]
        return `${String(hour).padStart(2, '0')}:00<br/>Trips: <b>${Number(point?.value).toLocaleString('en-US')}</b>`
      },
    },
    xAxis: {
      type: 'category',
      data: hourlyData.map((item) => `${String(item.group_value).padStart(2, '0')}:00`),
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
    series: [
      {
        type: 'bar',
        data: hourlyData.map((item) => ({
          value: item.trip_count,
          itemStyle: {
            color: item.group_value === peakHour?.group_value ? '#31d7f5' : '#287fa8',
            borderRadius: [4, 4, 0, 0],
          },
        })),
        barMaxWidth: 23,
      },
    ],
  }

  const revenueOption: EChartsOption = {
    backgroundColor: 'transparent',
    grid: { top: 12, right: 35, bottom: 25, left: 95 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#111b2a',
      borderColor: '#294156',
      textStyle: { color: '#e8f0fa' },
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params
        return `${point?.name}<br/>Fare revenue: <b>${money(Number(point?.value))}</b>`
      },
    },
    xAxis: {
      type: 'value',
      axisLabel: {
        color: '#8fa5b8',
        formatter: (value: number) => `$${compact(value)}`,
      },
      splitLine: { lineStyle: { color: 'rgba(130,160,185,.12)' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: boroughRevenue.map((item) => item.group_value),
      axisLabel: { color: '#c2d2df', fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: 'bar',
        data: boroughRevenue.map((item, index) => ({
          value: item.fare_revenue,
          itemStyle: {
            color: index === 0 ? '#31d7f5' : '#287fa8',
            borderRadius: [0, 5, 5, 0],
          },
        })),
        barMaxWidth: 22,
      },
    ],
  }

  return (
    <div className="patterns-page">
      <div className="patterns-heading">
        <div>
          <div className="patterns-eyebrow"><span /> MOBILITY ANALYTICS</div>
          <h1>Understand the <span>rhythm of the city.</span></h1>
          <p>Explore hourly demand, borough-level fare revenue and trip characteristics.</p>
        </div>
        <div className="patterns-period"><Clock3 size={15} /> JANUARY 2025</div>
      </div>

      {error && <div className="patterns-error">{error}</div>}

      {loading ? (
        <div className="patterns-loading">Loading mobility analytics...</div>
      ) : (
        <>
          <div className="patterns-metrics">
            <article className="patterns-metric">
              <div><span>PEAK DEMAND HOUR</span><Activity size={17} /></div>
              <strong>{peakHour ? `${String(peakHour.group_value).padStart(2, '0')}:00` : '--'}</strong>
              <small>{peakHour ? `${peakHour.trip_count.toLocaleString('en-US')} trips recorded` : 'No data available'}</small>
            </article>
            <article className="patterns-metric">
              <div><span>AVERAGE FARE</span><DollarSign size={17} /></div>
              <strong>{metrics ? `$${metrics.average_fare.toFixed(2)}` : '--'}</strong>
              <small>Per valid fare record</small>
            </article>
            <article className="patterns-metric">
              <div><span>AVERAGE TRIP DISTANCE</span><MapPin size={17} /></div>
              <strong>{metrics ? `${metrics.average_distance_miles.toFixed(1)} mi` : '--'}</strong>
              <small>Across valid distance records</small>
            </article>
            <article className="patterns-metric">
              <div><span>AVERAGE TRIP DURATION</span><TrendingUp size={17} /></div>
              <strong>{metrics ? `${metrics.average_duration_minutes.toFixed(1)} min` : '--'}</strong>
              <small>Across valid duration records</small>
            </article>
          </div>

          <section className="patterns-panel">
            <div className="patterns-panel-heading">
              <div>
                <span>01 / DEMAND PROFILE</span>
                <h2>When does New York move?</h2>
                <p>Trip volume by pickup hour across the day</p>
              </div>
              <span className="patterns-unit">TRIPS / HOUR</span>
            </div>
            <ReactECharts option={demandOption} style={{ height: 330 }} notMerge lazyUpdate />
            {peakHour && (
              <div className="patterns-observation">
                <Activity size={17} />
                <p>
                  Demand peaks at <strong>{String(peakHour.group_value).padStart(2, '0')}:00</strong>,
                  with <strong>{peakHour.trip_count.toLocaleString('en-US')}</strong> trips.
                  This is the busiest hour in the January 2025 dataset.
                </p>
              </div>
            )}
          </section>

          <section className="patterns-panel">
            <div className="patterns-panel-heading">
              <div>
                <span>02 / REVENUE DISTRIBUTION</span>
                <h2>Where is fare revenue concentrated?</h2>
                <p>Recorded fare revenue grouped by pickup borough</p>
              </div>
              <span className="patterns-unit">USD</span>
            </div>
            <ReactECharts option={revenueOption} style={{ height: 300 }} notMerge lazyUpdate />
            <div className="patterns-revenue-list">
              {boroughRevenue.slice(0, 3).map((item) => (
                <div key={item.group_value}>
                  <span>{item.group_value}</span>
                  <strong>{money(item.fare_revenue)}</strong>
                  <small>{item.valid_fare_trips.toLocaleString('en-US')} valid fares</small>
                </div>
              ))}
            </div>
            <p className="patterns-footnote">
              Fare revenue is calculated from valid fare records. It excludes tips, tolls and other additional charges.
            </p>
          </section>
        </>
      )}

      <footer className="patterns-footer">
        <span>NYC TLC YELLOW TAXI DATASET</span>
        <span>HISTORICAL COVERAGE: JAN 01 - JAN 31, 2025</span>
      </footer>
    </div>
  )
}

export default PatternsPage