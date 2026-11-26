import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import { setWorkerUrl } from 'maplibre-gl'
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import 'maplibre-gl/dist/maplibre-gl.css'

setWorkerUrl(maplibreWorkerUrl)

type ZoneFeature = {
  type: 'Feature'
  geometry: {
    type: 'Polygon' | 'MultiPolygon'
    coordinates: number[][][] | number[][][][]
  }
  properties: Record<string, unknown>
}

type ZoneCollection = {
  type: 'FeatureCollection'
  features: ZoneFeature[]
}

type RankingRecord = {
  location_id: number
  zone_name: string
  borough: string
  trip_count: number
}

type SelectedZone = {
  id: number
  name: string
  borough: string
  trips: number
  rank: number | null
  hasData: boolean
}

type CityMapProps = {
  layer: string
  onZoneSelect?: (zone: { id: number; name: string; borough: string }) => void
  onViewHistorical?: () => void
}

function CityMap({ layer, onZoneSelect, onViewHistorical }: CityMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [zones, setZones] = useState<ZoneCollection | null>(null)
  const [selected, setSelected] = useState<SelectedZone | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!containerRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [-73.95, 40.72],
      zoom: 9,
      minZoom: 8,
      maxZoom: 16,
    })

    mapRef.current = map
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: true }),
      'top-right',
    )

    map.on('load', () => {
      setMapReady(true)
    })


    const resizeObserver = new ResizeObserver(() => map.resize())
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()

    fetch('/data/nyc-taxi-zones.geojson', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not load NYC taxi-zone boundaries.')
        return response.json() as Promise<ZoneCollection>
      })
      .then(setZones)
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name !== 'AbortError') {
          setError(cause.message)
        }
      })

    return () => controller.abort()
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReady || !zones) return

    const controller = new AbortController()
    setLoading(true)
    setError('')

    const kind = layer === 'Dropoff demand' ? 'dropoff' : 'pickup'

    fetch(`/api/v1/analytics/zones?kind=${kind}&limit=300`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not retrieve zone demand from the API.')
        return response.json() as Promise<{ data: RankingRecord[] }>
      })
      .then((payload) => {
        const rankings = new Map(
          payload.data.map((record, index) => [
            Number(record.location_id),
            { ...record, rank: index + 1 },
          ]),
        )

        const features = zones.features.map((feature) => {
          const id = Number(feature.properties.locationid)
          const record = rankings.get(id)

          return {
            ...feature,
            properties: {
              ...feature.properties,
              trip_count: record?.trip_count ?? 0,
              rank: record?.rank ?? 0,
              has_data: Boolean(record),
              zone_name: record?.zone_name ?? String(feature.properties.zone ?? 'Unknown zone'),
              borough: record?.borough ?? String(feature.properties.borough ?? 'Unknown'),
            },
          }
        })

        const enriched: ZoneCollection = {
          type: 'FeatureCollection',
          features,
        }

        // Refresh the selected zone using the newly loaded layer data.
        setSelected((current) => {
          if (!current) return current

          const matchingFeature = enriched.features.find(
            (feature) => Number(feature.properties.locationid) === current.id,
          )

          if (!matchingFeature) return current

          const properties = matchingFeature.properties

          return {
            ...current,
            name: String(properties.zone_name ?? current.name),
            borough: String(properties.borough ?? current.borough),
            trips: Number(properties.trip_count ?? 0),
            rank: Number(properties.rank) || null,
            hasData: Boolean(properties.has_data),
          }
        })

        const maxDemand = Math.max(
          1,
          ...payload.data.map((record) => record.trip_count),
        )

        const source = map.getSource('taxi-zones') as maplibregl.GeoJSONSource | undefined

        if (source) {
          source.setData(enriched as never)
        } else {
          map.addSource('taxi-zones', {
            type: 'geojson',
            data: enriched as never,
          })

          map.addLayer({
            id: 'zone-fill',
            type: 'fill',
            source: 'taxi-zones',
            paint: {
              'fill-color': layer === 'Zone rankings'
                ? [
                    'case',
                    ['==', ['get', 'has_data'], false], '#202b38',
                    ['<=', ['get', 'rank'], 10], '#27c9e8',
                    ['<=', ['get', 'rank'], 25], '#287fa8',
                    '#254052',
                  ]
                : [
                    'case',
                    ['==', ['get', 'has_data'], false], '#202b38',
                    [
                      'interpolate',
                      ['linear'],
                      ['get', 'trip_count'],
                      0, '#142536',
                      maxDemand * 0.15, '#15506a',
                      maxDemand * 0.4, '#147e9b',
                      maxDemand * 0.7, '#22b8d2',
                      maxDemand, '#9bf2ff',
                    ],
                  ],
              'fill-opacity': 0.84,
            },
          })

          map.addLayer({
            id: 'zone-outline',
            type: 'line',
            source: 'taxi-zones',
            paint: {
              'line-color': '#83b9d0',
              'line-width': 0.85,
              'line-opacity': 0.68,
            },
          })

          map.addLayer({
            id: 'zone-selected-outline',
            type: 'line',
            source: 'taxi-zones',
            filter: ['==', ['get', 'locationid'], '__none__'],
            paint: {
              'line-color': '#31D7F5',
              'line-width': 3.5,
              'line-opacity': 1,
            },
          })

          map.on('click', 'zone-fill', (event: maplibregl.MapLayerMouseEvent) => {
            const feature = event.features?.[0]
            if (!feature?.properties) return

            const properties = feature.properties
            const id = Number(properties.locationid)

            const zoneName = String(properties.zone_name ?? properties.zone ?? 'Unknown zone')
            const borough = String(properties.borough ?? 'Unknown')

            setSelected({
              id,
              name: zoneName,
              borough,
              trips: Number(properties.trip_count ?? 0),
              rank: Number(properties.rank) || null,
              hasData: Boolean(properties.has_data),
            })

            onZoneSelect?.({ id, name: zoneName, borough })

            map.setFilter('zone-selected-outline', [
              '==',
              ['to-number', ['get', 'locationid']],
              id,
            ])
          })

          map.on('mouseenter', 'zone-fill', () => {
            map.getCanvas().style.cursor = 'pointer'
          })

          map.on('mouseleave', 'zone-fill', () => {
            map.getCanvas().style.cursor = ''
          })

          map.fitBounds(
            [[-74.27, 40.49], [-73.68, 40.92]],
            { padding: 35, duration: 900 },
          )
        }

        if (layer === 'Zone rankings') {
          map.setPaintProperty('zone-fill', 'fill-color', [
            'case',
            ['==', ['get', 'has_data'], false], '#202b38',
            ['<=', ['get', 'rank'], 10], '#27c9e8',
            ['<=', ['get', 'rank'], 25], '#287fa8',
            '#254052',
          ])
        } else {
          map.setPaintProperty('zone-fill', 'fill-color', [
            'case',
            ['==', ['get', 'has_data'], false], '#202b38',
            [
              'interpolate',
              ['linear'],
              ['get', 'trip_count'],
              0, '#142536',
              maxDemand * 0.15, '#15506a',
              maxDemand * 0.4, '#147e9b',
              maxDemand * 0.7, '#22b8d2',
              maxDemand, '#9bf2ff',
            ],
          ])
        }

        setLoading(false)
      })
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name !== 'AbortError') {
          setError(cause.message)
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [layer, mapReady, zones, onZoneSelect])

  return (
    <div className="map-stage real-map-stage">
      <div className="map-canvas" ref={containerRef} />

      <div className="map-data-status">
        <span className="map-status-dot" />
        {loading ? 'LOADING ZONE DATA' : 'JANUARY 2025 | HISTORICAL'}
      </div>

      {error && <div className="map-error">{error}</div>}

      <div className="map-legend real-map-legend">
        <span className="legend-gradient" />
        <span>{layer === 'Zone rankings' ? 'TOP 10' : 'LOW'} </span>
        <span className="legend-gradient-high" />
        <span>{layer === 'Zone rankings' ? 'TOP 25' : 'HIGH DEMAND'}</span>
      </div>

      {selected && (
        <aside className="zone-inspector">
          <button
            className="zone-inspector-close"
            type="button"
            onClick={() => {
              setSelected(null)
              mapRef.current?.setFilter('zone-selected-outline', [
                '==',
                ['get', 'locationid'],
                '__none__',
              ])
            }}
            aria-label="Close zone details"
          >
            x
          </button>
          <span className="zone-inspector-label">SELECTED TAXI ZONE</span>
          <strong>{selected.name}</strong>
          <span className="zone-inspector-borough">{selected.borough} - Zone {selected.id}</span>
          {selected.hasData ? (
            <>
              <strong className="zone-inspector-trips">
                {selected.trips.toLocaleString('en-US')}
              </strong>
              <span className="zone-inspector-caption">
                {kindLabel(layer)} trips - January 2025
              </span>
              {selected.rank && (
                <span className="zone-inspector-rank">Rank #{selected.rank}</span>
              )}
            </>
          ) : (
            <span className="zone-inspector-caption">
              No {kindLabel(layer).toLowerCase()} records in this dataset
            </span>
          )}
          <button
            className="zone-inspector-history"
            type="button"
            onClick={onViewHistorical}
          >
            View historical insights
          </button>
        </aside>
      )}
    </div>
  )
}

function kindLabel(layer: string) {
  return layer === 'Dropoff demand' ? 'Drop-off' : 'Pickup'
}

export default CityMap
