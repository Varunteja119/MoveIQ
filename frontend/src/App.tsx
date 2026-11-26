import { useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CalendarDays,
  ChevronDown,
  Clock3,
  Compass,
  Layers3,
  History,
  MapPin,
  Search,
  Settings2,
  Sparkles,
  TrendingUp,
  Zap,
} from 'lucide-react'
import './App.css'
import CityMap from './components/CityMap'
import PatternsPage from './pages/PatternsPage'
import HistoricalInsights from './pages/HistoricalInsights'
import ForecastPage from './pages/ForecastPage'
import CopilotPage from './pages/CopilotPage'

const navigation = [
  { id: 'explore', label: 'Explore', icon: Compass },
  { id: 'patterns', label: 'Patterns', icon: BarChart3 },
  { id: 'historical', label: 'Historical Insights', icon: History },
  { id: 'forecast', label: 'Forecast', icon: TrendingUp },
  { id: 'copilot', label: 'AI Copilot', icon: BrainCircuit },
]

function App() {
  const root = useRef<HTMLDivElement>(null)
  const [activePage, setActivePage] = useState('explore')
  const [activeLayer, setActiveLayer] = useState('Pickup demand')
  const [selectedZone, setSelectedZone] = useState({
    id: 161,
    name: 'Midtown Center',
    borough: 'Manhattan',
  })

  useGSAP(
    () => {
      gsap.fromTo(
        '.reveal',
        { opacity: 0, y: 14 },
        { opacity: 1, y: 0, duration: 0.7, stagger: 0.09, ease: 'power3.out' },
      )
    },
    { scope: root },
  )

  const pageTitle =
    navigation.find((item) => item.id === activePage)?.label ?? 'Explore'

  return (
    <main className="app-shell" ref={root}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={21} strokeWidth={2.2} />
          </div>
          <div className="brand-copy">
            <span className="brand-name">MOVE<span>IQ</span></span>
            <span className="brand-caption">MOBILITY INTELLIGENCE</span>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="section-label">WORKSPACE</span>
          <nav className="main-nav" aria-label="Main navigation">
            {navigation.map(({ id, label, icon: Icon }) => (
              <button
                className={`nav-item ${activePage === id ? 'active' : ''}`}
                key={id}
                onClick={() => setActivePage(id)}
                type="button"
              >
                <Icon size={18} />
                <span>{label}</span>
                {id === 'copilot' && <span className="nav-new">AI</span>}
              </button>
            ))}
          </nav>
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <span>Analytics engine ready</span>
          </div>
          <button className="nav-item settings-button" type="button">
            <Settings2 size={18} />
            <span>Settings</span>
          </button>
          <div className="profile">
            <div className="profile-avatar">M</div>
            <div className="profile-copy">
              <strong>MoveIQ Workspace</strong>
              <span>Local environment</span>
            </div>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <span>WORKSPACE</span>
            <span className="breadcrumb-slash">/</span>
            <strong>{pageTitle.toUpperCase()}</strong>
          </div>
          <div className="topbar-actions">
            <div className="dataset-status">
              <span className="status-dot" />
              HISTORICAL DATA
            </div>
            <button className="icon-button" aria-label="Search" type="button">
              <Search size={18} />
            </button>
            <button className="icon-button" aria-label="Help" type="button">
              <Sparkles size={18} />
            </button>
          </div>
        </header>

        <div className="content">
          {activePage === 'forecast' ? <ForecastPage /> : activePage === 'copilot' ? <CopilotPage /> : activePage === 'patterns' ? <PatternsPage /> :
            activePage === 'historical' ? (
              <HistoricalInsights
                initialZoneId={selectedZone.id}
                onZoneChange={(id) => setSelectedZone((current) => ({ ...current, id }))}
              />
            ) : <>
          <div className="page-heading reveal">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                URBAN OPERATIONS INTELLIGENCE
              </div>
              <h1>The city, <span>in motion.</span></h1>
              <p className="page-description">
                Explore the patterns behind New York's mobility network.
              </p>
            </div>
            <button className="date-selector" type="button">
              <CalendarDays size={16} />
              January 2025
              <ChevronDown size={15} />
            </button>
          </div>

          <section className="metrics-row reveal" aria-label="Dataset overview">
            <article className="metric-card">
              <div className="metric-top">
                <span>TOTAL TRIPS</span>
                <Activity size={16} />
              </div>
              <div className="metric-value">3.47<span>M</span></div>
              <div className="metric-foot">NYC yellow taxi  -  January</div>
            </article>
            <article className="metric-card">
              <div className="metric-top">
                <span>PICKUP ZONES</span>
                <MapPin size={16} />
              </div>
              <div className="metric-value">261</div>
              <div className="metric-foot">Across New York City</div>
            </article>
            <article className="metric-card">
              <div className="metric-top">
                <span>VALID FARE RECORDS</span>
                <Zap size={16} />
              </div>
              <div className="metric-value">3.33<span>M</span></div>
              <div className="metric-foot">Validated analytics records</div>
            </article>
            <article className="metric-card accent-metric">
              <div className="metric-top">
                <span>TOP PICKUP ZONE</span>
                <TrendingUp size={16} />
              </div>
              <div className="metric-value metric-name">Midtown Center</div>
              <div className="metric-foot">169,977 trips  -  Manhattan</div>
            </article>
          </section>

          <section className="city-section reveal">
            <div className="section-heading">
              <div>
                <div className="section-title">
                  <span className="live-indicator" />
                  CITY INTELLIGENCE
                </div>
                <p>New York mobility network</p>
              </div>
              <label className="layer-selector">
                <Layers3 size={15} />
                <select
                  value={activeLayer}
                  onChange={(event) => setActiveLayer(event.target.value)}
                  aria-label="Select map layer"
                >
                  <option>Pickup demand</option>
                  <option>Dropoff demand</option>
                  <option>Zone rankings</option>
                </select>
                <ChevronDown size={14} />
              </label>
            </div>

            <CityMap
              layer={activeLayer}
              onZoneSelect={setSelectedZone}
              onViewHistorical={() => setActivePage('historical')}
            />
          </section>

          <section className="bottom-grid reveal">
            <article className="insight-card">
              <div className="card-heading">
                <div>
                  <span className="section-title">MOBILITY SIGNAL</span>
                  <p>Highest pickup activity</p>
                </div>
                <ArrowUpRight size={17} />
              </div>
              <div className="insight-main">
                <div>
                  <span className="insight-zone">Midtown Center</span>
                  <span className="insight-location">Manhattan  -  Zone 161</span>
                </div>
                <strong>169,977</strong>
              </div>
              <div className="insight-foot">
                <span className="signal-bar"><i /></span>
                <span>January 2025  -  Historical</span>
              </div>
            </article>

            <article className="copilot-card">
              <div className="copilot-icon"><BrainCircuit size={20} /></div>
              <div className="copilot-copy">
                <span className="section-title">MOVEIQ COPILOT</span>
                <p>Ask the city a question.</p>
                <span>Investigate patterns using natural language.</span>
              </div>
              <button
                className="copilot-action"
                onClick={() => setActivePage('copilot')}
                type="button"
                aria-label="Open AI Copilot"
              >
                <ArrowUpRight size={18} />
              </button>
            </article>
          </section>

          <footer className="workspace-footer">
            <span><Clock3 size={13} /> DATA COVERAGE: JAN 01  -  JAN 31, 2025</span>
            <span>NYC TLC YELLOW TAXI DATASET</span>
          </footer>
          </>}
        </div>
      </section>
    </main>
  )
}

export default App


