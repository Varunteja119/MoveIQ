
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ReactECharts from 'echarts-for-react'
import { useState, useEffect } from 'react'
import { BrainCircuit, RotateCcw, Send, Sparkles, User } from 'lucide-react'
import './CopilotPage.css'

type ChartData = {
  chart_type: 'bar' | 'line' | 'pie'
  title: string
  x_key: string
  series: {
    data_key: string
    label: string
  }[]
  data: {
    category: string
    value: number
  }[]
  metadata?: {
    metric?: string
    group_by?: string
    filters?: Record<string, unknown>
    record_count?: number
  }
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  charts?: ChartData[]
}

type SessionResponse = {
  session_id: string
  expires_in_seconds: number
}

type AskResponse = {
  session_id: string
  answer: string
  charts?: ChartData[]
  expires_in_seconds: number
}

const SESSION_STORAGE_KEY = 'moveiq_copilot_session'
const MESSAGES_STORAGE_KEY = 'moveiq_copilot_messages'

async function readError(response: Response) {
  try {
    const body = await response.json()
    return typeof body.detail === 'string'
      ? body.detail
      : `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

function CopilotPage() {
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem(SESSION_STORAGE_KEY) || ''
  )

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = sessionStorage.getItem(MESSAGES_STORAGE_KEY)
      const parsed = saved ? JSON.parse(saved) : []
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  })

  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId)
  }, [sessionId])

  useEffect(() => {
    sessionStorage.setItem(MESSAGES_STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  async function createSession() {
    const response = await fetch('/api/v1/copilot/sessions', {
      method: 'POST',
    })

    if (!response.ok) throw new Error(await readError(response))

    const result = await response.json() as SessionResponse

    sessionStorage.setItem(SESSION_STORAGE_KEY, result.session_id)
    setSessionId(result.session_id)

    return result.session_id
  }

  async function sendQuestion(value: string) {
    const trimmed = value.trim()
    if (trimmed.length < 3 || loading) return

    setLoading(true)
    setError('')
    setQuestion('')
    setMessages(current => [
      ...current,
      { role: 'user', content: trimmed },
    ])

    try {
      const activeSession = sessionId || await createSession()

      const response = await fetch('/api/v1/copilot/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession,
          question: trimmed,
        }),
      })

      if (!response.ok) throw new Error(await readError(response))

      const result = await response.json() as AskResponse

      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          content: result.answer,
          charts: result.charts ?? [],
        },
      ])
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to contact MoveIQ Copilot.'
      )
    } finally {
      setLoading(false)
    }
  }

  async function resetConversation() {
    setError('')
    setLoading(true)

    try {
      if (sessionId) {
        const response = await fetch(
          `/api/v1/copilot/sessions/${sessionId}/reset`,
          { method: 'POST' }
        )

        if (!response.ok) throw new Error(await readError(response))
      }

      setMessages([])
      sessionStorage.removeItem(MESSAGES_STORAGE_KEY)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to reset conversation.'
      )
    } finally {
      setLoading(false)
    }
  }

  function renderChart(chart: ChartData, chartIndex: number) {
    const option = chart.chart_type === 'pie'
      ? {
          title: {
            text: chart.title,
            left: 'center',
          },
          tooltip: {
            trigger: 'item' as const,
          },
          legend: {
            bottom: 0,
          },
          series: [
            {
              name: chart.series[0]?.label ?? 'Value',
              type: 'pie' as const,
              radius: '60%',
              data: chart.data.map(item => ({
                name: item.category,
                value: item.value,
              })),
            },
          ],
        }
      : {
          title: {
            text: chart.title,
            left: 'center',
          },
          tooltip: {
            trigger: 'axis' as const,
          },
          grid: {
            left: 55,
            right: 25,
            top: 65,
            bottom: 75,
          },
          xAxis: {
            type: 'category' as const,
            data: chart.data.map(item => item.category),
            axisLabel: {
              rotate: chart.data.length > 8 ? 35 : 0,
            },
          },
          yAxis: {
            type: 'value' as const,
          },
          series: [
            {
              name: chart.series[0]?.label ?? 'Value',
              type: chart.chart_type,
              data: chart.data.map(item => item.value),
              smooth: chart.chart_type === 'line',
            },
          ],
        }

    return (
      <div className="copilot-chart" key={chartIndex}>
        <ReactECharts
          option={option}
          style={{
            height: 360,
            width: '100%',
          }}
          opts={{
            renderer: 'canvas',
          }}
          notMerge
        />
      </div>
    )
  }

  const suggestions = [
    'What are the busiest pickup hours in New York?',
    'Which areas have the highest taxi demand?',
    'Explain the main mobility patterns in the dataset.',
  ]

  return (
    <div className="copilot-page">
      <div className="copilot-heading">
        <div>
          <div className="copilot-eyebrow">
            <span /> MOVEIQ INTELLIGENCE ASSISTANT
          </div>
          <h1>Ask the <span>city anything.</span></h1>
          <p>
            Explore mobility data and get operational insights using natural language.
          </p>
        </div>

        <button
          className="copilot-reset"
          onClick={() => void resetConversation()}
          disabled={loading}
        >
          <RotateCcw size={15} /> New conversation
        </button>
      </div>

      <section className="copilot-chat">
        <div className="copilot-chat-header">
          <div className="copilot-avatar">
            <BrainCircuit size={20} />
          </div>
          <div>
            <strong>MoveIQ Copilot</strong>
            <span><i /> AI assistant · Mobility operations</span>
          </div>
          <Sparkles className="copilot-sparkle" size={19} />
        </div>

        <div className="copilot-messages">
          {messages.length === 0 && (
            <div className="copilot-welcome">
              <div className="copilot-welcome-icon">
                <BrainCircuit size={27} />
              </div>
              <h2>What would you like to understand?</h2>
              <p>
                Ask questions about New York taxi demand, pickup zones,
                historical patterns and forecasts.
              </p>

              <div className="copilot-suggestions">
                {suggestions.map(item => (
                  <button
                    key={item}
                    type="button"
                    disabled={loading}
                    onClick={() => void sendQuestion(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              className={`copilot-message ${message.role}`}
              key={`${index}-${message.role}`}
            >
              <div className="copilot-message-icon">
                {message.role === 'user'
                  ? <User size={16} />
                  : <BrainCircuit size={16} />}
              </div>

              <div className="copilot-message-body">
                <span>
                  {message.role === 'user' ? 'YOU' : 'MOVEIQ COPILOT'}
                </span>

                {message.role === 'user' ? (
                  <p>{message.content}</p>
                ) : (
                  <>
                    <div className="copilot-markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>

                    {message.charts?.map((chart, chartIndex) =>
                      renderChart(chart, chartIndex)
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="copilot-typing">
              <BrainCircuit size={16} />
              MoveIQ is analysing your question...
            </div>
          )}
        </div>

        {error && <div className="copilot-error">{error}</div>}

        <form
          className="copilot-composer"
          onSubmit={event => {
            event.preventDefault()
            void sendQuestion(question)
          }}
        >
          <textarea
            value={question}
            onChange={event => setQuestion(event.target.value)}
            placeholder="Ask about demand, zones, trends or forecasts..."
            rows={2}
            maxLength={4000}
            disabled={loading}
          />

          <div className="copilot-composer-footer">
            <span>MoveIQ uses your session conversation context.</span>
            <button
              type="submit"
              disabled={loading || question.trim().length < 3}
            >
              <Send size={15} /> Send
            </button>
          </div>
        </form>
      </section>

      <footer className="copilot-footer">
        <span>POWERED BY MOVEIQ AI AGENT</span>
        <span>Conversation sessions are stored temporarily</span>
      </footer>
    </div>
  )
}

export default CopilotPage
