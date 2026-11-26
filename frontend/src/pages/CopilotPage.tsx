import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useState, useEffect } from 'react'
import { BrainCircuit, RotateCcw, Send, Sparkles, User } from 'lucide-react'
import './CopilotPage.css'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type SessionResponse = {
  session_id: string
  expires_in_seconds: number
}

type AskResponse = {
  session_id: string
  answer: string
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
    setMessages(current => [...current, { role: 'user', content: trimmed }])

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
        { role: 'assistant', content: result.answer },
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
                  <div className="copilot-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>
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
