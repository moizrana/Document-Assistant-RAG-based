import { useState, useEffect, useRef } from 'react'
import { Send, Brain, User, Trash2, CheckCircle2, Plus, ChevronRight, ChevronDown, Loader2, Search, Shield, Gauge, RefreshCcw, Sparkles } from 'lucide-react'
import { v4 as uuidv4 } from 'uuid'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api").replace(/\/+$/, "")

const FEATURE_CARDS = [
  {
    icon: Search,
    title: "Hybrid Search",
    description: "FAISS dense + BM25 sparse retrieval with reciprocal rank fusion."
  },
  {
    icon: Brain,
    title: "Cross-Encoder Reranking",
    description: "Precision reranking using transformer cross-encoders."
  },
  {
    icon: Shield,
    title: "Guardrails",
    description: "Prompt-injection protection with safer response validation."
  },
  {
    icon: Gauge,
    title: "Semantic Caching",
    description: "Faster answers for semantically similar questions."
  },
  {
    icon: RefreshCcw,
    title: "Query Expansion",
    description: "Multi-query retrieval to improve recall and context quality."
  },
  {
    icon: Sparkles,
    title: "Evaluation Metrics",
    description: "Faithfulness, relevancy, and context scoring signals."
  }
]

const SourceExpander = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  if (!sources || sources.length === 0) return null;
  
  return (
    <div className="sources-container">
      <div className="source-header" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>📚 Sources ({sources.length})</span>
      </div>
      
      {isOpen && (
        <div className="source-content-wrapper">
          {sources.map((src, i) => (
            <div key={i} className="source-card">
              <h4>📄 {src.source} {src.page ? `| Page ${src.page}` : ''}</h4>
              <p>{src.content.substring(0, 150)}...</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const EvalScores = ({ scores }) => {
  const [isOpen, setIsOpen] = useState(false)
  if (!scores) return null

  const entries = Object.entries(scores).filter(([, v]) => typeof v === 'number')
  if (entries.length === 0) return null

  const labelMap = {
    faithfulness: "Faithfulness",
    answer_relevancy: "Answer relevancy",
    context_relevancy: "Context relevancy"
  }

  return (
    <div className="eval-container">
      <div className="eval-header" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>📊 Evaluation ({entries.length})</span>
      </div>

      {isOpen && (
        <div className="eval-grid">
          {entries.map(([k, v]) => (
            <div key={k} className="eval-chip">
              <span className="eval-label">{labelMap[k] || k}</span>
              <span className="eval-value">{Number(v).toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function App() {
  const [sessionId, setSessionId] = useState("")
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [isThinking, setIsThinking] = useState(false)
  
  // Settings
  const [useEval, setUseEval] = useState(true)
  const [useCache, setUseCache] = useState(true)
  const [useQueryExp, setUseQueryExp] = useState(false)
  
  // File Upload State
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [pendingFiles, setPendingFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  
  const messagesEndRef = useRef(null)
  const chatContainerRef = useRef(null)
  const showFeatureCards = true

  useEffect(() => {
    // Generate or retrieve Session ID
    let sid = localStorage.getItem('documind_session_id')
    if (!sid) {
      sid = uuidv4()
      localStorage.setItem('documind_session_id', sid)
    }
    setSessionId(sid)
    
    // Keep chat empty initially so feature cards are visible on load.
    setMessages([])
  }, [])

  useEffect(() => {
    if (messages.length === 0) {
      chatContainerRef.current?.scrollTo({ top: 0, behavior: 'auto' })
      return
    }
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleFileUpload = async (e) => {
    const filesArray = Array.from(e.target.files)
    if (!filesArray.length) return

    setIsUploading(true)
    setPendingFiles(filesArray.map(f => f.name))
    
    const formData = new FormData()
    formData.append('session_id', sessionId)
    for (let i = 0; i < filesArray.length; i++) {
      formData.append('files', filesArray[i])
    }

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (res.ok) {
        setUploadedFiles(data.total_files || [])
      } else {
        alert(data.detail || "Upload failed")
      }
    } catch (err) {
      console.error(err)
      alert("Error connecting to server")
    } finally {
      setIsUploading(false)
      setPendingFiles([])
      e.target.value = null
    }
  }

  const handleClearChat = () => {
    setMessages([])
  }

  const handleDeleteSession = async () => {
    try {
      await fetch(`${API_BASE}/session`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      })
    } catch (err) {
      console.error(err)
    } finally {
      setMessages([])
      setUploadedFiles([])
      setPendingFiles([])
      setInput("")
      setIsThinking(false)
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isThinking) return

    const userMsg = input.trim()
    setInput("")
    
    // Add User Message
    setMessages(prev => [...prev, { role: 'user', content: userMsg, isComplete: true }])
    
    // Add empty Assistant Message placeholder
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      content: "", 
      isComplete: false,
      sources: null
    }])
    
    setIsThinking(true)

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMsg,
          session_id: sessionId,
          use_query_expansion: useQueryExp,
          use_evaluation: useEval,
          use_cache: useCache
        })
      })

      // Read SSE Stream
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let done = false

      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        if (value) {
          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n')
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6).trim()
              if (!dataStr) continue
              
              try {
                const data = JSON.parse(dataStr)
                if (data.chunk) {
                  setMessages(prev => {
                    const newMsgs = [...prev]
                    const lastMsg = { ...newMsgs[newMsgs.length - 1] }
                    lastMsg.content += data.chunk
                    newMsgs[newMsgs.length - 1] = lastMsg
                    return newMsgs
                  })
                }
                if (data.final) {
                  setMessages(prev => {
                    const newMsgs = [...prev]
                    const lastMsg = { ...newMsgs[newMsgs.length - 1] }
                    lastMsg.content = data.final.answer
                    lastMsg.sources = data.final.sources
                    lastMsg.evalScores = data.final.eval_scores
                    lastMsg.isComplete = true
                    newMsgs[newMsgs.length - 1] = lastMsg
                    return newMsgs
                  })
                }
                if (data.error) {
                   setMessages(prev => {
                    const newMsgs = [...prev]
                    const lastMsg = { ...newMsgs[newMsgs.length - 1] }
                    lastMsg.content = `Error: ${data.error}`
                    lastMsg.isComplete = true
                    newMsgs[newMsgs.length - 1] = lastMsg
                    return newMsgs
                  })
                }
              } catch (e) {
                // Ignore parse errors from partial lines in stream
              }
            }
          }
        }
      }
    } catch (err) {
      console.error(err)
      setMessages(prev => {
        const newMsgs = [...prev]
        newMsgs[newMsgs.length - 1].content = "Connection error."
        newMsgs[newMsgs.length - 1].isComplete = true
        return newMsgs
      })
    } finally {
      setIsThinking(false)
    }
  }

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!isThinking && uploadedFiles.length > 0 && input.trim()) {
        sendMessage(e)
      }
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <Brain size={40} className="brand-icon" />
          <h2 className="brand-title">DocuMind AI</h2>
          <p>Intelligent Document Assistant</p>
        </div>

        <div className="upload-section">
          <label className="upload-zone">
            <input 
              type="file" 
              multiple 
              onChange={handleFileUpload} 
              style={{ display: 'none' }} 
              accept=".pdf,.txt,.html,.md"
            />
            {isUploading ? (
              <Loader2 className="spin-anim" size={28} color="var(--accent-color)" />
            ) : (
              <Plus size={28} color="var(--accent-color)" />
            )}
            <p>{isUploading ? "Uploading & Processing..." : "Click to upload documents"}</p>
          </label>
          
          {/* Pending Files */}
          {pendingFiles.length > 0 && (
            <div style={{marginTop: '1rem', fontSize: '0.8rem'}}>
              <p style={{color: 'var(--accent-color)', marginBottom: '0.5rem'}}>⏳ Processing Files:</p>
              {pendingFiles.map((f, i) => (
                <div key={i} style={{marginBottom: '0.75rem'}}>
                  <div style={{display:'flex', alignItems:'center', gap:'0.5rem', color:'var(--text-secondary)'}}>
                    <Loader2 size={14} className="spin-anim" color="#A855F7"/> {f}
                  </div>
                  <div className="progress-bar-container">
                    <div className="progress-bar"></div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Uploaded Files */}
          {uploadedFiles.length > 0 && (
            <div style={{marginTop: '1rem', fontSize: '0.8rem'}}>
              <p style={{color: 'var(--accent-color)', marginBottom: '0.5rem'}}>📚 Indexed Files:</p>
              {uploadedFiles.map((f, i) => (
                <div key={i} style={{display:'flex', alignItems:'center', gap:'0.5rem', color:'var(--text-secondary)'}}>
                  <CheckCircle2 size={14} color="#22C55E"/> {f}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="settings-section">
          <h3>⚙️ Pipeline Settings</h3>
          
          <div className="toggle-row">
            <span>📈 Evaluation Metrics</span>
            <label className="switch">
              <input type="checkbox" checked={useEval} onChange={e => setUseEval(e.target.checked)} />
              <span className="slider"></span>
            </label>
          </div>
          <div className="toggle-row">
            <span>⚡ Semantic Cache</span>
            <label className="switch">
              <input type="checkbox" checked={useCache} onChange={e => setUseCache(e.target.checked)} />
              <span className="slider"></span>
            </label>
          </div>
          <div className="toggle-row">
            <span>🔄 Query Expansion</span>
            <label className="switch">
              <input type="checkbox" checked={useQueryExp} onChange={e => setUseQueryExp(e.target.checked)} />
              <span className="slider"></span>
            </label>
          </div>
        </div>

        <button className="btn-clear" onClick={handleDeleteSession}>
          <Trash2 size={16} style={{display:'inline', verticalAlign:'text-bottom', marginRight:'0.5rem'}}/> 
          Delete Chat
        </button>
      </aside>

      {/* Main Chat Area */}
      <main className="main-content">
        <header className="header glass-panel">
          <h1>Intelligent Document Assistant</h1>
          <p>Upload documents and ask questions — powered by RAG with hybrid search & reranking</p>
        </header>

        <div className="chat-container" ref={chatContainerRef}>
          {showFeatureCards && (
            <section className="features-container">
              <div className="features-grid">
                {FEATURE_CARDS.map((feature) => {
                  const Icon = feature.icon
                  return (
                    <article key={feature.title} className="feature-card">
                      <div className="feature-icon">
                        <Icon size={24} />
                      </div>
                      <h3 className="feature-title">{feature.title}</h3>
                      <p className="feature-desc">{feature.description}</p>
                    </article>
                  )
                })}
              </div>
            </section>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="avatar">
                {msg.role === 'user' ? <User size={20} /> : <Brain size={20} />}
              </div>
              <div className="message-content">
                <div>
                  {!msg.isComplete && msg.content === "" && isThinking ? (
                    <div className="thinking-indicator"> Thinking...</div>
                  ) : (
                    <>
                      {msg.content}
                      {!msg.isComplete && <span className="typing-cursor"></span>}
                    </>
                  )}
                </div>
                
                {/* Sources Expander */}
                <SourceExpander sources={msg.sources} />
                <EvalScores scores={msg.evalScores} />
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={sendMessage} className="input-box">
            <textarea
              placeholder={uploadedFiles.length === 0 ? "Please upload documents first..." : "Ask a question about your documents..."}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              disabled={isThinking || uploadedFiles.length === 0}
              rows={1}
            />
            <button type="submit" className="send-btn" disabled={!input.trim() || isThinking || uploadedFiles.length === 0}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
