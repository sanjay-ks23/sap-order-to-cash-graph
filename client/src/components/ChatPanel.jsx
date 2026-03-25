import { useState, useRef, useEffect } from 'react';
import { streamChat } from '../api/client';

const QUICK_QUERIES = [
  'Which customer has the most orders?',
  'Show me incomplete order flows',
  'Trace billing document 90504248',
  'Average order value',
  'Orders for Nelson, Fitzpatrick and Jordan',
];

export default function ChatPanel({ onHighlightNodes }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I can help you analyze the **Order to Cash** process.',
    },
  ]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const messagesRef = useRef(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, streamStatus]);

  const sendMessage = async (text) => {
    const query = text || input.trim();
    if (!query || streaming) return;

    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setStreaming(true);
    setStreamStatus('Classifying intent...');

    const conversationHistory = messages
      .filter((m) => m.role !== 'system')
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      let finalAnswer = '';
      let finalMeta = null;
      let finalStructured = null;

      for await (const event of streamChat(query, conversationHistory)) {
        switch (event.type) {
          case 'status':
            setStreamStatus(event.message);
            break;
          case 'intent':
            setStreamStatus(`Intent: ${event.intent} (${event.llm_ms?.toFixed(0)}ms)`);
            break;
          case 'result':
            if (event.meta?.nodes_traversed?.length) {
              onHighlightNodes?.(event.meta.nodes_traversed);
            }
            break;
          case 'answer':
            finalAnswer = event.content;
            finalMeta = event.meta;
            finalStructured = event.structured || null;
            break;
          case 'done':
            break;
        }
      }

      const assistantMsg = {
        role: 'assistant',
        content: finalAnswer || 'No response received.',
        meta: finalMeta,
        structured: finalStructured,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${e.message}`, isError: true },
      ]);
    }

    setStreaming(false);
    setStreamStatus('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-top">
          <h2>Chat with Graph</h2>
          <span className="chat-header-sub">Order to Cash</span>
        </div>
      </div>

      <div className="chat-messages" ref={messagesRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.isError ? 'error' : msg.role}`}>
            {msg.role === 'assistant' && !msg.isError && (
              <div className="chat-branding">
                <div className="branding-icon">D</div>
                <div>
                  <div className="branding-name">Dodge AI</div>
                  <div className="branding-role">Graph Agent</div>
                </div>
              </div>
            )}
            {msg.role === 'user' && (
              <div className="chat-user-label">You</div>
            )}
            {msg.structured ? (
              <StructuredMessage structured={msg.structured} />
            ) : (
              <div dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }} />
            )}
            {msg.meta && (
              <div className="message-meta">
                {msg.meta.intent && (
                  <span className="meta-tag intent">{msg.meta.intent}</span>
                )}
                {msg.meta.intent_source && (
                  <span className={`meta-tag ${msg.meta.intent_source === 'rules' ? 'rule-tag' : ''}`}>
                    {msg.meta.intent_source === 'rules' ? 'rule-match' : 'llm'}
                  </span>
                )}
                {msg.meta.total_ms != null && (
                  <span className="meta-tag latency">{msg.meta.total_ms.toFixed(0)}ms</span>
                )}
                {msg.meta.query_ms != null && (
                  <span className="meta-tag">query: {msg.meta.query_ms.toFixed(1)}ms</span>
                )}
                {msg.meta.nodes_traversed?.length > 0 && (
                  <span className="meta-tag">{msg.meta.nodes_traversed.length} nodes</span>
                )}
              </div>
            )}
          </div>
        ))}

        {streaming && (
          <div className="streaming-status">
            <div className="streaming-dot" />
            {streamStatus}
          </div>
        )}
      </div>

      <div className="chat-input-container">
        <div className="agent-status">
          <div className="status-dot-green" />
          Dodge AI is awaiting instructions
        </div>
        <div className="chat-input-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Analyze anything"
            rows={1}
            disabled={streaming}
          />
          <button className="send-btn" onClick={() => sendMessage()} disabled={streaming || !input.trim()}>
            Send
          </button>
        </div>
        <div className="quick-actions">
          {QUICK_QUERIES.map((q, i) => (
            <button key={i} onClick={() => sendMessage(q)} disabled={streaming}>
              {q.length > 35 ? q.slice(0, 35) + '…' : q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}


function StructuredMessage({ structured }) {
  const [showDetails, setShowDetails] = useState(false);
  const [showSql, setShowSql] = useState(false);

  return (
    <div className="structured-msg">
      <div dangerouslySetInnerHTML={{ __html: formatMarkdown(structured.summary || '') }} />

      {structured.insights?.length > 0 && (
        <div className="insights-container">
          {structured.insights.map((insight, i) => (
            <div key={i} className={`insight-card ${classifyInsight(insight)}`}>
              <span className="insight-icon">{insightIcon(insight)}</span>
              {insight}
            </div>
          ))}
        </div>
      )}

      {structured.generated_sql && (
        <div className="collapsible-section">
          <button className="collapsible-toggle" onClick={() => setShowSql(!showSql)}>
            {showSql ? '▾' : '▸'} Generated SQL
          </button>
          {showSql && (
            <pre className="sql-block">{structured.generated_sql}</pre>
          )}
        </div>
      )}

      {structured.details && Object.keys(structured.details).length > 0 && (
        <div className="collapsible-section">
          <button className="collapsible-toggle" onClick={() => setShowDetails(!showDetails)}>
            {showDetails ? '▾' : '▸'} Raw Details
          </button>
          {showDetails && (
            <pre className="details-block">{JSON.stringify(structured.details, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}


function classifyInsight(text) {
  const t = text.toLowerCase();
  if (t.includes('leakage') || t.includes('cancelled') || t.includes('risk'))
    return 'insight-danger';
  if (t.includes('pending') || t.includes('open') || t.includes('not found') || t.includes('no '))
    return 'insight-warning';
  return 'insight-info';
}

function insightIcon(text) {
  const t = text.toLowerCase();
  if (t.includes('leakage') || t.includes('cancelled') || t.includes('risk')) return '!';
  if (t.includes('pending') || t.includes('open') || t.includes('not found') || t.includes('no ')) return '?';
  return 'i';
}


function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}


function formatMarkdown(text) {
  if (!text) return '';

  const lines = text.split('\n');
  const sections = [];
  let buf = [];
  let tbl = [];
  let inTable = false;

  for (const line of lines) {
    const trimmed = line.trim();
    const isSep = /^\|?\s*[-:]+(\s*\|\s*[-:]+)+\s*\|?\s*$/.test(trimmed);
    const isPipe = trimmed.includes('|') && !isSep;

    if (isSep) continue;

    if (isPipe && (trimmed.startsWith('**') || trimmed.startsWith('|') || /\w\s*\|/.test(trimmed))) {
      if (!inTable) {
        if (buf.length) { sections.push({ t: 'md', v: buf.join('\n') }); buf = []; }
        inTable = true;
      }
      tbl.push(trimmed);
    } else {
      if (inTable) {
        sections.push({ t: 'tbl', v: tbl });
        tbl = [];
        inTable = false;
      }
      buf.push(line);
    }
  }
  if (inTable && tbl.length) sections.push({ t: 'tbl', v: tbl });
  if (buf.length) sections.push({ t: 'md', v: buf.join('\n') });

  return sections.map(s => s.t === 'tbl' ? renderTable(s.v) : inlineMarkdown(s.v)).join('');
}


function renderTable(rows) {
  const parsed = rows.map(r =>
    r.split('|').map(c => c.trim().replace(/\*\*/g, '')).filter(c => c !== '')
  );
  if (!parsed.length) return '';

  const hdr = parsed[0];
  const body = parsed.slice(1);

  let h = '<div class="table-wrap"><table class="chat-table"><thead><tr>';
  hdr.forEach(c => { h += `<th>${escapeHtml(c)}</th>`; });
  h += '</tr></thead><tbody>';
  body.forEach(row => {
    h += '<tr>';
    for (let i = 0; i < hdr.length; i++) {
      h += `<td>${escapeHtml(row[i] || '')}</td>`;
    }
    h += '</tr>';
  });
  h += '</tbody></table></div>';
  return h;
}


function inlineMarkdown(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^• (.+)$/gm, '<li>$1</li>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/gs, '<ul>$&</ul>')
    .replace(/\n/g, '<br/>');
}
