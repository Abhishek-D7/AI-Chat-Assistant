import { useState, useRef, useEffect } from 'react';
import { Send, Square } from 'lucide-react';
import { Message } from '../App';
import CalendarPicker from './CalendarPicker';
import { v4 as uuidv4 } from 'uuid';

interface ChatInterfaceProps {
  userName: string;
  userId: string;
  threadId: string;
}

export default function ChatInterface({ userName, userId, threadId }: ChatInterfaceProps) {
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  
  const [showCalendarForIdx, setShowCalendarForIdx] = useState<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, streamingMessage]);

  const cancelStream = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    if (currentSessionId) {
      try {
        await fetch('http://localhost:8000/chat/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: currentSessionId })
        });
      } catch (err) {
        console.error("Failed to send cancel signal to server", err);
      }
    }
    
    setIsStreaming(false);
    if (streamingMessage) {
      setHistory(prev => [...prev, { role: 'bot', content: streamingMessage + "\n\n*(Stream cancelled)*" }]);
      setStreamingMessage('');
    }
    setCurrentSessionId(null);
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    
    const userMsg = input.trim();
    setInput('');
    setHistory(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsStreaming(true);
    setStreamingMessage('');
    
    const sessionId = uuidv4();
    setCurrentSessionId(sessionId);
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: userMsg,
          user_name: userName,
          user_id: userId,
          stream_enabled: true,
          context_summary: "User chat from React UI",
          thread_id: threadId
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error("Network response was not ok");
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let fullResponse = '';
      let intent = 'general';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim().startsWith('data: '));
        
        for (const line of lines) {
          const dataStr = line.replace('data: ', '').trim();
          if (dataStr === '[DONE]') break;
          
          try {
            const data = JSON.parse(dataStr);
            if (data.type === 'token') {
              fullResponse += data.content;
              setStreamingMessage(fullResponse);
            } else if (data.type === 'intent') {
              intent = data.content;
            } else if (data.type === 'cancelled' || data.type === 'done') {
              break;
            }
          } catch (e) {
            // sometimes it sends raw text directly depending on the backend implementation
            if (dataStr && !dataStr.startsWith('{')) {
              fullResponse += dataStr;
              setStreamingMessage(fullResponse);
            }
          }
        }
      }

      setIsStreaming(false);
      setStreamingMessage('');
      const isBooking = intent === 'booking' || fullResponse.includes('BOOKING_REQUEST');
      setHistory(prev => [...prev, { role: 'bot', content: fullResponse, agent: intent, is_booking: isBooking }]);
      
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Fetch aborted');
      } else {
        console.error('Fetch error:', err);
        setIsStreaming(false);
        setStreamingMessage('');
        setHistory(prev => [...prev, { role: 'bot', content: "Sorry, an error occurred." }]);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', padding: '20px' }}>
      {/* Header */}
      <header style={{ paddingBottom: '20px', borderBottom: '1px solid var(--glass-border)', marginBottom: '20px', display: 'flex', justifyContent: 'space-between' }}>
        <h2 className="text-aurora">Chat Session</h2>
      </header>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px', paddingRight: '10px' }}>
        {history.length === 0 && !isStreaming && (
          <div style={{ margin: 'auto', color: 'var(--text-secondary)', textAlign: 'center' }}>
            <p>No messages yet.</p>
            <p>Start chatting below!</p>
          </div>
        )}
        
        {history.map((msg, idx) => (
          <div key={idx} style={{ 
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '75%'
          }}>
            <div className={`glass-panel animate-fade-in`} style={{
              padding: '15px', 
              background: msg.role === 'user' ? 'rgba(96, 239, 255, 0.05)' : 'var(--glass-bg)',
              border: msg.role === 'user' ? '1px solid rgba(96, 239, 255, 0.2)' : '1px solid var(--glass-border)',
            }}>
              {msg.role === 'bot' && (
                <div style={{ fontSize: '0.8rem', color: 'var(--aurora-blue)', marginBottom: '8px', textTransform: 'uppercase' }}>
                  🤖 {msg.agent || 'ASSISTANT'}
                </div>
              )}
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                {msg.content}
              </div>
            </div>

            {msg.is_booking && showCalendarForIdx !== idx && (
              <button 
                onClick={() => setShowCalendarForIdx(idx)} 
                className="btn-secondary animate-fade-in" 
                style={{ marginTop: '10px', fontSize: '0.8rem', padding: '6px 12px' }}
              >
                📅 Open Calendar
              </button>
            )}

            {showCalendarForIdx === idx && (
              <div className="animate-fade-in">
                <CalendarPicker 
                  onConfirm={(d, t) => {
                    alert(`Meeting booked for ${d} at ${t}`);
                    setShowCalendarForIdx(null);
                  }}
                  onCancel={() => setShowCalendarForIdx(null)}
                />
              </div>
            )}
          </div>
        ))}

        {isStreaming && (
          <div style={{ alignSelf: 'flex-start', maxWidth: '75%' }}>
            <div className="glass-panel animate-fade-in" style={{ padding: '15px' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--aurora-blue)', marginBottom: '8px' }}>🤖 ASSISTANT</div>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                {streamingMessage}
                <span style={{ display: 'inline-block', width: '8px', height: '15px', background: 'var(--aurora-green)', marginLeft: '4px', animation: 'blink 1s infinite' }}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSend() }}
          placeholder="Type your message..."
          className="input-glass"
          style={{ flex: 1 }}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button onClick={cancelStream} className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ff4b4b', borderColor: 'rgba(255, 75, 75, 0.3)' }}>
            <Square size={20} fill="currentColor" /> Stop
          </button>
        ) : (
          <button onClick={handleSend} className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Send size={20} /> Send
          </button>
        )}
      </div>
    </div>
  );
}
