"use client";

import { useState, useRef, useEffect } from 'react';
import { Send, Square, Zap, ZapOff, ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';
import CalendarPicker from './CalendarPicker';
import { v4 as uuidv4 } from 'uuid';
import { getApiBaseUrl } from '@/utils/api';

export interface Message {
  role: 'user' | 'bot';
  content: string;
  agent?: string;
  is_booking?: boolean;
  guardrail_triggered?: boolean;
  guardrail_info?: {
    blocked?: boolean;
    guardrail?: string;
    guardrail_name?: string;
    reason?: string;
    suggestion?: string;
    warning_only?: boolean;
  };
}

interface ChatInterfaceProps {
  userName: string;
  userId: string;
  threadId: string;
  guardrails?: Record<string, boolean>;
}

export default function ChatInterface({ userName, userId, threadId, guardrails = {} }: ChatInterfaceProps) {
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [useStreaming, setUseStreaming] = useState(false); // Default to stable non-streaming, toggleable
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
        await fetch(`${getApiBaseUrl()}/chat/cancel`, {
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
      const response = await fetch(`${getApiBaseUrl()}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: userMsg,
          user_name: userName,
          user_id: userId,
          stream_enabled: useStreaming,
          context_summary: "",
          thread_id: threadId,
          guardrail_settings: guardrails
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error(`Server returned error: ${response.status}`);

      const contentType = response.headers.get("content-type") || "";

      // 1. NON-STREAMING / JSON RESPONSE PATH
      if (!useStreaming || contentType.includes("application/json")) {
        const data = await response.json();
        const botResponse = data.bot_response || "No response received from the assistant.";
        const intent = data.intent || "SupportAgent";
        const isGuardrail = data.guardrail_triggered || 
          botResponse.includes('🛡️ **Guardrail') || 
          botResponse.includes('🛡️ **Privacy') || 
          intent === 'guardrail_blocked';

        const isBooking = intent === 'BookingAgent' || 
          botResponse.toLowerCase().includes('appointment') || 
          botResponse.toLowerCase().includes('booked') ||
          botResponse.toLowerCase().includes('meeting') ||
          botResponse.toLowerCase().includes('open calendar') || 
          botResponse.includes('BOOKING_REQUEST');

        setIsStreaming(false);
        setHistory(prev => [...prev, {
          role: 'bot',
          content: botResponse,
          agent: isGuardrail ? 'GUARDRAIL ENGINE' : (intent === 'BookingAgent' ? 'BOOKING AGENT' : 'SUPPORT AGENT'),
          is_booking: isBooking,
          guardrail_triggered: isGuardrail,
          guardrail_info: data.guardrail_info
        }]);
        return;
      }

      // 2. STREAMING (SSE) PATH
      if (!response.body) throw new Error("No response body received");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let fullResponse = '';
      let intent = 'SupportAgent';
      let guardrailInfo: any = null;
      let isGuardrail = false;

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
            } else if (data.type === 'content') {
              fullResponse += data.content;
              setStreamingMessage(fullResponse);
            } else if (data.type === 'intent') {
              intent = data.content;
              if (data.content === 'guardrail_blocked') isGuardrail = true;
            } else if (data.type === 'guardrail') {
              guardrailInfo = data.data;
              isGuardrail = true;
            } else if (data.type === 'cancelled' || data.type === 'done') {
              break;
            }
          } catch (e) {
            // raw text token fallback
            if (dataStr && !dataStr.startsWith('{')) {
              fullResponse += dataStr;
              setStreamingMessage(fullResponse);
            }
          }
        }
      }

      setIsStreaming(false);
      setStreamingMessage('');

      // Fallback if streaming ended without tokens
      if (!fullResponse.trim()) {
        try {
          const fallbackRes = await fetch(`${getApiBaseUrl()}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_message: userMsg,
              user_name: userName,
              user_id: userId,
              stream_enabled: false,
              context_summary: "",
              thread_id: threadId,
              guardrail_settings: guardrails
            })
          });
          const fbData = await fallbackRes.json();
          fullResponse = fbData.bot_response || "No response received.";
          if (fbData.guardrail_triggered) {
            isGuardrail = true;
            guardrailInfo = fbData.guardrail_info;
          }
        } catch {
          fullResponse = "The free model did not stream tokens. Try toggling 'Stream OFF' above for direct responses.";
        }
      }

      if (fullResponse.includes('🛡️ **Guardrail') || fullResponse.includes('🛡️ **Privacy')) {
        isGuardrail = true;
      }

      const isBooking = intent === 'BookingAgent' || fullResponse.includes('BOOKING_REQUEST');
      setHistory(prev => [...prev, {
        role: 'bot',
        content: fullResponse,
        agent: isGuardrail ? 'GUARDRAIL ENGINE' : (intent === 'BookingAgent' ? 'BOOKING AGENT' : 'SUPPORT AGENT'),
        is_booking: isBooking,
        guardrail_triggered: isGuardrail,
        guardrail_info: guardrailInfo
      }]);
      
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Fetch aborted');
      } else {
        console.error('Fetch error:', err);
        setIsStreaming(false);
        setStreamingMessage('');
        setHistory(prev => [...prev, {
          role: 'bot',
          content: `Unable to get a response: ${err.message || 'Server connection error'}. Please try again or toggle Stream mode.`
        }]);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', padding: '20px' }}>
      {/* Header */}
      <header style={{ paddingBottom: '20px', borderBottom: '1px solid var(--glass-border)', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 className="text-aurora" style={{ margin: 0 }}>Chat Session</h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Knowledge Retrieval (RAG) + Multi-Agent Assistant
          </span>
        </div>

        {/* Streaming Mode Toggle */}
        <button
          onClick={() => setUseStreaming(!useStreaming)}
          className="btn-secondary"
          title="Toggle between instant direct response and token-by-token streaming"
          style={{
            fontSize: '0.8rem',
            padding: '6px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            borderColor: useStreaming ? 'var(--aurora-green)' : 'var(--glass-border)',
            color: useStreaming ? 'var(--aurora-green)' : 'var(--text-secondary)',
            background: useStreaming ? 'rgba(74, 222, 128, 0.1)' : 'transparent',
            borderRadius: '20px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          {useStreaming ? <Zap size={14} /> : <ZapOff size={14} />}
          <span>Stream: {useStreaming ? 'ON' : 'OFF'}</span>
        </button>
      </header>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px', paddingRight: '10px' }}>
        {history.length === 0 && !isStreaming && (
          <div style={{ margin: 'auto', color: 'var(--text-secondary)', textAlign: 'center' }}>
            <p style={{ fontSize: '1.1rem', marginBottom: '8px' }}>👋 Welcome, {userName}!</p>
            <p style={{ fontSize: '0.9rem' }}>Ask questions about your uploaded documents, policies, or request assistance.</p>
          </div>
        )}
        
        {history.map((msg, idx) => (
          <div key={idx} style={{ 
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '85%'
          }}>
            <div className={`glass-panel animate-fade-in`} style={{
              padding: '16px 20px', 
              background: msg.role === 'user' 
                ? 'rgba(96, 239, 255, 0.08)' 
                : (msg.guardrail_triggered ? 'rgba(255, 180, 0, 0.06)' : 'var(--glass-bg)'),
              border: msg.role === 'user' 
                ? '1px solid rgba(96, 239, 255, 0.25)' 
                : (msg.guardrail_triggered ? '1px solid rgba(255, 180, 0, 0.45)' : '1px solid var(--glass-border)'),
              borderRadius: '12px',
              boxShadow: msg.guardrail_triggered ? '0 0 20px rgba(255, 180, 0, 0.15)' : 'none'
            }}>
              {msg.role === 'bot' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                  {msg.guardrail_triggered ? (
                    <div style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '0.74rem',
                      color: '#ffb400',
                      background: 'rgba(255, 180, 0, 0.15)',
                      border: '1px solid rgba(255, 180, 0, 0.35)',
                      borderRadius: '14px',
                      padding: '3px 10px',
                      fontWeight: 700
                    }}>
                      <ShieldAlert size={14} /> 
                      {msg.guardrail_info?.guardrail_name || 'GUARDRAIL ALERT'}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.75rem', color: 'var(--aurora-blue)', fontWeight: 600, letterSpacing: '0.5px' }}>
                      🤖 {msg.agent || 'SUPPORT AGENT'}
                    </div>
                  )}
                </div>
              )}
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '0.95rem' }}>
                {msg.content}
              </div>

              {/* Actionable Correction Box for Guardrail Alerts */}
              {msg.guardrail_info?.suggestion && (
                <div style={{
                  marginTop: '12px',
                  padding: '10px 14px',
                  background: 'rgba(255, 180, 0, 0.08)',
                  border: '1px solid rgba(255, 180, 0, 0.25)',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  fontSize: '0.82rem',
                  color: '#fef3c7'
                }}>
                  <AlertTriangle size={16} color="#ffb400" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <strong style={{ color: '#ffb400', display: 'block', marginBottom: '2px' }}>
                      How to correct this:
                    </strong>
                    {msg.guardrail_info.suggestion}
                  </div>
                </div>
              )}
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
          <div style={{ alignSelf: 'flex-start', maxWidth: '80%' }}>
            <div className="glass-panel animate-fade-in" style={{ padding: '16px 20px', borderRadius: '12px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--aurora-blue)', marginBottom: '8px', fontWeight: 600 }}>
                🤖 SUPPORT AGENT
              </div>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '0.95rem' }}>
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
          placeholder="Type your question or query..."
          className="input-glass"
          style={{ flex: 1, padding: '12px 16px' }}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button onClick={cancelStream} className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ff4b4b', borderColor: 'rgba(255, 75, 75, 0.3)' }}>
            <Square size={18} fill="currentColor" /> Stop
          </button>
        ) : (
          <button onClick={handleSend} className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Send size={18} /> Send
          </button>
        )}
      </div>
    </div>
  );
}
