"use client";

import { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Sidebar, { GuardrailMeta } from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';
import { Bot, LogIn } from 'lucide-react';

import { getApiBaseUrl } from '@/utils/api';

import DocumentIngest from '@/components/DocumentIngest';
import GraphVisualizer from '@/components/GraphVisualizer';
import GuardrailManager from '@/components/GuardrailManager';

const DEFAULT_RAILS_META: GuardrailMeta[] = [
  {
    id: "prompt_injection",
    name: "Prompt Injection Defense",
    category: "Input Security",
    description: "Intercepts prompt injections, system overrides, roleplay bypasses, and system prompt exfiltration before any LLM is called.",
    trigger: "Triggers on patterns like 'Ignore previous instructions', 'DAN mode', 'System override', or 'Print instructions'.",
    action: "Instantly blocks the turn, saving tokens and preserving safety directives.",
    user_help: "Rephrase your query to ask a direct question about documents, services, or bookings without system commands."
  },
  {
    id: "pii_detection",
    name: "PII Redaction & Privacy",
    category: "Privacy & Data Protection",
    description: "Detects and redacts sensitive personal data (credit card numbers, SSNs, passwords) before sending to external model providers.",
    trigger: "Triggers when valid card numbers (13-16 digits), SSN formats (XXX-XX-XXXX), or explicit credentials are typed.",
    action: "Masks sensitive tokens with [REDACTED_...] and warns user to keep credentials private.",
    user_help: "Avoid typing real financial or government ID details in the chat window."
  },
  {
    id: "content_moderation",
    name: "Profanity & Explicit Content Filter",
    category: "Content Safety",
    description: "Detects and blocks NSFW content, adult material, profanity, and abusive or vulgar language before reaching the agent or LLM.",
    trigger: "Triggers when explicit adult terms (e.g. 'porn', 'xxx', 'nsfw') or vulgar/abusive words (e.g. 'shit', 'fuck', 'bitch', 'asshole') are detected.",
    action: "Instantly blocks the turn, prevents unnecessary LLM token spend, and guides the user toward acceptable inquiries.",
    user_help: "Refrain from vulgar, adult, or abusive terms. Rephrase using polite, professional language related to office services or document queries."
  },
  {
    id: "booking_rules",
    name: "Working Hours & Booking Rules",
    category: "Tool Action Safety",
    description: "Validates scheduling requests: strictly future dates, weekday working hours (9:00 AM - 6:00 PM), and reasonable duration.",
    trigger: "Triggers when an appointment is requested outside 9 AM - 6 PM, on weekends, or for a past date.",
    action: "Blocks invalid booking and returns current operational schedule with alternative slot recommendations.",
    user_help: "Request a time between 9:00 AM and 6:00 PM on a weekday (e.g. 'Tomorrow at 2:00 PM')."
  },
  {
    id: "anti_flooding",
    name: "Anti-Flooding Rate Limiter",
    category: "Action Rate Limiting",
    description: "Limits Google Calendar bookings to a maximum of 3 appointments per session to prevent calendar spamming and DoS flooding.",
    trigger: "Triggers when a single session attempts to book more than 3 meetings in a short period.",
    action: "Halts automated booking tool execution and requests administrative confirmation.",
    user_help: "Contact our office administrator directly if you need to coordinate bulk or group appointments."
  },
  {
    id: "rag_grounding",
    name: "RAG Grounding & Hallucination Guard",
    category: "Knowledge Retrieval",
    description: "Enforces strict vector similarity score verification on Pinecone results to prevent hallucinating ungrounded policies.",
    trigger: "Triggers when query relevance score in vector database is below confidence threshold (cosine similarity < 0.35).",
    action: "Informs the user that no verified source documents matched the query instead of fabricating facts.",
    user_help: "Verify that relevant PDFs were uploaded in the 'Ingest Documents' tab, or rephrase with specific document keywords."
  },
  {
    id: "secret_leak",
    name: "Secret & Credential Leak Filter",
    category: "Output Security",
    description: "Inspects model outputs in real time to guarantee zero leakage of internal API keys, database credentials, or OAuth tokens.",
    trigger: "Triggers if output text contains patterns matching OpenRouter, Pinecone, HuggingFace tokens or OAuth secrets.",
    action: "Instantly redacts the secret tokens before the payload reaches the browser.",
    user_help: "System security remains protected. No action required by user."
  }
];

export default function Home() {
  const [userName, setUserName] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'chat' | 'ingest' | 'graph' | 'guardrail'>('chat');
  const [showLogin, setShowLogin] = useState<boolean>(true);
  const [loginInput, setLoginInput] = useState('');

  // Guardrails State
  const [guardrails, setGuardrails] = useState<Record<string, boolean>>({
    prompt_injection: true,
    pii_detection: true,
    content_moderation: true,
    booking_rules: true,
    anti_flooding: true,
    rag_grounding: true,
    secret_leak: true
  });
  const [guardrailsMeta, setGuardrailsMeta] = useState<GuardrailMeta[]>(DEFAULT_RAILS_META);

  // Fetch Guardrail Settings on Load
  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const res = await fetch(`${getApiBaseUrl()}/guardrails/config`);
        if (res.ok) {
          const data = await res.json();
          if (data.settings) setGuardrails(data.settings);
          if (data.metadata && data.metadata.length > 0) setGuardrailsMeta(data.metadata);
        }
      } catch (err) {
        console.error("Could not load backend guardrails config", err);
      }
    };
    fetchGuardrails();
  }, []);

  const handleToggleGuardrail = async (id: string) => {
    const updated = {
      ...guardrails,
      [id]: !guardrails[id]
    };
    setGuardrails(updated);

    try {
      await fetch(`${getApiBaseUrl()}/guardrails/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: updated })
      });
    } catch (err) {
      console.error("Failed to sync guardrail toggle with backend", err);
    }
  };

  const handleSetAllGuardrails = async (enabled: boolean) => {
    const updated: Record<string, boolean> = {};
    guardrailsMeta.forEach((r) => {
      updated[r.id] = enabled;
    });
    setGuardrails(updated);

    try {
      await fetch(`${getApiBaseUrl()}/guardrails/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: updated })
      });
    } catch (err) {
      console.error("Failed to sync bulk guardrails update with backend", err);
    }
  };

  // Login handler
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginInput.trim()) return;

    try {
      const response = await fetch(`${getApiBaseUrl()}/user/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_name: loginInput })
      });

      
      if (response.ok) {
        const data = await response.json();
        setUserName(data.user_name);
        setUserId(data.user_id);
        setThreadId(uuidv4());
        setShowLogin(false);
      } else {
        alert('Login failed');
      }
    } catch (error) {
      console.error('Error logging in:', error);
      alert('Backend is unreachable. Please start the FastAPI server.');
    }
  };

  const handleLogout = () => {
    setUserName(null);
    setUserId(null);
    setThreadId('');
    setActiveTab('chat');
    setShowLogin(true);
  };

  if (showLogin) {
    return (
      <div className="app-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div className="glass-panel" style={{ padding: '40px', width: '400px', textAlign: 'center' }}>
          <Bot size={48} color="var(--aurora-blue)" style={{ marginBottom: '20px' }} />
          <h1 style={{ marginBottom: '30px' }} className="text-aurora">Welcome to AI Chat</h1>
          <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <input 
              type="text" 
              className="input-glass"
              placeholder="Enter your name" 
              value={loginInput}
              onChange={(e) => setLoginInput(e.target.value)}
              required
            />
            <button type="submit" className="btn-primary" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <LogIn size={20} />
              Start Chatting
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar 
        userName={userName!} 
        threadId={threadId} 
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onNewChat={() => setThreadId(uuidv4())} 
        onLogout={handleLogout} 
        guardrails={guardrails}
        guardrailsMeta={guardrailsMeta}
        onToggleGuardrail={handleToggleGuardrail}
      />
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', position: 'relative', overflow: 'hidden' }}>
        {activeTab === 'chat' && (
          <ChatInterface 
            userName={userName!} 
            userId={userId!} 
            threadId={threadId} 
            guardrails={guardrails}
          />
        )}
        {activeTab === 'ingest' && (
          <DocumentIngest 
            userName={userName!} 
          />
        )}
        {activeTab === 'graph' && (
          <GraphVisualizer />
        )}
        {activeTab === 'guardrail' && (
          <GuardrailManager 
            guardrails={guardrails}
            guardrailsMeta={guardrailsMeta}
            onToggleGuardrail={handleToggleGuardrail}
            onSetAllGuardrails={handleSetAllGuardrails}
          />
        )}
      </main>
    </div>
  );
}

