"use client";

import React, { useEffect, useState, useRef } from 'react';
import { 
  Network, 
  ExternalLink, 
  CheckCircle2, 
  AlertCircle, 
  Copy, 
  Check, 
  Cpu, 
  Calendar, 
  FileText, 
  ArrowRight, 
  Compass, 
  ShieldCheck, 
  RefreshCw,
  Terminal,
  Layers,
  Sparkles
} from 'lucide-react';
import { getApiBaseUrl } from '@/utils/api';

interface GraphInfo {
  mermaid: string;
  studio_url: string;
  local_dev_url: string;
  langsmith_tracing: boolean;
  project_name: string;
}

export default function GraphVisualizer() {
  const [graphInfo, setGraphInfo] = useState<GraphInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [studioOnline, setStudioOnline] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeNode, setActiveNode] = useState<'supervisor' | 'booking' | 'support' | 'tools'>('supervisor');
  const [viewMode, setViewMode] = useState<'visual' | 'mermaid'>('visual');
  const mermaidContainerRef = useRef<HTMLDivElement>(null);

  const fetchGraphData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${getApiBaseUrl()}/graph/info`);
      if (res.ok) {
        const data = await res.json();
        setGraphInfo(data);
      }
    } catch (err) {
      console.error("Failed to load graph info:", err);
    }

    // Check if LangGraph Dev server (2024) is reachable
    try {
      const studioRes = await fetch('http://127.0.0.1:2024/ok', { mode: 'no-cors' });
      setStudioOnline(true);
    } catch (e) {
      // In case no-cors still flags or server is down
      setStudioOnline(false);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchGraphData();
  }, []);

  // Dynamically load Mermaid CDN script when viewing Mermaid tab
  useEffect(() => {
    if (viewMode === 'mermaid' && graphInfo?.mermaid) {
      const renderMermaid = async () => {
        try {
          // Check if mermaid is already loaded on window
          let mermaid = (window as any).mermaid;
          if (!mermaid) {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
            script.async = true;
            document.body.appendChild(script);
            await new Promise((resolve) => {
              script.onload = resolve;
            });
            mermaid = (window as any).mermaid;
          }

          if (mermaid && mermaidContainerRef.current) {
            mermaid.initialize({
              startOnLoad: false,
              theme: 'dark',
              themeVariables: {
                darkMode: true,
                primaryColor: '#60efff',
                primaryTextColor: '#ffffff',
                primaryBorderColor: '#0061ff',
                lineColor: '#00ff87',
                secondaryColor: '#1a1f35',
                tertiaryColor: '#0d1117'
              }
            });

            // Clean clean syntax from backend
            let cleanSyntax = graphInfo.mermaid;
            // Remove frontmatter config if present for standalone render
            if (cleanSyntax.startsWith('---')) {
              const parts = cleanSyntax.split('---');
              if (parts.length >= 3) {
                cleanSyntax = parts.slice(2).join('---').trim();
              }
            }

            const { svg } = await mermaid.render('mermaid-chart-svg-' + Date.now(), cleanSyntax);
            if (mermaidContainerRef.current) {
              mermaidContainerRef.current.innerHTML = svg;
            }
          }
        } catch (e) {
          console.error("Mermaid rendering error:", e);
        }
      };

      renderMermaid();
    }
  }, [viewMode, graphInfo]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      width: '100%',
      overflowY: 'auto',
      padding: '30px',
      gap: '24px'
    }}>
      {/* Top Header Banner */}
      <div className="glass-panel" style={{
        padding: '24px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '20px',
        background: 'linear-gradient(135deg, rgba(96, 239, 255, 0.08) 0%, rgba(0, 255, 135, 0.04) 100%)',
        border: '1px solid rgba(96, 239, 255, 0.25)'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span style={{
              background: 'rgba(96, 239, 255, 0.15)',
              padding: '6px 12px',
              borderRadius: '20px',
              color: 'var(--aurora-blue)',
              fontSize: '0.75rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '1px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <Sparkles size={13} /> LangGraph Multi-Agent Architecture
            </span>

            <span style={{
              background: studioOnline ? 'rgba(0, 255, 135, 0.15)' : 'rgba(255, 180, 0, 0.15)',
              padding: '6px 12px',
              borderRadius: '20px',
              color: studioOnline ? 'var(--aurora-green)' : '#ffb400',
              fontSize: '0.75rem',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: studioOnline ? 'var(--aurora-green)' : '#ffb400',
                display: 'inline-block',
                boxShadow: studioOnline ? '0 0 8px var(--aurora-green)' : 'none'
              }} />
              {studioOnline ? 'Studio Dev Server Online (:2024)' : 'Studio Dev Server Ready'}
            </span>
          </div>

          <h1 className="text-aurora" style={{ fontSize: '1.8rem', fontWeight: 800, margin: 0 }}>
            LangSmith Studio & Graph Visualizer
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '6px', maxWidth: '640px' }}>
            Interactive state graph visualization with hierarchical supervisor routing, sub-agent execution pipelines, and real-time LangSmith Cloud tracing.
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <button
            onClick={fetchGraphData}
            className="btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 16px', fontSize: '0.9rem' }}
          >
            <RefreshCw size={16} /> Refresh
          </button>

          <a
            href={graphInfo?.studio_url || "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 20px',
              fontSize: '0.92rem',
              fontWeight: 700,
              textDecoration: 'none',
              boxShadow: '0 0 20px rgba(96, 239, 255, 0.3)'
            }}
          >
            <Compass size={18} /> Launch LangSmith Studio <ExternalLink size={15} />
          </a>
        </div>
      </div>

      {/* Observability & Status Metric Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '16px'
      }}>
        <div className="glass-panel" style={{ padding: '18px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '8px' }}>
            <Layers size={16} color="var(--aurora-blue)" /> GRAPH ENGINE
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
            LangGraph 1.2+
          </div>
          <div style={{ color: 'var(--aurora-green)', fontSize: '0.8rem', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle2 size={13} /> Hierarchical Supervisor StateGraph
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '18px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '8px' }}>
            <ShieldCheck size={16} color="var(--aurora-green)" /> LANGSMITH TRACING
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
            {graphInfo?.langsmith_tracing ? 'Enabled (V2)' : 'Active (Local)'}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px' }}>
            Project: <code style={{ color: 'var(--aurora-blue)' }}>{graphInfo?.project_name || 'AI-Chat-Assistant'}</code>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '18px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '8px' }}>
            <Terminal size={16} color="#c084fc" /> STUDIO DEV SERVER
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
            Port 2024
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px' }}>
            Base URL: <code style={{ color: 'var(--aurora-blue)' }}>http://127.0.0.1:2024</code>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '18px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '8px' }}>
            <Cpu size={16} color="#fb7185" /> RESILIENT LLM ROUTER
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
            Multi-Model Fallback
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px' }}>
            OpenRouter Free Pool + Safety Filter
          </div>
        </div>
      </div>

      {/* Main Graph Flow Visualization & Inspector Split */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.7fr) minmax(320px, 1fr)',
        gap: '24px',
        alignItems: 'start'
      }}>
        {/* Left Column: Visual Flow Architecture Map */}
        <div className="glass-panel" style={{ padding: '24px', minHeight: '520px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setViewMode('visual')}
                style={{
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: viewMode === 'visual' ? '1px solid var(--aurora-blue)' : '1px solid var(--glass-border)',
                  background: viewMode === 'visual' ? 'rgba(96, 239, 255, 0.12)' : 'transparent',
                  color: viewMode === 'visual' ? 'var(--aurora-blue)' : 'var(--text-secondary)',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  cursor: 'pointer'
                }}
              >
                Interactive Flow Diagram
              </button>
              <button
                onClick={() => setViewMode('mermaid')}
                style={{
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: viewMode === 'mermaid' ? '1px solid var(--aurora-green)' : '1px solid var(--glass-border)',
                  background: viewMode === 'mermaid' ? 'rgba(0, 255, 135, 0.12)' : 'transparent',
                  color: viewMode === 'mermaid' ? 'var(--aurora-green)' : 'var(--text-secondary)',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  cursor: 'pointer'
                }}
              >
                Mermaid Render
              </button>
            </div>

            {viewMode === 'mermaid' && graphInfo?.mermaid && (
              <button
                onClick={() => copyToClipboard(graphInfo.mermaid)}
                className="btn-secondary"
                style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', fontSize: '0.8rem' }}
              >
                {copied ? <Check size={14} color="var(--aurora-green)" /> : <Copy size={14} />}
                {copied ? 'Copied' : 'Copy Mermaid'}
              </button>
            )}
          </div>

          {/* Tab 1: Interactive Node Flow Diagram */}
          {viewMode === 'visual' && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '24px',
              padding: '10px 0',
              flex: 1,
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {/* START Node */}
              <div style={{
                background: 'linear-gradient(90deg, #0061ff, #60efff)',
                padding: '8px 24px',
                borderRadius: '30px',
                color: '#000',
                fontWeight: 800,
                fontSize: '0.85rem',
                letterSpacing: '1px',
                boxShadow: '0 0 16px rgba(96, 239, 255, 0.4)'
              }}>
                ● USER MESSAGE IN (START)
              </div>

              {/* Down Arrow */}
              <div style={{ width: '2px', height: '24px', background: 'var(--aurora-blue)', position: 'relative' }}>
                <span style={{ position: 'absolute', bottom: '-4px', left: '-4px', width: '10px', height: '10px', borderBottom: '2px solid var(--aurora-blue)', borderRight: '2px solid var(--aurora-blue)', transform: 'rotate(45deg)' }} />
              </div>

              {/* SUPERVISOR Node */}
              <div 
                onClick={() => setActiveNode('supervisor')}
                style={{
                  width: '90%',
                  maxWidth: '420px',
                  background: activeNode === 'supervisor' ? 'rgba(96, 239, 255, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                  border: activeNode === 'supervisor' ? '2px solid var(--aurora-blue)' : '1px solid var(--glass-border)',
                  borderRadius: '16px',
                  padding: '16px 20px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: activeNode === 'supervisor' ? '0 0 25px rgba(96, 239, 255, 0.2)' : 'none'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: 'var(--aurora-blue)' }}>
                    <Cpu size={18} /> Supervisor Router
                  </div>
                  <span style={{ fontSize: '0.72rem', background: 'rgba(96, 239, 255, 0.2)', padding: '2px 8px', borderRadius: '12px', color: '#fff' }}>
                    Orchestrator
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Evaluates user intent. Deterministically routes booking queries vs knowledge/support queries.
                </p>
              </div>

              {/* Branching Connectors */}
              <div style={{ width: '80%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative' }}>
                <div style={{ position: 'absolute', top: '-10px', left: '20%', right: '20%', height: '2px', background: 'var(--glass-border)' }} />
                
                {/* Branch to Booking */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '48%' }}>
                  <span style={{ fontSize: '0.72rem', color: '#c084fc', marginBottom: '6px', fontWeight: 600 }}>
                    Booking Intent
                  </span>
                  <div style={{ width: '2px', height: '20px', background: '#c084fc' }} />
                  
                  {/* BOOKING AGENT */}
                  <div
                    onClick={() => setActiveNode('booking')}
                    style={{
                      width: '100%',
                      background: activeNode === 'booking' ? 'rgba(192, 132, 252, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      border: activeNode === 'booking' ? '2px solid #c084fc' : '1px solid var(--glass-border)',
                      borderRadius: '14px',
                      padding: '14px 16px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: activeNode === 'booking' ? '0 0 20px rgba(192, 132, 252, 0.25)' : 'none'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: '#c084fc', marginBottom: '4px' }}>
                      <Calendar size={16} /> BookingAgent
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      Slot check & live calendar sync
                    </div>
                  </div>

                  {/* Down Arrow to Booking Tool */}
                  <div style={{ width: '2px', height: '14px', background: 'var(--glass-border)' }} />
                  <div
                    onClick={() => setActiveNode('tools')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px dashed #c084fc',
                      borderRadius: '8px',
                      padding: '6px 10px',
                      fontSize: '0.75rem',
                      color: '#ddd',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer'
                    }}
                  >
                    ⚙️ booking_agent_tool
                  </div>
                </div>

                {/* Branch to Support */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '48%' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--aurora-green)', marginBottom: '6px', fontWeight: 600 }}>
                    Support / Doc Intent
                  </span>
                  <div style={{ width: '2px', height: '20px', background: 'var(--aurora-green)' }} />

                  {/* SUPPORT AGENT */}
                  <div
                    onClick={() => setActiveNode('support')}
                    style={{
                      width: '100%',
                      background: activeNode === 'support' ? 'rgba(0, 255, 135, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      border: activeNode === 'support' ? '2px solid var(--aurora-green)' : '1px solid var(--glass-border)',
                      borderRadius: '14px',
                      padding: '14px 16px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      boxShadow: activeNode === 'support' ? '0 0 20px rgba(0, 255, 135, 0.25)' : 'none'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: 'var(--aurora-green)', marginBottom: '4px' }}>
                      <FileText size={16} /> SupportAgent
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      RAG Document Similarity Search
                    </div>
                  </div>

                  {/* Down Arrow to Support Tool */}
                  <div style={{ width: '2px', height: '14px', background: 'var(--glass-border)' }} />
                  <div
                    onClick={() => setActiveNode('tools')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px dashed var(--aurora-green)',
                      borderRadius: '8px',
                      padding: '6px 10px',
                      fontSize: '0.75rem',
                      color: '#ddd',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer'
                    }}
                  >
                    🔍 similarity_search_tool
                  </div>
                </div>
              </div>

              {/* Loop back & Finish */}
              <div style={{ width: '2px', height: '20px', background: 'var(--glass-border)' }} />
              <div style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid var(--glass-border)',
                padding: '6px 20px',
                borderRadius: '20px',
                color: '#fff',
                fontSize: '0.8rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <CheckCircle2 size={15} color="var(--aurora-green)" /> Turn Finalized (END)
              </div>
            </div>
          )}

          {/* Tab 2: Dynamic Mermaid Diagram */}
          {viewMode === 'mermaid' && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'auto',
              minHeight: '380px'
            }}>
              <div 
                ref={mermaidContainerRef} 
                style={{ width: '100%', display: 'flex', justifyContent: 'center' }}
              >
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  Loading Mermaid diagram...
                </div>
              </div>

              {graphInfo?.mermaid && (
                <div style={{ marginTop: '20px', width: '100%' }}>
                  <details style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '10px', padding: '12px', fontSize: '0.8rem' }}>
                    <summary style={{ cursor: 'pointer', color: 'var(--aurora-blue)', fontWeight: 600 }}>
                      View Mermaid Raw Definition
                    </summary>
                    <pre style={{ margin: '10px 0 0 0', overflowX: 'auto', color: '#94a3b8' }}>
                      {graphInfo.mermaid}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Node Details Inspector & Studio Guide */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Node Inspector Card */}
          <div className="glass-panel" style={{ padding: '22px' }}>
            <h2 style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '14px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Node Inspector
            </h2>

            {activeNode === 'supervisor' && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <div style={{ padding: '8px', borderRadius: '10px', background: 'rgba(96, 239, 255, 0.15)', color: 'var(--aurora-blue)' }}>
                    <Cpu size={22} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.15rem' }}>supervisor</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--aurora-blue)' }}>Entrypoint Node</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', marginBottom: '16px' }}>
                  The Supervisor acts as the central router for incoming user turns. It inspects query tokens and context summary, routing to <strong>BookingAgent</strong> for scheduling requests or <strong>SupportAgent</strong> for documents and questions.
                </div>

                <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '12px', fontSize: '0.8rem', marginBottom: '12px' }}>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>Transitions:</div>
                  <div style={{ color: '#c084fc' }}>● BookingAgent (if scheduling keywords matched)</div>
                  <div style={{ color: 'var(--aurora-green)', marginTop: '2px' }}>● SupportAgent (knowledge / FAQ queries)</div>
                  <div style={{ color: '#94a3b8', marginTop: '2px' }}>● FINISH (when response generated)</div>
                </div>
              </div>
            )}

            {activeNode === 'booking' && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <div style={{ padding: '8px', borderRadius: '10px', background: 'rgba(192, 132, 252, 0.15)', color: '#c084fc' }}>
                    <Calendar size={22} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.15rem' }}>BookingAgent</h3>
                    <span style={{ fontSize: '0.75rem', color: '#c084fc' }}>Sub-Agent StateGraph</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', marginBottom: '16px' }}>
                  Manages calendar appointments and meeting booking flows. Bound to Google Calendar API tools with automatic conflict checking.
                </div>

                <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '12px', fontSize: '0.8rem', marginBottom: '12px' }}>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>Bound Tool:</div>
                  <div style={{ color: '#fff', fontWeight: 600 }}>booking_agent_tool</div>
                  <div style={{ color: 'var(--aurora-green)', fontSize: '0.75rem', marginTop: '2px' }}>
                    Dual sync & async StructuredTool with live OAuth sync
                  </div>
                </div>
              </div>
            )}

            {activeNode === 'support' && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <div style={{ padding: '8px', borderRadius: '10px', background: 'rgba(0, 255, 135, 0.15)', color: 'var(--aurora-green)' }}>
                    <FileText size={22} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.15rem' }}>SupportAgent</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--aurora-green)' }}>RAG Sub-Agent</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', marginBottom: '16px' }}>
                  Executes dense vector similarity search across ingested company and user documents. Injects top-k context directly into LLM prompts.
                </div>

                <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '12px', fontSize: '0.8rem', marginBottom: '12px' }}>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>Vector Pipeline:</div>
                  <div style={{ color: '#fff', fontWeight: 600 }}>BAAI/bge-large-en-v1.5 + Pinecone</div>
                  <div style={{ color: 'var(--aurora-green)', fontSize: '0.75rem', marginTop: '2px' }}>
                    Tool: similarity_search_tool (top_k = 4)
                  </div>
                </div>
              </div>
            )}

            {activeNode === 'tools' && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <div style={{ padding: '8px', borderRadius: '10px', background: 'rgba(255, 255, 255, 0.1)', color: '#fff' }}>
                    <Terminal size={22} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Agent Tools Node</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--aurora-blue)' }}>Prebuilt ToolNode</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', marginBottom: '16px' }}>
                  Sub-agents invoke external APIs safely via LangGraph's isolated <code>ToolNode</code>. Tool results are appended to state as <code>ToolMessage</code> items.
                </div>

                <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '10px', padding: '12px', fontSize: '0.8rem' }}>
                  <div style={{ color: 'var(--aurora-blue)' }}>1. booking_agent_tool</div>
                  <div style={{ color: 'var(--aurora-green)', marginTop: '4px' }}>2. similarity_search_tool</div>
                </div>
              </div>
            )}
          </div>

          {/* LangSmith Studio Quick Launch Card */}
          <div className="glass-panel" style={{
            padding: '22px',
            background: 'linear-gradient(180deg, rgba(20, 24, 40, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%)',
            border: '1px solid rgba(96, 239, 255, 0.2)'
          }}>
            <h2 style={{ fontSize: '1rem', color: '#fff', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Compass size={18} color="var(--aurora-blue)" /> LangSmith Studio Instructions
            </h2>

            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '16px' }}>
              LangSmith Studio runs on top of your local graph via <code>langgraph dev</code> on port <strong>2024</strong>. You can inspect live state transitions, modify inputs interactively, and view full execution graphs.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <a
                href="https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '10px 16px',
                  fontSize: '0.85rem',
                  textDecoration: 'none'
                }}
              >
                Open Studio in Browser <ExternalLink size={15} />
              </a>

              <a
                href="https://smith.langchain.com"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '10px 16px',
                  fontSize: '0.85rem',
                  textDecoration: 'none'
                }}
              >
                LangSmith Cloud Dashboard <ExternalLink size={15} />
              </a>
            </div>

            <div style={{ marginTop: '16px', borderTop: '1px solid var(--glass-border)', paddingTop: '12px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              To restart LangGraph dev server from terminal:
              <pre style={{ background: 'rgba(0,0,0,0.5)', padding: '6px 10px', borderRadius: '6px', color: 'var(--aurora-blue)', marginTop: '4px', overflowX: 'auto' }}>
                langgraph dev --port 2024
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
