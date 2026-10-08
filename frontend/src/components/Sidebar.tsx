"use client";

import { useEffect, useState } from 'react';
import { 
  LogOut, 
  PlusCircle, 
  Activity, 
  MessageSquare, 
  UploadCloud, 
  Network, 
  Shield
} from 'lucide-react';
import { getApiBaseUrl } from '@/utils/api';

export interface GuardrailMeta {
  id: string;
  name: string;
  category: string;
  description: string;
  trigger: string;
  action: string;
  user_help: string;
}

interface SidebarProps {
  userName: string;
  threadId: string;
  activeTab: 'chat' | 'ingest' | 'graph' | 'guardrail';
  onSelectTab: (tab: 'chat' | 'ingest' | 'graph' | 'guardrail') => void;
  onNewChat: () => void;
  onLogout: () => void;
  guardrails: Record<string, boolean>;
  guardrailsMeta: GuardrailMeta[];
  onToggleGuardrail?: (id: string) => void;
}

interface Stats {
  total_turns: number;
  intents: Record<string, number>;
  last_active: string;
}

export default function Sidebar({ 
  userName, 
  threadId, 
  activeTab, 
  onSelectTab, 
  onNewChat, 
  onLogout,
  guardrails,
  guardrailsMeta
}: SidebarProps) {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/user/${userName}/stats`);
        if (response.ok) {
          setStats(await response.json());
        }
      } catch (err) {
        console.error("Could not fetch stats", err);
      }
    };
    fetchStats();
  }, [userName]);

  const activeCount = Object.values(guardrails).filter(Boolean).length;
  const totalCount = guardrailsMeta.length || Object.keys(guardrails).length;

  return (
    <aside style={{
      width: '310px',
      background: 'rgba(0,0,0,0.6)',
      borderRight: '1px solid var(--glass-border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '20px 16px',
      height: '100vh',
      overflow: 'hidden'
    }}>
      {/* Scrollable Main Area */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', paddingRight: '4px' }}>
        
        {/* User Header */}
        <div style={{ marginBottom: '20px' }}>
          <h2 className="text-aurora" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem', margin: 0 }}>
            <Activity size={20} />
            {userName}
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginTop: '4px', margin: 0 }}>
            Thread: {threadId.substring(0, 8)}...
          </p>
        </div>

        {/* Page Navigation Tabs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
          <button
            onClick={() => onSelectTab('chat')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 14px',
              borderRadius: '10px',
              border: activeTab === 'chat' ? '1px solid var(--aurora-blue)' : '1px solid transparent',
              background: activeTab === 'chat' ? 'rgba(96, 239, 255, 0.12)' : 'transparent',
              color: activeTab === 'chat' ? 'var(--aurora-blue)' : 'var(--text-secondary)',
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease',
              fontSize: '0.9rem'
            }}
          >
            <MessageSquare size={18} /> Chat Session
          </button>

          <button
            onClick={() => onSelectTab('ingest')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 14px',
              borderRadius: '10px',
              border: activeTab === 'ingest' ? '1px solid var(--aurora-green)' : '1px solid transparent',
              background: activeTab === 'ingest' ? 'rgba(0, 255, 135, 0.12)' : 'transparent',
              color: activeTab === 'ingest' ? 'var(--aurora-green)' : 'var(--text-secondary)',
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease',
              fontSize: '0.9rem'
            }}
          >
            <UploadCloud size={18} /> Ingest Documents
          </button>

          <button
            onClick={() => onSelectTab('graph')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 14px',
              borderRadius: '10px',
              border: activeTab === 'graph' ? '1px solid #c084fc' : '1px solid transparent',
              background: activeTab === 'graph' ? 'rgba(192, 132, 252, 0.12)' : 'transparent',
              color: activeTab === 'graph' ? '#c084fc' : 'var(--text-secondary)',
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease',
              fontSize: '0.9rem'
            }}
          >
            <Network size={18} /> Graph Flow & Studio
          </button>

          <button
            onClick={() => onSelectTab('guardrail')}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              borderRadius: '10px',
              border: activeTab === 'guardrail' ? '1px solid #10b981' : '1px solid transparent',
              background: activeTab === 'guardrail' ? 'rgba(16, 185, 129, 0.15)' : 'transparent',
              color: activeTab === 'guardrail' ? '#34d399' : 'var(--text-secondary)',
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease',
              fontSize: '0.9rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Shield size={18} color={activeTab === 'guardrail' ? '#34d399' : undefined} /> 
              <span>Guardrails</span>
            </div>
            <span style={{
              fontSize: '0.7rem',
              fontWeight: 700,
              background: activeCount > 0 ? 'rgba(0, 255, 135, 0.15)' : 'rgba(255, 255, 255, 0.08)',
              color: activeCount > 0 ? 'var(--aurora-green)' : 'var(--text-secondary)',
              padding: '2px 8px',
              borderRadius: '12px',
              border: activeCount > 0 ? '1px solid rgba(0, 255, 135, 0.3)' : '1px solid transparent'
            }}>
              {activeCount}/{totalCount} Active
            </span>
          </button>
        </div>

        {/* New Conversation Button (Chat Tab only) */}
        {activeTab === 'chat' && (
          <button 
            onClick={onNewChat}
            className="btn-primary" 
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '22px' }}
          >
            <PlusCircle size={18} /> New Conversation
          </button>
        )}

        {/* Activity Stats Panel */}
        <div>
          <h3 style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Activity Stats
          </h3>
          
          {stats ? (
            <div className="glass-panel" style={{ padding: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Total Turns:</span>
                <span style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>{stats.total_turns}</span>
              </div>
              
              <div>
                <span style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '4px', fontSize: '0.8rem' }}>Intents:</span>
                {Object.entries(stats.intents || {}).length === 0 ? (
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>No intents yet</span>
                ) : (
                  Object.entries(stats.intents || {}).map(([intent, count]) => (
                    <div key={intent} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
                      <span style={{ color: '#cbd5e1' }}>{intent}</span>
                      <span style={{ color: 'var(--aurora-blue)', fontWeight: 600 }}>{count as React.ReactNode}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Loading stats...</p>
          )}
        </div>

      </div>

      {/* Logout Action */}
      <button 
        onClick={onLogout}
        className="btn-secondary"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: '12px', flexShrink: 0 }}
      >
        <LogOut size={16} /> Logout
      </button>
    </aside>
  );
}
