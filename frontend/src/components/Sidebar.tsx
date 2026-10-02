"use client";

import { useEffect, useState } from 'react';
import { LogOut, PlusCircle, Activity, MessageSquare, UploadCloud, Database } from 'lucide-react';
import { getApiBaseUrl } from '@/utils/api';

interface SidebarProps {
  userName: string;
  threadId: string;
  activeTab: 'chat' | 'ingest';
  onSelectTab: (tab: 'chat' | 'ingest') => void;
  onNewChat: () => void;
  onLogout: () => void;
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
  onLogout 
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

  return (
    <aside style={{
      width: '280px',
      background: 'rgba(0,0,0,0.5)',
      borderRight: '1px solid var(--glass-border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '20px'
    }}>
      {/* User Header */}
      <div style={{ marginBottom: '24px' }}>
        <h2 className="text-aurora" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem' }}>
          <Activity size={20} />
          {userName}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '5px' }}>
          Thread: {threadId.substring(0, 8)}...
        </p>
      </div>

      {/* Page Navigation Tabs */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '25px' }}>
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
      </div>

      {/* New Conversation Button (Chat Tab only) */}
      {activeTab === 'chat' && (
        <button 
          onClick={onNewChat}
          className="btn-primary" 
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '25px' }}
        >
          <PlusCircle size={18} /> New Conversation
        </button>
      )}

      {/* Activity Stats Panel */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Activity Stats
        </h3>
        
        {stats ? (
          <div className="glass-panel" style={{ padding: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Total Turns:</span>
              <span style={{ fontWeight: 'bold' }}>{stats.total_turns}</span>
            </div>
            
            <div>
              <span style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '6px', fontSize: '0.85rem' }}>Intents:</span>
              {Object.entries(stats.intents || {}).length === 0 ? (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>No intents yet</span>
              ) : (
                Object.entries(stats.intents || {}).map(([intent, count]) => (
                  <div key={intent} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span>{intent}</span>
                    <span style={{ color: 'var(--aurora-blue)' }}>{count as React.ReactNode}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        ) : (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Loading stats...</p>
        )}
      </div>

      {/* Logout Action */}
      <button 
        onClick={onLogout}
        className="btn-secondary"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: '16px' }}
      >
        <LogOut size={18} /> Logout
      </button>
    </aside>
  );
}
