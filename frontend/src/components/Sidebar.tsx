import { useEffect, useState } from 'react';
import { LogOut, PlusCircle, Activity } from 'lucide-react';

interface SidebarProps {
  userName: string;
  threadId: string;
  onNewChat: () => void;
  onLogout: () => void;
}

interface Stats {
  total_turns: number;
  intents: Record<string, number>;
  last_active: string;
}

export default function Sidebar({ userName, threadId, onNewChat, onLogout }: SidebarProps) {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`http://localhost:8000/user/${userName}/stats`);
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
      <div style={{ marginBottom: '30px' }}>
        <h2 className="text-aurora" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem' }}>
          <Activity size={20} />
          {userName}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '5px' }}>
          Thread: {threadId.substring(0, 8)}...
        </p>
      </div>

      <button 
        onClick={onNewChat}
        className="btn-primary" 
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '30px' }}
      >
        <PlusCircle size={18} /> New Conversation
      </button>

      <div style={{ flex: 1 }}>
        <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '15px', textTransform: 'uppercase' }}>
          Activity Stats
        </h3>
        
        {stats ? (
          <div className="glass-panel" style={{ padding: '15px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Total Turns:</span>
              <span style={{ fontWeight: 'bold' }}>{stats.total_turns}</span>
            </div>
            
            <div style={{ marginBottom: '10px' }}>
              <span style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '5px' }}>Intents:</span>
              {Object.entries(stats.intents).length === 0 ? (
                <span style={{ fontSize: '0.8rem' }}>No data</span>
              ) : (
                Object.entries(stats.intents).map(([intent, count]) => (
                  <div key={intent} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span>{intent}</span>
                    <span style={{ color: 'var(--aurora-blue)' }}>{count}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        ) : (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Loading stats...</p>
        )}
      </div>

      <button 
        onClick={onLogout}
        className="btn-secondary"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: 'auto' }}
      >
        <LogOut size={18} /> Logout
      </button>
    </aside>
  );
}
