"use client";

import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Sidebar from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';
import { Bot, LogIn } from 'lucide-react';

export default function Home() {
  const [userName, setUserName] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string>('');
  const [showLogin, setShowLogin] = useState<boolean>(true);
  const [loginInput, setLoginInput] = useState('');

  // Login handler
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginInput.trim()) return;

    try {
      // Assuming backend is running on 8000
      const response = await fetch('http://localhost:8000/user/login', {
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
        onNewChat={() => setThreadId(uuidv4())} 
        onLogout={handleLogout} 
      />
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', position: 'relative' }}>
        <ChatInterface 
          userName={userName!} 
          userId={userId!} 
          threadId={threadId} 
        />
      </main>
    </div>
  );
}
