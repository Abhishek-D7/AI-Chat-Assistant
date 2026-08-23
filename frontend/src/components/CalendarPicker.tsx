import { useState } from 'react';

interface CalendarPickerProps {
  onConfirm: (date: string, time: string) => void;
  onCancel: () => void;
}

export default function CalendarPicker({ onConfirm, onCancel }: CalendarPickerProps) {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [time, setTime] = useState('09:00');

  return (
    <div className="glass-panel" style={{ padding: '20px', marginTop: '10px', maxWidth: '400px' }}>
      <h3 style={{ marginBottom: '15px' }}>📅 Select Meeting Time</h3>
      
      <div style={{ display: 'flex', gap: '15px', marginBottom: '20px' }}>
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '5px' }}>Date</label>
          <input 
            type="date" 
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="input-glass" 
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '5px' }}>Time</label>
          <input 
            type="time" 
            value={time}
            onChange={(e) => setTime(e.target.value)}
            className="input-glass" 
            style={{ width: '100%' }}
          />
        </div>
      </div>
      
      <div style={{ display: 'flex', gap: '10px' }}>
        <button onClick={() => onConfirm(date, time)} className="btn-primary" style={{ flex: 1 }}>
          Confirm
        </button>
        <button onClick={onCancel} className="btn-secondary" style={{ flex: 1 }}>
          Cancel
        </button>
      </div>
    </div>
  );
}
