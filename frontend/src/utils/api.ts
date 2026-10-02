/**
 * Helper to determine the backend API base URL.
 * Automatically adapts if accessing via localhost or LAN IP (e.g. 192.168.x.x).
 */
export const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined' && window.location.hostname && window.location.hostname !== 'localhost') {
    return `http://${window.location.hostname}:8000`;
  }
  return 'http://localhost:8000';
};
