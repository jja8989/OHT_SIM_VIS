export function getClientId(): string {
    if (typeof window === 'undefined') return '';  // SSR 방지
    let id = localStorage.getItem('client_id');
    if (!id) {
      id = 'cid-' + Math.random().toString(36).substring(2, 15);
      localStorage.setItem('client_id', id);
    }
    return id;
  }