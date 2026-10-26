export type Source = { id: string; source: string; content: string; chunk_id?: string };
export type Message = { id: string; role: 'user' | 'assistant'; content: string; sources: Source[]; error?: string };
export type Conversation = { id: string; title: string; updatedAt: number; messages: Message[] };
export type Memory = { key: string; value: string };
export const STORAGE_KEY = 'smartlab.conversations.v1';

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export function errorMessage(status: number): string {
  if (status === 403) return 'This session cannot access that conversation. Start a new conversation.';
  if (status === 422 || status === 400) return 'The request was not accepted. Check your input and try again.';
  if (status === 429) return 'Too many requests. Please try again shortly.';
  if (status === 503 || status === 502 || status === 504) return 'The model or database is unavailable. Please try again shortly.';
  return 'The request could not be completed. Please try again.';
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...init.headers },
      signal: AbortSignal.timeout(240_000),
    });
  } catch (error) {
    if (error instanceof DOMException && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
      throw new Error('The request timed out. The server may still be processing it.');
    }
    throw new Error('Cannot reach SmartLab. Check that the API is running.');
  }
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status));
  if (response.status === 204) return undefined as T;
  try { return await response.json() as T; }
  catch { throw new Error('The server returned an unreadable response.'); }
}

export function parseSources(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item) => item && typeof item.id === 'string' && typeof item.source === 'string' && typeof item.content === 'string')
    .map((item) => ({ id: item.id, source: item.source, content: item.content, chunk_id: typeof item.chunk_id === 'string' ? item.chunk_id : undefined }));
}

export function parseConversationStore(raw: string | null): Conversation[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is Conversation => {
      if (!item || typeof item !== 'object') return false;
      if (typeof item.id !== 'string' || !item.id.trim() || item.id.length > 128 || typeof item.title !== 'string') return false;
      if (typeof item.updatedAt !== 'number' || !Number.isFinite(item.updatedAt) || !Array.isArray(item.messages)) return false;
      return item.messages.every((message: Message) => message && typeof message.id === 'string'
        && (message.role === 'user' || message.role === 'assistant') && typeof message.content === 'string'
        && Array.isArray(message.sources) && (message.error === undefined || typeof message.error === 'string'));
    }).map((item) => ({ ...item, messages: item.messages.map((message) => ({ ...message, sources: parseSources(message.sources) })) }));
  } catch { return []; }
}

export function createConversation(): Conversation {
  return { id: crypto.randomUUID(), title: 'New conversation', updatedAt: Date.now(), messages: [] };
}

export function conversationTitle(message: string): string {
  const clean = message.trim().replace(/\s+/g, ' ');
  return clean.length > 48 ? `${clean.slice(0, 45)}...` : clean;
}

export function upsertConversation(conversations: Conversation[], conversation: Conversation): Conversation[] {
  return [conversation, ...conversations.filter((item) => item.id !== conversation.id)]
    .sort((first, second) => second.updatedAt - first.updatedAt);
}