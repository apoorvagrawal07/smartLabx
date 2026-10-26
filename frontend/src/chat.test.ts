import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, apiRequest, conversationTitle, errorMessage, parseConversationStore, parseSources, upsertConversation } from './chat';

afterEach(() => vi.unstubAllGlobals());

describe('browser conversation history', () => {
  const conversation = { id: 'test-thread', title: 'A question', updatedAt: 10, messages: [{ id: 'message', role: 'user', content: 'Hello', sources: [] }] };
  it('restores valid history and rejects malformed records', () => {
    expect(parseConversationStore(JSON.stringify([conversation, null, { id: 42 }]))).toEqual([conversation]);
    expect(parseConversationStore('{broken')).toEqual([]);
    expect(parseConversationStore(null)).toEqual([]);
    expect(parseConversationStore(JSON.stringify([{ ...conversation, messages: [null] }]))).toEqual([]);
  });
  it('normalizes source data and rejects malformed entries', () => {
    expect(parseSources([{ id: 'S1', source: 'knowledge/kafka.md', content: 'Evidence' }, { id: 1 }])).toEqual([
      { id: 'S1', source: 'knowledge/kafka.md', content: 'Evidence', chunk_id: undefined },
    ]);
  });
  it('updates only the target conversation and keeps newest first', () => {
    const original = parseConversationStore(JSON.stringify([conversation]))[0];
    const updated = { ...original, title: 'Updated', updatedAt: 20 };
    expect(upsertConversation([original], updated)).toEqual([updated]);
    expect(original.title).toBe('A question');
  });
  it('makes readable, bounded titles', () => {
    expect(conversationTitle('  hello\nworld  ')).toBe('hello world');
    expect(conversationTitle('a'.repeat(80))).toHaveLength(48);
  });
});

describe('API requests', () => {
  it('retains session cookies and sends the supplied body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ answer: '5250' })));
    vi.stubGlobal('fetch', fetchMock);
    const body = JSON.stringify({ message: 'Calculate 75 * 70', conversation_id: 'thread' });
    expect(await apiRequest('/chat', { method: 'POST', body })).toEqual({ answer: '5250' });
    expect(fetchMock).toHaveBeenCalledWith('/chat', expect.objectContaining({ credentials: 'same-origin', body, method: 'POST' }));
  });
  it('handles empty delete responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
    expect(await apiRequest('/memories/key', { method: 'DELETE' })).toBeUndefined();
  });
  it('does not expose raw server errors or silently retry requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('private server detail', { status: 403 }));
    vi.stubGlobal('fetch', fetchMock);
    await expect(apiRequest('/chat')).rejects.toEqual(new ApiError(403, errorMessage(403)));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it('reports network failures without exposing exception details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('private connection detail')));
    await expect(apiRequest('/chat')).rejects.toThrow('Cannot reach SmartLab');
  });
});