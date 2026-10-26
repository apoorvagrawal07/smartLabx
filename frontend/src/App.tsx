import { useDeferredValue, useEffect, useRef, useState, type ButtonHTMLAttributes, type FormEvent, type ReactNode } from 'react';
import { ArrowUp, ArrowUpRight, BookOpen, Check, ChevronDown, ChevronRight, CircleHelp, Copy, Download, FileText, FlaskConical, LoaderCircle, Menu, MessageSquare, NotebookPen, PanelRight, Pencil, Plus, Search, Server, SquarePen, Trash2, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ApiError, STORAGE_KEY, apiRequest, conversationTitle, createConversation, parseConversationStore, parseSources, upsertConversation, type Conversation, type Memory, type Message, type Source } from './chat';

const suggestions = [
  { icon: BookOpen, label: 'Knowledge', message: 'What happens if a Kafka consumer crashes?', tone: 'green' },
  { icon: FlaskConical, label: 'Calculation', message: 'Calculate (125 + 75) * 3', tone: 'blue' },
  { icon: MessageSquare, label: 'Conversation', message: 'Explain what an API gateway is.', tone: 'coral' },
];

function IconButton({ label, children, className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; children: ReactNode }) {
  return <button type="button" aria-label={label} title={label} className={`icon-button ${className}`} {...props}>{children}</button>;
}

function Dialog({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return <dialog ref={ref} className="dialog" aria-labelledby="dialog-title" onCancel={(event) => { event.preventDefault(); onClose(); }}>
    <div className="dialog-heading"><h2 id="dialog-title">{title}</h2><IconButton label="Close dialog" onClick={onClose}><X size={18} /></IconButton></div>
    {children}
  </dialog>;
}

function Markdown({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml disallowedElements={['img']} components={{
    a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
    table: ({ children }) => <div className="table-scroll"><table>{children}</table></div>,
  }}>{content}</ReactMarkdown>;
}

function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) return <div className="panel-empty"><BookOpen size={28} strokeWidth={1.4} /><h3>No sources yet</h3><p>No documents are attached to this response.</p></div>;
  return <div className="source-list">
    <div className="panel-section-label">REFERENCED DOCUMENTS <span>{sources.length.toString().padStart(2, '0')}</span></div>
    {sources.map((source, index) => <details className="source-item" key={`${source.id}-${index}`} open={index === 0}>
      <summary><span className="file-icon"><FileText size={17} /></span><span><strong>{source.source.split(/[\\/]/).pop() || source.source}</strong><small>{source.id}</small></span><ChevronDown size={16} className="source-chevron" /></summary>
      <div className="source-content"><p>{source.content}</p><span className="source-path">{source.source}</span>{source.chunk_id && <span className="chunk-id">Chunk {source.chunk_id}</span>}</div>
    </details>)}
  </div>;
}

function MemoryPanel({ revision }: { revision: number }) {
  const [entries, setEntries] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [needsSession, setNeedsSession] = useState(false);
  const [error, setError] = useState('');
  const [memoryKey, setMemoryKey] = useState('');
  const [value, setValue] = useState('');
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true); setError('');
    apiRequest<Memory[]>('/memories').then((result) => {
      if (!Array.isArray(result) || !result.every((entry) => entry && typeof entry.key === 'string' && typeof entry.value === 'string')) throw new Error('The memory response could not be read.');
      if (active) { setEntries(result); setNeedsSession(false); }
    }).catch((failure) => {
      if (active) {
        setNeedsSession(failure instanceof ApiError && failure.status === 403);
        setError(failure instanceof ApiError && failure.status === 403 ? '' : failure.message);
      }
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [revision]);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (saving || !value.trim()) return;
    if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$/.test(memoryKey)) {
      setError('Use a key starting with a letter, number, or underscore; only dots and hyphens are also allowed.');
      return;
    }
    setSaving(true); setError('');
    try {
      await apiRequest('/memories', { method: 'POST', body: JSON.stringify({ key: memoryKey, value: value.trim() }) });
      setEntries((current) => [{ key: memoryKey, value: value.trim() }, ...current.filter((entry) => entry.key !== memoryKey)]);
      setMemoryKey(''); setValue(''); setEditing(false);
    } catch (failure) { setError((failure as Error).message); }
    finally { setSaving(false); }
  }

  async function remove(key: string) {
    setSaving(true); setError('');
    try {
      await apiRequest(`/memories/${encodeURIComponent(key)}`, { method: 'DELETE' });
      setEntries((current) => current.filter((entry) => entry.key !== key));
      if (memoryKey === key) { setMemoryKey(''); setValue(''); setEditing(false); }
      setDeleting(null);
    } catch (failure) { setError((failure as Error).message); }
    finally { setSaving(false); }
  }

  return <div className="memory-panel">
    {error && <div className="inline-error" role="alert">{error}</div>}
    {loading ? <div className="panel-empty"><LoaderCircle className="spin" size={24} /><p>Loading memories...</p></div>
      : needsSession ? <div className="panel-empty"><NotebookPen size={28} strokeWidth={1.4} /><h3>No chat session yet</h3><p>Send your first message to start a session.</p></div>
      : <>
        <div className="panel-section-label">SAVED FACTS <span>{entries.length.toString().padStart(2, '0')}</span></div>
        {!entries.length && <p className="no-memories">No saved memories.</p>}
        <div className="memory-list">{entries.map((entry) => <article className="memory-item" key={entry.key}>
          <div className="memory-item-heading"><code>{entry.key}</code><div className="memory-actions">
            <IconButton label={`Edit ${entry.key}`} disabled={saving} onClick={() => { setMemoryKey(entry.key); setValue(entry.value); setEditing(true); }}><Pencil size={14} /></IconButton>
            <IconButton label={`Delete ${entry.key}`} disabled={saving} onClick={() => setDeleting(entry.key)}><Trash2 size={14} /></IconButton>
          </div></div><p>{entry.value}</p>
          {deleting === entry.key && <div className="delete-memory"><span>Delete this memory?</span><button type="button" disabled={saving} onClick={() => remove(entry.key)}>Delete</button><button type="button" disabled={saving} onClick={() => setDeleting(null)}>Cancel</button></div>}
        </article>)}</div>
        <form className="memory-form" onSubmit={save}>
          <h3>{editing ? 'Edit memory' : 'Add a memory'}</h3>
          <label htmlFor="memory-key">Key</label><input id="memory-key" placeholder="preferred_language" value={memoryKey} onChange={(event) => setMemoryKey(event.target.value)} maxLength={64} required disabled={saving || editing} />
          <label htmlFor="memory-value">Value</label><textarea id="memory-value" placeholder="Python" value={value} onChange={(event) => setValue(event.target.value)} maxLength={1000} required disabled={saving} rows={3} />
          <div className="memory-form-actions"><button className="primary-button" disabled={saving || !memoryKey || !value.trim()}>{saving ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />}{editing ? 'Save changes' : 'Save memory'}</button>
            {editing && <button type="button" className="text-button" disabled={saving} onClick={() => { setEditing(false); setMemoryKey(''); setValue(''); }}>Cancel</button>}
          </div>
        </form>
      </>}
  </div>;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try { const saved = parseConversationStore(localStorage.getItem(STORAGE_KEY)); return saved.length ? saved : [createConversation()]; }
    catch { return [createConversation()]; }
  });
  const [activeId, setActiveId] = useState(conversations[0].id);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busyId, setBusyId] = useState<string | null>(null);
  const busyRef = useRef(false);
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const [viewportWidth, setViewportWidth] = useState(window.innerWidth);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [panel, setPanel] = useState<'sources' | 'memory' | null>(null);
  const [sourceMessageId, setSourceMessageId] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'offline'>('checking');
  const [storageWarning, setStorageWarning] = useState('');
  const [notice, setNotice] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [memoryRevision, setMemoryRevision] = useState(0);
  const [dialog, setDialog] = useState<{ kind: 'remove' | 'rename'; id: string } | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const mobileNavigation = viewportWidth <= 760;
  const panelOpen = panel !== null;
  const overlayOpen = panelOpen && viewportWidth <= 1150;
  const active = conversations.find((item) => item.id === activeId) || conversations[0];
  const draft = drafts[active.id] || '';
  const selectedMessage = active.messages.find((message) => message.id === sourceMessageId)
    || active.messages.at(-1);
  const sources = selectedMessage?.sources || [];
  const visibleConversations = conversations.filter((item) => item.messages.length > 0 && `${item.title} ${item.messages.map((message) => message.content).join(' ')}`.toLowerCase().includes(deferredSearch.toLowerCase()));

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); setStorageWarning(''); }
    catch { setStorageWarning('Browser storage is unavailable or full. New conversations will not survive a refresh.'); }
  }, [conversations]);
  useEffect(() => {
    let current = true;
    fetch('/openapi.json', { signal: AbortSignal.timeout(5000) }).then((response) => {
      if (current) setApiStatus(response.ok ? 'connected' : 'offline');
    }).catch(() => { if (current) setApiStatus('offline'); });
    return () => { current = false; };
  }, []);
  useEffect(() => { document.title = `${active.title} | SmartLab`; }, [active.title]);
  useEffect(() => {
    function resize() { setViewportWidth(window.innerWidth); }
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);
  useEffect(() => {
    if (!panelOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    panelRef.current?.querySelector<HTMLButtonElement>('button')?.focus();
    return () => { if (previous?.isConnected) previous.focus(); };
  }, [panelOpen]);
  useEffect(() => {
    if (!sidebarOpen || !mobileNavigation) return;
    const previous = document.activeElement as HTMLElement | null;
    sidebarRef.current?.querySelector<HTMLAnchorElement>('a')?.focus();
    return () => { if (previous?.isConnected) previous.focus(); };
  }, [sidebarOpen, mobileNavigation]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'instant', block: 'end' }); }, [active.id, active.messages.length, busyId]);
  useEffect(() => {
    if (composerRef.current) {
      composerRef.current.style.height = 'auto';
      composerRef.current.style.height = `${Math.min(composerRef.current.scrollHeight, 160)}px`;
    }
  }, [draft, active.id]);
  useEffect(() => {
    function escape(event: KeyboardEvent) { if (event.key === 'Escape' && !dialog) { setSidebarOpen(false); setPanel(null); } }
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [dialog]);

  function setDraft(value: string) { setDrafts((current) => ({ ...current, [active.id]: value })); }
  function selectConversation(id: string) { setActiveId(id); setSourceMessageId(null); setSidebarOpen(false); setNotice(''); }
  function newConversation() {
    if (busyRef.current) return;
    const empty = conversations.find((item) => item.messages.length === 0);
    const next = empty || createConversation();
    if (!empty) setConversations((current) => upsertConversation(current, next));
    selectConversation(next.id);
    composerRef.current?.focus();
  }

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || content.length > 10_000 || busyRef.current) return;
    const conversation = active;
    const messageId = crypto.randomUUID();
    const userMessage: Message = { id: messageId, role: 'user', content, sources: [] };
    busyRef.current = true; setBusyId(conversation.id); setNotice('');
    setSourceMessageId(null);
    setDrafts((current) => ({ ...current, [conversation.id]: '' }));
    setConversations((current) => upsertConversation(current, {
      ...conversation, title: conversation.messages.length ? conversation.title : conversationTitle(content), updatedAt: Date.now(), messages: [...conversation.messages, userMessage],
    }));
    try {
      const result = await apiRequest<{ answer: unknown; sources: unknown }>('/chat', {
        method: 'POST', body: JSON.stringify({ message: content, conversation_id: conversation.id }),
      });
      if (typeof result.answer !== 'string' || !result.answer.trim()) throw new Error('The model returned an empty response.');
      const reply: Message = { id: crypto.randomUUID(), role: 'assistant', content: result.answer, sources: parseSources(result.sources) };
      setConversations((current) => current.map((item) => item.id === conversation.id ? { ...item, updatedAt: Date.now(), messages: [...item.messages, reply] } : item));
      setApiStatus('connected'); setMemoryRevision((current) => current + 1);
      if (reply.sources.length) setSourceMessageId(reply.id);
    } catch (failure) {
      setConversations((current) => current.map((item) => item.id === conversation.id ? {
        ...item, messages: item.messages.map((message) => message.id === messageId ? { ...message, error: (failure as Error).message } : message),
      } : item));
    } finally {
      busyRef.current = false; setBusyId(null); composerRef.current?.focus();
    }
  }

  async function copyMessage(message: Message) {
    try {
      await navigator.clipboard.writeText(message.content); setCopiedId(message.id);
      window.setTimeout(() => setCopiedId((current) => current === message.id ? null : current), 2000);
    } catch { setNotice('Clipboard access is unavailable in this browser.'); }
  }
  function exportConversation() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(active, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = `smartlab-${active.id.replace(/[^a-z0-9-]/gi, '')}.json`; anchor.click();
    URL.revokeObjectURL(url);
  }
  function removeConversation() {
    if (!dialog) return;
    const remaining = conversations.filter((item) => item.id !== dialog.id);
    const next = remaining.length ? remaining : [createConversation()];
    setConversations(next);
    if (active.id === dialog.id) selectConversation(next[0].id);
    setDialog(null);
  }

  return <div className={`workspace ${panel ? 'panel-visible' : ''}`}>
    {sidebarOpen && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside ref={sidebarRef} className={`sidebar ${sidebarOpen ? 'is-open' : ''}`} aria-label="Conversation navigation" inert={(mobileNavigation && !sidebarOpen) || overlayOpen}>
      <a className="brand" href="/" onClick={(event) => { event.preventDefault(); newConversation(); }}><span className="brand-mark"><FlaskConical size={23} strokeWidth={1.8} /></span><span>SmartLab<span className="brand-period">.</span></span></a>
      <button className="new-chat" onClick={newConversation} disabled={!!busyId}><SquarePen size={17} /><span>New conversation</span><Plus size={16} /></button>
      <div className="search-field"><Search size={15} /><input aria-label="Search conversations" placeholder="Search conversations" value={search} onChange={(event) => setSearch(event.target.value)} />{search && <IconButton label="Clear search" onClick={() => setSearch('')}><X size={13} /></IconButton>}</div>
      <div className="sidebar-label">CONVERSATIONS <span>{conversations.filter((item) => item.messages.length).length}</span></div>
      <nav className="conversation-list">
        {visibleConversations.map((conversation) => <div className={`conversation-row ${active.id === conversation.id ? 'selected' : ''}`} key={conversation.id}>
          <button className="conversation-select" title={conversation.title} aria-current={active.id === conversation.id ? 'page' : undefined} onClick={() => selectConversation(conversation.id)}><MessageSquare size={15} /><span>{conversation.title}</span>{busyId === conversation.id && <LoaderCircle size={13} className="spin" />}</button>
          <IconButton label={`Remove ${conversation.title} from this browser`} className="conversation-remove" disabled={busyId === conversation.id} onClick={() => setDialog({ kind: 'remove', id: conversation.id })}><Trash2 size={13} /></IconButton>
        </div>)}
        {!visibleConversations.length && <div className="empty-history"><MessageSquare size={20} strokeWidth={1.4} /><span>{search ? 'No matching conversations' : 'No conversations yet'}</span></div>}
      </nav>
      <div className="sidebar-bottom"><button className={`sidebar-link ${panel === 'memory' ? 'active' : ''}`} onClick={() => { setPanel(panel === 'memory' ? null : 'memory'); setSidebarOpen(false); }}><NotebookPen size={17} />Saved memories<ChevronRight size={15} /></button>
        <div className="workspace-profile"><span className="profile-icon"><FlaskConical size={18} /></span><div><strong>Personal workspace</strong><span>Local session</span></div><span className="local-label">LOCAL</span></div>
      </div>
    </aside>

    <main className="chat-main" inert={overlayOpen || (mobileNavigation && sidebarOpen)}>
      <header className="topbar"><div className="topbar-heading"><IconButton label="Open navigation" className="mobile-menu" onClick={() => { setSidebarOpen(true); setPanel(null); }}><Menu size={20} /></IconButton><span className="breadcrumb">Workspace</span><ChevronRight className="breadcrumb-chevron" size={14} /><h1 title={active.title}>{active.title}</h1>{active.messages.length > 0 && <IconButton label="Rename conversation" onClick={() => { setRenameValue(active.title); setDialog({ kind: 'rename', id: active.id }); }}><Pencil size={14} /></IconButton>}</div>
        <div className="topbar-actions"><span className={`api-status ${apiStatus}`}><span />{apiStatus === 'checking' ? 'Connecting' : apiStatus === 'connected' ? 'API connected' : 'API unavailable'}</span><IconButton label="Export conversation" disabled={!active.messages.length} onClick={exportConversation}><Download size={17} /></IconButton><IconButton label="Toggle context panel" className={panel ? 'is-active' : ''} onClick={() => setPanel(panel ? null : 'sources')}><PanelRight size={18} /></IconButton></div>
      </header>
      {(storageWarning || notice) && <div className="workspace-notice" role="status"><CircleHelp size={16} /><span>{storageWarning || notice}</span>{notice && <IconButton label="Dismiss notification" onClick={() => setNotice('')}><X size={15} /></IconButton>}</div>}
      <section className={`message-scroll ${!active.messages.length ? 'is-empty' : ''}`} aria-label="Chat messages">
        {!active.messages.length ? <div className="welcome">
          <div className="welcome-emblem"><FlaskConical size={34} strokeWidth={1.5} /><span className="emblem-line" /></div>
          <span className="eyebrow">YOUR PERSONAL WORKSPACE</span>
          <h2>What are we exploring?</h2>
          <div className="suggestions">{suggestions.map(({ icon: Icon, label, message, tone }) => <button className="suggestion" key={label} disabled={!!busyId} onClick={() => { setDraft(message); composerRef.current?.focus(); }}><span className={`suggestion-icon ${tone}`}><Icon size={19} strokeWidth={1.7} /></span><span><small>{label}</small><strong>{message}</strong></span><ArrowUpRight size={17} /></button>)}</div>
          <div className="welcome-meta"><span /><span>SmartLab workspace</span><span /></div>
        </div> : <div className="message-thread" role="log" aria-live="polite" aria-relevant="additions">
          {active.messages.map((message) => <article className={`message ${message.role}`} key={message.id}>
            {message.role === 'assistant' && <div className="message-author"><span className="assistant-mark"><FlaskConical size={15} /></span><strong>SmartLab</strong><span>ASSISTANT</span></div>}
            <div className="message-content">{message.role === 'assistant' ? <Markdown content={message.content} /> : <p>{message.content}</p>}</div>
            {message.error && <div className="message-error" role="alert"><CircleHelp size={16} /><span>{message.error}</span></div>}
            {message.role === 'assistant' && <div className="message-actions"><IconButton label="Copy response" onClick={() => copyMessage(message)}>{copiedId === message.id ? <Check size={15} /> : <Copy size={15} />}</IconButton>{message.sources.length > 0 && <button className="source-trigger" onClick={() => { setSourceMessageId(message.id); setPanel('sources'); }}><BookOpen size={14} />{message.sources.length} {message.sources.length === 1 ? 'source' : 'sources'}<ChevronRight size={13} /></button>}</div>}
          </article>)}
          {busyId === active.id && <div className="thinking" role="status"><span className="assistant-mark"><FlaskConical size={15} /></span><span>Thinking<span className="thinking-dots">...</span></span><LoaderCircle size={15} className="spin" /></div>}
        </div>}
        <div ref={bottomRef} />
      </section>
      <div className="composer-region"><form className={`composer ${busyId ? 'is-busy' : ''}`} onSubmit={send}>
        <textarea ref={composerRef} aria-label="Message SmartLab" placeholder="Message SmartLab..." value={draft} maxLength={10_000} rows={1} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } }} />
        <div className="composer-bottom"><span className="composer-mode"><span className="mode-dot" />SmartLab<span className="mode-separator">/</span>Auto</span><div className="composer-controls">{draft.length > 9000 && <span className="character-count">{draft.length.toLocaleString()} / 10,000</span>}<button className="send-button" type="submit" aria-label={busyId ? 'Waiting for response' : 'Send message'} title={busyId ? 'Waiting for response' : 'Send message'} disabled={!!busyId || !draft.trim()}>{busyId ? <LoaderCircle size={19} className="spin" /> : <ArrowUp size={20} />}</button></div></div>
      </form><div className="composer-footer"><span>AI responses can be inaccurate. Check important details.</span><span className="local-models"><Server size={12} />Local models</span></div></div>
    </main>

    {panel && <><button className="panel-backdrop" aria-label="Close context panel" onClick={() => setPanel(null)} /><aside ref={panelRef} className="context-panel" aria-label="Context panel">
      <header className="panel-header"><h2>Context</h2><IconButton label="Close context panel" onClick={() => setPanel(null)}><X size={18} /></IconButton></header>
      <div className="panel-tabs" role="tablist" aria-label="Context views" onKeyDown={(event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 'sources' : event.key === 'End' ? 'memory' : panel === 'sources' ? 'memory' : 'sources';
        setPanel(next); document.getElementById(`${next}-tab`)?.focus();
      }}><button id="sources-tab" role="tab" tabIndex={panel === 'sources' ? 0 : -1} aria-selected={panel === 'sources'} aria-controls="context-content" onClick={() => setPanel('sources')}><BookOpen size={15} />Sources{sources.length > 0 && <span>{sources.length}</span>}</button><button id="memory-tab" role="tab" tabIndex={panel === 'memory' ? 0 : -1} aria-selected={panel === 'memory'} aria-controls="context-content" onClick={() => setPanel('memory')}><NotebookPen size={15} />Memory</button></div>
      <div className="panel-body" id="context-content" role="tabpanel" aria-labelledby={panel === 'sources' ? 'sources-tab' : 'memory-tab'}>{panel === 'sources' ? <SourceList sources={sources} /> : <MemoryPanel revision={memoryRevision} />}</div>
      <footer className="panel-footer"><FlaskConical size={13} />SmartLab<span>Personal workspace</span></footer>
    </aside></>}

    {dialog && <Dialog title={dialog.kind === 'remove' ? 'Remove conversation?' : 'Rename conversation'} onClose={() => setDialog(null)}>
      {dialog.kind === 'remove' ? <><p>This removes the conversation from this browser only. Its server-side history will not be deleted.</p><div className="dialog-actions"><button className="secondary-button" onClick={() => setDialog(null)}>Cancel</button><button className="danger-button" onClick={removeConversation}>Remove locally</button></div></>
        : <form onSubmit={(event) => { event.preventDefault(); if (!renameValue.trim()) return; setConversations((current) => current.map((item) => item.id === dialog.id ? { ...item, title: renameValue.trim() } : item)); setDialog(null); }}><label htmlFor="conversation-title">Conversation title</label><input id="conversation-title" autoFocus value={renameValue} onChange={(event) => setRenameValue(event.target.value)} maxLength={120} required /><div className="dialog-actions"><button type="button" className="secondary-button" onClick={() => setDialog(null)}>Cancel</button><button className="primary-button" disabled={!renameValue.trim()}>Save title</button></div></form>}
    </Dialog>}
  </div>;
}