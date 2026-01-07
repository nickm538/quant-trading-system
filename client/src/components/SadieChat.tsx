import { useState, useRef, useEffect } from 'react';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Loader2, Send, Trash2, Brain, TrendingUp, Bot, User, Sparkles, AlertTriangle, CheckCircle, Target, Zap } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  data?: {
    symbol_detected?: string;
    model_used?: string;
    nuke_mode?: boolean;
  };
}

export function SadieChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'system',
      content: `🐿️ **Welcome to Sadie AI** - Your Ultimate Financial Intelligence Assistant

I'm powered by **GPT o1 Thinking Mode** with deep pattern recognition and smart connections.

**Commands:**
• Regular analysis: "Analyze NVDA", "Best options for AAPL"
• **NUKE MODE** ☢️: "Nuke $NVDA" for maximum overdrive analysis

**Ask me anything:** Market overview, TTM Squeeze setups, 5:1 R/R trades, options flow`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const chatMutation = trpc.ml.sadieChat.useMutation();
  const clearHistoryMutation = trpc.ml.sadieClearHistory.useMutation();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await chatMutation.mutateAsync({ message: userMessage.content });

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.success
          ? response.message
          : `⚠️ Sorry, I encountered an error: ${response.message}`,
        timestamp: new Date(),
        data: response.data,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Connection error: ${error.message}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleClearHistory = async () => {
    try {
      await clearHistoryMutation.mutateAsync();
      setMessages([
        {
          id: 'cleared',
          role: 'system',
          content: '🔄 Conversation cleared. How can I help you today?',
          timestamp: new Date(),
        },
      ]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Check if current input is NUKE mode
  const isNukeMode = input.toLowerCase().includes('nuke');

  return (
    <div className="flex flex-col h-[calc(100vh-180px)]">
      {/* Compact Header */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <div className="relative">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-md">
              <span className="text-lg">🐿️</span>
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-green-500 rounded-full border border-background animate-pulse" />
          </div>
          <div>
            <h2 className="text-base font-semibold flex items-center gap-2">
              Sadie AI
              <Badge variant="secondary" className="bg-gradient-to-r from-purple-500 to-pink-500 text-white text-[10px] px-1.5 py-0">
                <Brain className="w-2.5 h-2.5 mr-0.5" />
                o1 Thinking
              </Badge>
            </h2>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClearHistory}
          className="text-muted-foreground hover:text-destructive h-7 px-2"
        >
          <Trash2 className="w-3.5 h-3.5 mr-1" />
          Clear
        </Button>
      </div>

      {/* Messages Area - Takes up most of the space */}
      <Card className="flex-1 overflow-hidden border-orange-500/20">
        <ScrollArea className="h-full p-4" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[95%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white'
                      : message.role === 'system'
                      ? 'bg-gradient-to-r from-slate-800 to-slate-700 border border-slate-600'
                      : message.data?.nuke_mode
                      ? 'bg-gradient-to-br from-red-950/50 to-orange-950/50 border-2 border-red-500/50'
                      : 'bg-muted/50 border border-border'
                  }`}
                >
                  {/* Message Header */}
                  <div className="flex items-center gap-2 mb-2">
                    {message.role === 'user' ? (
                      <User className="w-3.5 h-3.5" />
                    ) : message.data?.nuke_mode ? (
                      <span className="text-lg">☢️</span>
                    ) : (
                      <Bot className="w-3.5 h-3.5 text-orange-500" />
                    )}
                    <span className="text-xs opacity-70">
                      {message.role === 'user' ? 'You' : message.data?.nuke_mode ? 'Sadie NUKE MODE' : 'Sadie'}
                      {' • '}
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.data?.symbol_detected && (
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        <TrendingUp className="w-2.5 h-2.5 mr-0.5" />
                        {message.data.symbol_detected}
                      </Badge>
                    )}
                    {message.data?.nuke_mode && (
                      <Badge className="bg-red-600 text-white text-[10px] px-1.5 py-0 animate-pulse">
                        ☢️ MAXIMUM OVERDRIVE
                      </Badge>
                    )}
                  </div>

                  {/* Message Content - Enhanced Formatting */}
                  <div className="text-sm leading-relaxed">
                    <MessageContent content={message.content} isNukeMode={message.data?.nuke_mode} />
                  </div>

                  {/* Model Info */}
                  {message.data?.model_used && (
                    <div className="mt-3 pt-2 border-t border-border/30">
                      <span className="text-[10px] opacity-50 flex items-center gap-1">
                        <Sparkles className="w-2.5 h-2.5" />
                        {message.data.model_used}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className={`border rounded-2xl px-4 py-3 ${
                  isNukeMode 
                    ? 'bg-gradient-to-br from-red-950/50 to-orange-950/50 border-red-500/50' 
                    : 'bg-muted/50 border-border'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <Loader2 className={`w-5 h-5 animate-spin ${isNukeMode ? 'text-red-500' : 'text-orange-500'}`} />
                      <Brain className="w-3 h-3 absolute -top-1 -right-1 text-purple-500 animate-pulse" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">
                        {isNukeMode ? '☢️ NUKE MODE: Deep analysis in progress...' : 'Sadie is thinking deeply...'}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {isNukeMode 
                          ? 'Running 14-section comprehensive analysis, multi-timeframe forecasts, smart money tracking...'
                          : 'Running pattern analysis, smart connections, forecasting models...'
                        }
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Input Area */}
      <div className="mt-3 flex gap-2">
        <Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={isNukeMode ? '☢️ NUKE MODE: Maximum overdrive analysis...' : 'Ask Sadie about stocks, options, patterns, or type "Nuke $SYMBOL" for full analysis...'}
          className={`flex-1 h-11 ${isNukeMode ? 'border-red-500/50 bg-red-950/20' : ''}`}
          disabled={isLoading}
        />
        <Button
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className={`h-11 px-5 ${
            isNukeMode 
              ? 'bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700'
              : 'bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600'
          }`}
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : isNukeMode ? (
            <Zap className="w-4 h-4" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
    </div>
  );
}

// Enhanced helper component to render markdown-like content with proper formatting
function MessageContent({ content, isNukeMode }: { content: string; isNukeMode?: boolean }) {
  const lines = content.split('\n');
  const elements: JSX.Element[] = [];
  let inCodeBlock = false;
  let codeBlockContent: string[] = [];
  let inTable = false;
  let tableRows: string[] = [];

  const processInlineFormatting = (text: string): string => {
    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-foreground">$1</strong>');
    // Italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Inline code
    text = text.replace(/`(.*?)`/g, '<code class="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">$1</code>');
    // Links
    text = text.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" class="text-orange-500 hover:underline" target="_blank">$1</a>');
    return text;
  };

  const renderTable = (rows: string[]): JSX.Element => {
    const headerRow = rows[0];
    const dataRows = rows.slice(2); // Skip header and separator
    
    const headers = headerRow.split('|').filter(h => h.trim()).map(h => h.trim());
    
    return (
      <div className="overflow-x-auto my-3">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-muted/50">
              {headers.map((header, i) => (
                <th key={i} className="border border-border/50 px-3 py-2 text-left font-semibold">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataRows.map((row, rowIndex) => {
              const cells = row.split('|').filter(c => c.trim()).map(c => c.trim());
              return (
                <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
                  {cells.map((cell, cellIndex) => (
                    <td 
                      key={cellIndex} 
                      className="border border-border/50 px-3 py-2"
                      dangerouslySetInnerHTML={{ __html: processInlineFormatting(cell) }}
                    />
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  lines.forEach((line, index) => {
    // Code block handling
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`code-${index}`} className="bg-slate-900 text-slate-100 p-3 rounded-lg text-xs font-mono overflow-x-auto my-2">
            <code>{codeBlockContent.join('\n')}</code>
          </pre>
        );
        codeBlockContent = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBlockContent.push(line);
      return;
    }

    // Table handling
    if (line.includes('|') && line.trim().startsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      tableRows.push(line);
      return;
    } else if (inTable) {
      elements.push(
        <div key={`table-${index}`}>
          {renderTable(tableRows)}
        </div>
      );
      tableRows = [];
      inTable = false;
    }

    // Section headers with special styling
    if (line.startsWith('===') && line.endsWith('===')) {
      const headerText = line.replace(/=/g, '').trim();
      elements.push(
        <div key={index} className={`mt-4 mb-2 p-2 rounded-lg ${isNukeMode ? 'bg-red-950/30 border border-red-500/30' : 'bg-orange-500/10 border border-orange-500/20'}`}>
          <h3 className="font-bold text-sm flex items-center gap-2">
            {headerText.includes('🕵️') || headerText.includes('SMART') ? <Target className="w-4 h-4 text-purple-500" /> : null}
            {headerText.includes('🎯') || headerText.includes('BOHEN') ? <Target className="w-4 h-4 text-green-500" /> : null}
            {headerText.includes('☢️') || headerText.includes('NUKE') ? <span>☢️</span> : null}
            <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(headerText) }} />
          </h3>
        </div>
      );
      return;
    }

    // H1 Headers
    if (line.startsWith('# ')) {
      elements.push(
        <h1 key={index} className="font-bold text-xl mt-4 mb-2 text-foreground border-b border-border/50 pb-1">
          {line.replace(/^#\s*/, '')}
        </h1>
      );
      return;
    }

    // H2 Headers
    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={index} className="font-bold text-lg mt-4 mb-2 text-foreground">
          {line.replace(/^##\s*/, '')}
        </h2>
      );
      return;
    }

    // H3 Headers
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={index} className="font-semibold text-base mt-3 mb-1.5 text-foreground">
          {line.replace(/^###\s*/, '')}
        </h3>
      );
      return;
    }

    // H4 Headers (--- Section ---)
    if (line.startsWith('---') && line.endsWith('---') && line.length > 6) {
      const headerText = line.replace(/^-+\s*/, '').replace(/\s*-+$/, '');
      elements.push(
        <h4 key={index} className="font-semibold text-sm mt-3 mb-1 text-muted-foreground border-b border-border/30 pb-1">
          {headerText}
        </h4>
      );
      return;
    }

    // Horizontal rule
    if (line.trim() === '---' || line.trim() === '***' || line.trim() === '___') {
      elements.push(<hr key={index} className="my-3 border-border/50" />);
      return;
    }

    // Numbered lists
    const numberedMatch = line.match(/^(\d+)\.\s+(.*)$/);
    if (numberedMatch) {
      elements.push(
        <div key={index} className="flex items-start gap-2 ml-1 my-1">
          <span className="font-semibold text-orange-500 min-w-[20px]">{numberedMatch[1]}.</span>
          <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(numberedMatch[2]) }} />
        </div>
      );
      return;
    }

    // Bullet points
    if (line.trim().startsWith('•') || line.trim().startsWith('-') || line.trim().startsWith('*')) {
      const bulletContent = line.replace(/^[\s•\-\*]+/, '').trim();
      
      // Check for special indicators
      let bulletColor = 'text-orange-500';
      let icon = null;
      
      if (bulletContent.includes('✅') || bulletContent.toLowerCase().includes('bullish') || bulletContent.toLowerCase().includes('buy')) {
        bulletColor = 'text-green-500';
      } else if (bulletContent.includes('❌') || bulletContent.toLowerCase().includes('bearish') || bulletContent.toLowerCase().includes('sell')) {
        bulletColor = 'text-red-500';
      } else if (bulletContent.includes('⚠️') || bulletContent.toLowerCase().includes('caution') || bulletContent.toLowerCase().includes('warning')) {
        bulletColor = 'text-yellow-500';
      }
      
      elements.push(
        <div key={index} className="flex items-start gap-2 ml-2 my-0.5">
          <span className={`${bulletColor} mt-1`}>•</span>
          <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(bulletContent) }} />
        </div>
      );
      return;
    }

    // Blockquotes
    if (line.startsWith('>')) {
      elements.push(
        <blockquote key={index} className="border-l-4 border-orange-500/50 pl-3 my-2 italic text-muted-foreground">
          <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(line.replace(/^>\s*/, '')) }} />
        </blockquote>
      );
      return;
    }

    // Key-value pairs (Label: Value)
    const kvMatch = line.match(/^([A-Za-z\s]+):\s*(.+)$/);
    if (kvMatch && !line.includes('http')) {
      elements.push(
        <div key={index} className="flex items-start gap-2 my-0.5">
          <span className="text-muted-foreground min-w-[120px]">{kvMatch[1]}:</span>
          <span className="font-medium" dangerouslySetInnerHTML={{ __html: processInlineFormatting(kvMatch[2]) }} />
        </div>
      );
      return;
    }

    // Empty lines
    if (!line.trim()) {
      elements.push(<div key={index} className="h-2" />);
      return;
    }

    // Regular paragraphs
    elements.push(
      <p key={index} className="my-1" dangerouslySetInnerHTML={{ __html: processInlineFormatting(line) }} />
    );
  });

  // Handle any remaining table
  if (inTable && tableRows.length > 0) {
    elements.push(
      <div key="final-table">
        {renderTable(tableRows)}
      </div>
    );
  }

  return <div className="space-y-0.5">{elements}</div>;
}

export default SadieChat;
