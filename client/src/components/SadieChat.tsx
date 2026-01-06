import { useState, useRef, useEffect } from 'react';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Loader2, Send, Trash2, Brain, TrendingUp, Bot, User, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  data?: {
    symbol_detected?: string;
    model_used?: string;
  };
}

export function SadieChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'system',
      content: `🐿️ **Welcome to Sadie AI** - Your Ultimate Financial Intelligence Assistant

I'm powered by **GPT o1 Thinking Mode** with deep pattern recognition and smart connections.

**Ask me anything:** "Analyze NVDA", "Best options for AAPL", "Market overview", "TTM Squeeze setups"`,
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
                  className={`max-w-[90%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white'
                      : message.role === 'system'
                      ? 'bg-gradient-to-r from-slate-800 to-slate-700 border border-slate-600'
                      : 'bg-muted/50 border border-border'
                  }`}
                >
                  {/* Message Header */}
                  <div className="flex items-center gap-2 mb-1.5">
                    {message.role === 'user' ? (
                      <User className="w-3.5 h-3.5" />
                    ) : (
                      <Bot className="w-3.5 h-3.5 text-orange-500" />
                    )}
                    <span className="text-xs opacity-70">
                      {message.role === 'user' ? 'You' : 'Sadie'}
                      {' • '}
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.data?.symbol_detected && (
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        <TrendingUp className="w-2.5 h-2.5 mr-0.5" />
                        {message.data.symbol_detected}
                      </Badge>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="prose prose-sm dark:prose-invert max-w-none text-sm">
                    <MessageContent content={message.content} />
                  </div>

                  {/* Model Info */}
                  {message.data?.model_used && (
                    <div className="mt-2 pt-1.5 border-t border-border/50">
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
                <div className="bg-muted/50 border border-border rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <Loader2 className="w-5 h-5 animate-spin text-orange-500" />
                      <Brain className="w-3 h-3 absolute -top-1 -right-1 text-purple-500 animate-pulse" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">Sadie is thinking deeply...</p>
                      <p className="text-xs text-muted-foreground">
                        Running pattern analysis, smart connections, forecasting models
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
          placeholder="Ask Sadie about stocks, options, patterns, or market analysis..."
          className="flex-1 h-11"
          disabled={isLoading}
        />
        <Button
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className="bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 h-11 px-5"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
    </div>
  );
}

// Helper component to render markdown-like content
function MessageContent({ content }: { content: string }) {
  const lines = content.split('\n');

  return (
    <div className="space-y-1.5">
      {lines.map((line, index) => {
        // Headers
        if (line.startsWith('###')) {
          return (
            <h4 key={index} className="font-semibold text-sm mt-2">
              {line.replace(/^###\s*/, '')}
            </h4>
          );
        }
        if (line.startsWith('##')) {
          return (
            <h3 key={index} className="font-semibold text-base mt-3">
              {line.replace(/^##\s*/, '')}
            </h3>
          );
        }
        if (line.startsWith('#')) {
          return (
            <h2 key={index} className="font-bold text-lg mt-3">
              {line.replace(/^#\s*/, '')}
            </h2>
          );
        }

        // Bold text
        let processed = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Bullet points
        if (line.trim().startsWith('•') || line.trim().startsWith('-') || line.trim().startsWith('*')) {
          return (
            <div key={index} className="flex items-start gap-2 ml-2">
              <span className="text-orange-500 mt-0.5">•</span>
              <span dangerouslySetInnerHTML={{ __html: processed.replace(/^[\s•\-\*]+/, '') }} />
            </div>
          );
        }

        // Tables (simple detection)
        if (line.includes('|') && line.trim().startsWith('|')) {
          return (
            <code key={index} className="block text-xs bg-muted/50 px-2 py-1 rounded">
              {line}
            </code>
          );
        }

        // Empty lines
        if (!line.trim()) {
          return <div key={index} className="h-1.5" />;
        }

        // Regular paragraphs
        return (
          <p key={index} dangerouslySetInnerHTML={{ __html: processed }} />
        );
      })}
    </div>
  );
}

export default SadieChat;
