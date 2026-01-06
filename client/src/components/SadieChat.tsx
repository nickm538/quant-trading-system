import { useState, useRef, useEffect } from 'react';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Loader2, Send, Trash2, Sparkles, Brain, TrendingUp, AlertTriangle, Zap, Bot, User } from 'lucide-react';

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
      content: `👋 **Welcome to Sadie AI** - Your Ultimate Financial Intelligence Assistant

I'm powered by **GPT-5 Thinking Mode** and integrated with the complete SadieAI financial engine suite:

🎯 **What I Can Do:**
• Deep stock & ETF analysis with 50+ technical indicators
• Monte Carlo simulations (20,000 paths) for price forecasting
• Options analysis with 12-factor institutional scoring
• Pattern recognition (TTM Squeeze, NR patterns, chart patterns)
• Risk management with Kelly Criterion position sizing
• Real-time market data and sentiment analysis

💡 **Try asking me:**
• "Analyze NVDA for a potential entry"
• "What's the best options play for AAPL right now?"
• "Scan the market for TTM Squeeze setups"
• "Is CVX a good buy after the Venezuela news?"

I think deeply about every question to give you gold-standard financial advice. Let's make some money! 🚀`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const chatMutation = trpc.trading.sadieChat.useMutation();
  const clearHistoryMutation = trpc.trading.sadieClearHistory.useMutation();

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

  // Quick action buttons
  const quickActions = [
    { label: 'Market Overview', query: 'Give me a quick market overview for today' },
    { label: 'Top Opportunities', query: 'What are the top trading opportunities right now?' },
    { label: 'Analyze SPY', query: 'Analyze SPY for a potential trade' },
    { label: 'Options Scan', query: 'Scan for the best options plays today' },
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] max-h-[800px]">
      {/* Header */}
      <Card className="mb-4 bg-gradient-to-r from-orange-500/10 via-amber-500/10 to-yellow-500/10 border-orange-500/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg">
                  <span className="text-2xl">🐿️</span>
                </div>
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-background animate-pulse" />
              </div>
              <div>
                <CardTitle className="text-xl flex items-center gap-2">
                  Sadie AI
                  <Badge variant="secondary" className="bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs">
                    <Brain className="w-3 h-3 mr-1" />
                    GPT-5 Thinking
                  </Badge>
                </CardTitle>
                <CardDescription className="text-sm">
                  Strategic Analysis & Dynamic Investment Engine
                </CardDescription>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearHistory}
              className="text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Clear
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-2 mb-4">
        {quickActions.map((action, index) => (
          <Button
            key={index}
            variant="outline"
            size="sm"
            onClick={() => {
              setInput(action.query);
              inputRef.current?.focus();
            }}
            className="text-xs hover:bg-orange-500/10 hover:border-orange-500/50"
          >
            <Zap className="w-3 h-3 mr-1 text-orange-500" />
            {action.label}
          </Button>
        ))}
      </div>

      {/* Messages Area */}
      <Card className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-4" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white'
                      : message.role === 'system'
                      ? 'bg-gradient-to-r from-slate-800 to-slate-700 border border-slate-600'
                      : 'bg-muted/50 border border-border'
                  }`}
                >
                  {/* Message Header */}
                  <div className="flex items-center gap-2 mb-2">
                    {message.role === 'user' ? (
                      <User className="w-4 h-4" />
                    ) : (
                      <Bot className="w-4 h-4 text-orange-500" />
                    )}
                    <span className="text-xs opacity-70">
                      {message.role === 'user' ? 'You' : 'Sadie'}
                      {' • '}
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.data?.symbol_detected && (
                      <Badge variant="outline" className="text-xs">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        {message.data.symbol_detected}
                      </Badge>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <MessageContent content={message.content} />
                  </div>

                  {/* Model Info */}
                  {message.data?.model_used && (
                    <div className="mt-2 pt-2 border-t border-border/50">
                      <span className="text-xs opacity-50 flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        Powered by {message.data.model_used}
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
                        Analyzing data, running models, validating signals
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
      <Card className="mt-4">
        <CardContent className="p-3">
          <div className="flex gap-2">
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask Sadie anything about stocks, options, or market analysis..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-center">
            💡 Tip: Ask about specific stocks like "$AAPL" or request analysis like "TTM Squeeze scan"
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper component to render markdown-like content
function MessageContent({ content }: { content: string }) {
  // Simple markdown-like rendering
  const lines = content.split('\n');

  return (
    <div className="space-y-2">
      {lines.map((line, index) => {
        // Headers
        if (line.startsWith('###')) {
          return (
            <h4 key={index} className="font-semibold text-base mt-3">
              {line.replace(/^###\s*/, '')}
            </h4>
          );
        }
        if (line.startsWith('##')) {
          return (
            <h3 key={index} className="font-semibold text-lg mt-4">
              {line.replace(/^##\s*/, '')}
            </h3>
          );
        }
        if (line.startsWith('#')) {
          return (
            <h2 key={index} className="font-bold text-xl mt-4">
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
              <span className="text-orange-500 mt-1">•</span>
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
          return <div key={index} className="h-2" />;
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
