import { useState, useRef, useEffect } from 'react';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Loader2, Send, Trash2, Brain, TrendingUp, Bot, User, Sparkles, AlertTriangle, CheckCircle, Target, Zap, Image, Paperclip, X, FileText, Upload } from 'lucide-react';

interface AttachedFile {
  id: string;
  file: File;
  preview?: string;
  type: 'image' | 'document';
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  attachments?: {
    type: 'image' | 'document';
    name: string;
    url?: string;
  }[];
  data?: {
    symbol_detected?: string;
    model_used?: string;
    nuke_mode?: boolean;
    perplexity_used?: boolean;
    vision_used?: boolean;
  };
}

export function SadieChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'system',
      content: `🐿️ **Welcome to Sadie AI v2.1** - Your Ultimate Financial Intelligence Assistant

I'm powered by **GPT-4o Vision** + **Perplexity AI** for real-time research with deep pattern recognition and smart connections.

**NEW: Image Upload** 📷
Upload charts, screenshots, or documents and I'll analyze them!

**Commands:**
• Regular analysis: "Analyze NVDA", "Best options for AAPL"
• **NUKE MODE** ☢️: "Nuke $NVDA" for maximum overdrive analysis
• **Image Analysis**: Upload a chart + "What do you see?"

**Ask me anything:** Market overview, TTM Squeeze setups, 5:1 R/R trades, options flow`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const chatMutation = trpc.ml.sadieChat.useMutation();
  const chatWithImageMutation = trpc.ml.sadieChatWithImage.useMutation();
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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newFiles: AttachedFile[] = [];
    
    Array.from(files).forEach((file) => {
      const isImage = file.type.startsWith('image/');
      const isDocument = file.type === 'application/pdf' || 
                         file.type.includes('document') ||
                         file.type === 'text/plain';
      
      if (!isImage && !isDocument) {
        alert('Please upload images (PNG, JPG, GIF, WebP) or documents (PDF, TXT)');
        return;
      }

      const attachedFile: AttachedFile = {
        id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        file,
        type: isImage ? 'image' : 'document',
      };

      // Create preview for images
      if (isImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setAttachedFiles((prev) => 
            prev.map((f) => 
              f.id === attachedFile.id 
                ? { ...f, preview: e.target?.result as string }
                : f
            )
          );
        };
        reader.readAsDataURL(file);
      }

      newFiles.push(attachedFile);
    });

    setAttachedFiles((prev) => [...prev, ...newFiles]);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (fileId: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const handleSend = async () => {
    if ((!input.trim() && attachedFiles.length === 0) || isLoading) return;

    // Prepare attachments info for display
    const messageAttachments = attachedFiles.map((f) => ({
      type: f.type,
      name: f.file.name,
      url: f.preview,
    }));

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim() || (attachedFiles.length > 0 ? 'Analyze this image' : ''),
      timestamp: new Date(),
      attachments: messageAttachments.length > 0 ? messageAttachments : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input.trim();
    const currentFiles = [...attachedFiles];
    setInput('');
    setAttachedFiles([]);
    setIsLoading(true);

    try {
      let response;

      if (currentFiles.length > 0) {
        // Convert files to base64 for sending
        const imagePromises = currentFiles
          .filter((f) => f.type === 'image')
          .map(async (f) => {
            return new Promise<string>((resolve) => {
              const reader = new FileReader();
              reader.onload = (e) => resolve(e.target?.result as string);
              reader.readAsDataURL(f.file);
            });
          });

        const imageBase64Array = await Promise.all(imagePromises);

        // Use vision-enabled endpoint
        response = await chatWithImageMutation.mutateAsync({
          message: currentInput || 'Analyze this image and provide insights',
          images: imageBase64Array,
        });
      } else {
        // Regular text chat
        response = await chatMutation.mutateAsync({ message: currentInput });
      }

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.success
          ? response.message
          : `⚠️ Sorry, I encountered an error: ${response.message}`,
        timestamp: new Date(),
        data: {
          ...response.data,
          vision_used: currentFiles.length > 0,
        },
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
      setAttachedFiles([]);
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
  const hasAttachments = attachedFiles.length > 0;

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
                Vision + o1
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
                    {message.data?.vision_used && (
                      <Badge className="bg-blue-600 text-white text-[10px] px-1.5 py-0">
                        <Image className="w-2.5 h-2.5 mr-0.5" />
                        Vision
                      </Badge>
                    )}
                  </div>

                  {/* Attachments Display */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-3">
                      {message.attachments.map((att, idx) => (
                        <div key={idx} className="relative">
                          {att.type === 'image' && att.url ? (
                            <img 
                              src={att.url} 
                              alt={att.name}
                              className="max-w-[200px] max-h-[150px] rounded-lg border border-white/20 object-cover"
                            />
                          ) : (
                            <div className="flex items-center gap-2 bg-black/20 rounded-lg px-3 py-2">
                              <FileText className="w-4 h-4" />
                              <span className="text-xs">{att.name}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Message Content - Enhanced Formatting */}
                  <div className="text-sm leading-relaxed">
                    <MessageContent content={message.content} isNukeMode={message.data?.nuke_mode} />
                  </div>

                  {/* Model Info */}
                  {message.data?.model_used && (
                    <div className="mt-3 pt-2 border-t border-border/30">
                      <span className="text-[10px] opacity-50 flex items-center gap-2">
                        <span className="flex items-center gap-1">
                          <Sparkles className="w-2.5 h-2.5" />
                          {message.data.model_used}
                        </span>
                        {message.data.perplexity_used && (
                          <span className="flex items-center gap-1 text-blue-400">
                            <Zap className="w-2.5 h-2.5" />
                            + Perplexity
                          </span>
                        )}
                        {message.data.vision_used && (
                          <span className="flex items-center gap-1 text-green-400">
                            <Image className="w-2.5 h-2.5" />
                            + Vision
                          </span>
                        )}
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
                        {hasAttachments 
                          ? '📷 Analyzing image with Vision AI...'
                          : isNukeMode 
                            ? '☢️ NUKE MODE: Deep analysis in progress...' 
                            : 'Sadie is thinking deeply...'
                        }
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {hasAttachments
                          ? 'Processing visual data, identifying patterns, extracting insights...'
                          : isNukeMode 
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

      {/* Attached Files Preview */}
      {attachedFiles.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 px-1">
          {attachedFiles.map((file) => (
            <div 
              key={file.id} 
              className="relative group bg-muted/50 rounded-lg border border-border overflow-hidden"
            >
              {file.type === 'image' && file.preview ? (
                <img 
                  src={file.preview} 
                  alt={file.file.name}
                  className="w-16 h-16 object-cover"
                />
              ) : (
                <div className="w-16 h-16 flex flex-col items-center justify-center p-2">
                  <FileText className="w-6 h-6 text-muted-foreground" />
                  <span className="text-[8px] text-muted-foreground truncate max-w-full mt-1">
                    {file.file.name.slice(0, 8)}...
                  </span>
                </div>
              )}
              <button
                onClick={() => removeFile(file.id)}
                className="absolute -top-1 -right-1 w-5 h-5 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input Area */}
      <div className="mt-3 flex gap-2">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.pdf,.txt"
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {/* Attachment button */}
        <Button
          variant="outline"
          size="icon"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="h-11 w-11 shrink-0 border-orange-500/30 hover:bg-orange-500/10 hover:border-orange-500"
          title="Attach image or document"
        >
          <Paperclip className="w-4 h-4" />
        </Button>

        <Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={
            hasAttachments 
              ? 'Describe what you want to know about the image...'
              : isNukeMode 
                ? '☢️ NUKE MODE: Maximum overdrive analysis...' 
                : 'Ask Sadie about stocks, options, patterns, or type "Nuke $SYMBOL" for full analysis...'
          }
          className={`flex-1 h-11 ${
            hasAttachments 
              ? 'border-blue-500/50 bg-blue-950/20'
              : isNukeMode 
                ? 'border-red-500/50 bg-red-950/20' 
                : ''
          }`}
          disabled={isLoading}
        />
        <Button
          onClick={handleSend}
          disabled={(!input.trim() && attachedFiles.length === 0) || isLoading}
          className={`h-11 px-5 ${
            hasAttachments
              ? 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700'
              : isNukeMode 
                ? 'bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700'
                : 'bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600'
          }`}
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : hasAttachments ? (
            <Upload className="w-4 h-4" />
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

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const key = `line-${i}`;

    // Code block handling
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={key} className="bg-slate-900 rounded-lg p-3 my-2 overflow-x-auto">
            <code className="text-xs font-mono text-green-400">
              {codeBlockContent.join('\n')}
            </code>
          </pre>
        );
        codeBlockContent = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockContent.push(line);
      continue;
    }

    // Table handling
    if (line.includes('|') && line.trim().startsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      tableRows.push(line);
      
      // Check if next line is not a table row
      if (i === lines.length - 1 || !lines[i + 1]?.includes('|')) {
        if (tableRows.length >= 2) {
          elements.push(<div key={key}>{renderTable(tableRows)}</div>);
        }
        inTable = false;
        tableRows = [];
      }
      continue;
    }

    // Headers
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={key} className={`text-base font-bold mt-4 mb-2 ${isNukeMode ? 'text-red-400' : 'text-orange-400'}`}>
          {line.replace('### ', '')}
        </h3>
      );
      continue;
    }
    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={key} className={`text-lg font-bold mt-4 mb-2 ${isNukeMode ? 'text-red-400' : 'text-orange-400'}`}>
          {line.replace('## ', '')}
        </h2>
      );
      continue;
    }
    if (line.startsWith('# ')) {
      elements.push(
        <h1 key={key} className={`text-xl font-bold mt-4 mb-2 ${isNukeMode ? 'text-red-400' : 'text-orange-400'}`}>
          {line.replace('# ', '')}
        </h1>
      );
      continue;
    }

    // Horizontal rule
    if (line.match(/^[-=]{3,}$/)) {
      elements.push(<hr key={key} className="my-4 border-border/50" />);
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      elements.push(
        <blockquote 
          key={key} 
          className="border-l-4 border-orange-500/50 pl-3 my-2 italic text-muted-foreground"
          dangerouslySetInnerHTML={{ __html: processInlineFormatting(line.replace('> ', '')) }}
        />
      );
      continue;
    }

    // List items
    if (line.match(/^[-•*]\s/)) {
      elements.push(
        <div key={key} className="flex gap-2 my-1">
          <span className="text-orange-500">•</span>
          <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(line.replace(/^[-•*]\s/, '')) }} />
        </div>
      );
      continue;
    }

    // Numbered list
    if (line.match(/^\d+\.\s/)) {
      const num = line.match(/^(\d+)\./)?.[1];
      elements.push(
        <div key={key} className="flex gap-2 my-1">
          <span className="text-orange-500 font-semibold min-w-[1.5rem]">{num}.</span>
          <span dangerouslySetInnerHTML={{ __html: processInlineFormatting(line.replace(/^\d+\.\s/, '')) }} />
        </div>
      );
      continue;
    }

    // Empty line
    if (line.trim() === '') {
      elements.push(<div key={key} className="h-2" />);
      continue;
    }

    // Regular paragraph
    elements.push(
      <p 
        key={key} 
        className="my-1"
        dangerouslySetInnerHTML={{ __html: processInlineFormatting(line) }}
      />
    );
  }

  return <>{elements}</>;
}
