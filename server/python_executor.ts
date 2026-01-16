import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

const PYTHON_SYSTEM_PATH = path.join(process.cwd(), 'python_system');

// Detect Python binary - use python3.11 for consistency
const PYTHON_BIN = 'python3.11';
const WRAPPER_SCRIPT = path.join(PYTHON_SYSTEM_PATH, 'run_analysis.py');
const PRODUCTION_ANALYZER = path.join(PYTHON_SYSTEM_PATH, 'run_perfect_analysis.py');

export interface StockAnalysisParams {
  symbol: string;
  monte_carlo_sims?: number;
  forecast_days?: number;
  bankroll?: number;
}

export interface StockAnalysisResult {
  symbol: string;
  current_price: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  target_price: number;
  stop_loss: number;
  position_size: number;
  technical_analysis: {
    overall_score: number;
    momentum_score: number;
    trend_score: number;
    volatility_score: number;
    rsi: number;
    adx: number;
    volatility: number;
  };
  stochastic_analysis: {
    expected_price: number;
    expected_return: number;
    confidence_interval_lower: number;
    confidence_interval_upper: number;
    var_95: number;
    cvar_95: number;
    max_drawdown: number;
    fat_tail_df: number;
  };
  options_analysis: {
    recommended_option: any | null;
    total_options_analyzed: number;
  };
  news_sentiment: {
    sentiment_score: number;
    total_articles: number;
    recent_headlines: string[];
  };
  risk_assessment: {
    risk_reward_ratio: number;
    potential_gain_pct: number;
    potential_loss_pct: number;
  };
  timestamp: string;
}

/**
 * Execute Python trading system analysis
 */
export async function analyzeStock(params: StockAnalysisParams): Promise<StockAnalysisResult> {
  const { symbol, monte_carlo_sims = 20000, forecast_days = 30, bankroll = 1000 } = params;

  // Use production analyzer with 100% real data (no placeholders)
  const command = `${PYTHON_BIN} ${PRODUCTION_ANALYZER} ${symbol} ${bankroll}`;

  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
      timeout: 120000, // 2 minute timeout
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        // Add library path for numpy/scipy C extensions in Nix
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    // Parse JSON output
    const result = JSON.parse(stdout);
    return result as StockAnalysisResult;
  } catch (error: any) {
    console.error('Python execution error:', error);
    throw new Error(`Failed to analyze stock ${symbol}: ${error.message}`);
  }
}

/**
 * Analyze options chain for a stock
 */
export async function getGreeksHeatmap(params: {
  symbol: string;
  num_strikes?: number;
  num_expirations?: number;
}): Promise<any> {
  const { symbol, num_strikes = 15, num_expirations = 6 } = params;

  const command = `${PYTHON_BIN} ${path.join(PYTHON_SYSTEM_PATH, 'greeks_heatmap.py')} ${symbol} ${num_strikes} ${num_expirations}`;

  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 120000, // 2 minutes for heatmap calculation
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Greeks heatmap error:', error);
    return {
      success: false,
      error: `Failed to generate Greeks heatmap for ${symbol}: ${error.message}`,
      symbol,
    };
  }
}

export async function analyzeOptions(params: {
  symbol: string;
  min_delta?: number;
  max_delta?: number;
  min_days?: number;
}): Promise<any> {
  // Widened default filters to return more results
  const { symbol, min_delta = 0.15, max_delta = 0.85, min_days = 3 } = params;

  const command = `${PYTHON_BIN} ${WRAPPER_SCRIPT} analyze_options ${symbol} ${min_delta} ${max_delta} ${min_days}`;

  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 600000, // 10 minutes for options chain analysis
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Options analysis error:', error);
    // Return JSON error instead of throwing to avoid HTML error page
    return {
      success: false,
      error: `Failed to analyze options for ${symbol}: ${error.message}`,
      symbol,
    };
  }
}

/**
 * Institutional-grade options analysis with advanced Greeks and pattern recognition
 */
export async function analyzeInstitutionalOptions(params: {
  symbol: string;
}): Promise<any> {
  const { symbol } = params;

  const INSTITUTIONAL_SCRIPT = path.join(PYTHON_SYSTEM_PATH, 'run_institutional_options.py');
  const command = `${PYTHON_BIN} ${INSTITUTIONAL_SCRIPT} ${symbol}`;

  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 600000, // 10 minutes for comprehensive analysis
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Institutional options analysis error:', error);
    return {
      success: false,
      error: `Failed to analyze institutional options for ${symbol}: ${error.message}`,
      symbol,
    };
  }
}

/**
 * Scan the market for best opportunities
 */
export async function scanMarket(params: {
  top_n?: number;
}): Promise<any> {
  const { top_n = 20 } = params;

  const command = `${PYTHON_BIN} ${WRAPPER_SCRIPT} scan_market ${top_n}`;

  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 50 * 1024 * 1024, // Increased to 50MB for large scans
      timeout: 1800000, // 30 minutes for full scan
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    // Extract JSON from stdout (may have logs before JSON)
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.error('No JSON found in stdout:', stdout.substring(0, 500));
      throw new Error('Python script did not return valid JSON');
    }

    return JSON.parse(jsonMatch[0]);
  } catch (error: any) {
    console.error('Market scan error:', error);
    // Return a proper error object instead of throwing
    return {
      error: true,
      message: error.message || 'Market scan failed',
      opportunities: [],
      total_analyzed: 0,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Quick health check for Python system
 */
/**
 * Ultimate Options Intelligence Engine - Market Scan
 * Scans entire market for best options opportunities
 */
export async function scanUltimateOptions(params: {
  max_results?: number;
  option_type?: 'call' | 'put' | 'both';
}): Promise<any> {
  const { max_results = 10, option_type = 'both' } = params;

  const ULTIMATE_SCRIPT = path.join(PYTHON_SYSTEM_PATH, 'run_ultimate_options.py');
  const command = `${PYTHON_BIN} ${ULTIMATE_SCRIPT} scan --max-results ${max_results} --type ${option_type}`;

  try {
    console.log(`🚀 Starting Ultimate Options scan for top ${max_results} ${option_type} opportunities...`);
    
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 50 * 1024 * 1024, // 50MB buffer
      timeout: 900000, // 15 minutes for full market scan
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Ultimate Options stderr:', stderr);
    }

    console.log(`✅ Ultimate Options scan completed`);
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Ultimate Options scan error:', error);
    return {
      success: false,
      error: `Ultimate Options scan failed: ${error.message}`,
      opportunities: [],
    };
  }
}

/**
 * Ultimate Options Intelligence Engine - Symbol Analysis
 * Deep analysis of a single stock's options
 */
export async function analyzeUltimateOptions(params: {
  symbol: string;
  option_type?: 'call' | 'put' | 'both';
}): Promise<any> {
  const { symbol, option_type = 'both' } = params;

  const ULTIMATE_SCRIPT = path.join(PYTHON_SYSTEM_PATH, 'run_ultimate_options.py');
  const command = `${PYTHON_BIN} ${ULTIMATE_SCRIPT} analyze ${symbol} --type ${option_type}`;

  try {
    console.log(`🔍 Starting Ultimate Options analysis for ${symbol}...`);
    
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 20 * 1024 * 1024, // 20MB buffer
      timeout: 300000, // 5 minutes for single symbol analysis
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Ultimate Options stderr:', stderr);
    }

    console.log(`✅ Ultimate Options analysis completed for ${symbol}`);
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Ultimate Options analysis error:', error);
    return {
      success: false,
      error: `Ultimate Options analysis failed for ${symbol}: ${error.message}`,
      symbol,
    };
  }
}

export async function checkPythonSystem(): Promise<boolean> {
  try {
    const command = `${PYTHON_BIN} ${WRAPPER_SCRIPT} health_check`;
    const { stdout } = await execAsync(command, { 
      timeout: 10000,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });
    return stdout.trim() === 'OK';
  } catch (error) {
    console.error('Python system check failed:', error);
    return false;
  }
}


/**
 * Get comprehensive fundamentals analysis with educational content
 */
export async function analyzeFundamentals(params: {
  symbol: string;
}): Promise<any> {
  const { symbol } = params;

  const command = `${PYTHON_BIN} -c "
import json
from fundamentals_analyzer import analyze_fundamentals
result = analyze_fundamentals('${symbol}')
print(json.dumps(result, default=str))
"`;

  try {
    console.log(`📊 Analyzing fundamentals for ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Fundamentals analysis error:', error);
    return {
      success: false,
      error: `Failed to analyze fundamentals for ${symbol}: ${error.message}`,
      symbol,
    };
  }
}

/**
 * Get trading education content
 */
export async function getEducation(params: {
  topic?: string;
}): Promise<any> {
  const { topic } = params;

  const topicArg = topic ? `'${topic}'` : 'None';
  const command = `${PYTHON_BIN} -c "
import json
from trading_education import get_education
result = get_education(${topicArg})
print(json.dumps(result, default=str))
"`;

  try {
    console.log(`📚 Getting education content${topic ? ` for: ${topic}` : ''}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 30000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });

    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Python stderr:', stderr);
    }

    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Education content error:', error);
    return {
      error: `Failed to get education content: ${error.message}`,
    };
  }
}


/**
 * Sadie AI Chatbot - The Ultimate Financial Intelligence Assistant
 */
export interface SadieChatParams {
  message: string;
}

export interface SadieChatWithImageParams {
  message: string;
  images: string[];  // Base64 encoded images
}

export interface SadieChatResult {
  success: boolean;
  message: string;
  data: {
    symbol_detected?: string;
    model_used?: string;
    tokens_used?: any;
  };
  timestamp: string;
}

/**
 * Chat with Sadie AI - GPT-5 powered financial assistant
 */
export async function sadieChat(params: SadieChatParams): Promise<SadieChatResult> {
  const { message } = params;
  
  // Escape the message for shell
  const escapedMessage = message.replace(/'/g, "'\\''");
  const command = `${PYTHON_BIN} run_sadie_chat.py chat '${escapedMessage}'`;
  
  try {
    console.log('🐿️ Sadie AI is thinking...');
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 20 * 1024 * 1024, // 20MB buffer for long responses
      timeout: 180000, // 3 minute timeout for thinking mode
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY || '',
        KEY: process.env.KEY || '', // Finnhub
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Sadie stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Sadie chat error:', error);
    return {
      success: false,
      message: `Sadie encountered an error: ${error.message}`,
      data: {},
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Chat with Sadie AI with image analysis - Vision-enabled financial assistant
 */
export async function sadieChatWithImage(params: SadieChatWithImageParams): Promise<SadieChatResult> {
  const { message, images } = params;
  
  try {
    console.log('🐿️📷 Sadie AI Vision is analyzing image...');
    
    // Write images to temp files and pass paths to Python
    const fs = await import('fs/promises');
    const os = await import('os');
    const tempDir = os.tmpdir();
    const imagePaths: string[] = [];
    
    for (let i = 0; i < images.length; i++) {
      const base64Data = images[i];
      // Extract the actual base64 data (remove data:image/...;base64, prefix)
      const base64Match = base64Data.match(/^data:image\/\w+;base64,(.+)$/);
      const pureBase64 = base64Match ? base64Match[1] : base64Data;
      
      const imagePath = path.join(tempDir, `sadie_image_${Date.now()}_${i}.png`);
      await fs.writeFile(imagePath, Buffer.from(pureBase64, 'base64'));
      imagePaths.push(imagePath);
    }
    
    // Escape the message for shell
    const escapedMessage = message.replace(/'/g, "'\\''")
    const escapedPaths = imagePaths.join(',');
    
    const command = `${PYTHON_BIN} run_sadie_chat.py chat_with_image '${escapedMessage}' '${escapedPaths}'`;
    
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 20 * 1024 * 1024, // 20MB buffer for long responses
      timeout: 180000, // 3 minute timeout for vision analysis
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY || '',
        KEY: process.env.KEY || '', // Finnhub
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Sadie Vision stderr:', stderr);
    }
    
    // Clean up temp files
    for (const imagePath of imagePaths) {
      try {
        await fs.unlink(imagePath);
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Sadie Vision error:', error);
    return {
      success: false,
      message: `Sadie Vision encountered an error: ${error.message}`,
      data: {},
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Get quick analysis from Sadie AI
 */
export async function sadieAnalyze(params: { symbol: string }): Promise<any> {
  const { symbol } = params;
  const command = `${PYTHON_BIN} run_sadie_chat.py analyze ${symbol}`;
  
  try {
    console.log(`🐿️ Sadie analyzing ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 20 * 1024 * 1024,
      timeout: 120000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Sadie analyze stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Sadie analyze error:', error);
    return {
      success: false,
      message: `Analysis failed: ${error.message}`,
      data: {},
    };
  }
}

/**
 * Clear Sadie's conversation history
 */
export async function sadieClearHistory(): Promise<any> {
  const command = `${PYTHON_BIN} run_sadie_chat.py clear`;
  
  try {
    const { stdout } = await execAsync(command, {
      maxBuffer: 1024 * 1024,
      timeout: 10000,
      cwd: PYTHON_SYSTEM_PATH,
    });
    
    return JSON.parse(stdout);
  } catch (error: any) {
    return {
      success: false,
      message: `Failed to clear history: ${error.message}`,
    };
  }
}


// ============================================================================
// NEW SCANNER MODULES (from financial-analysis-system)
// ============================================================================

const SCANNER_SCRIPT = path.join(PYTHON_SYSTEM_PATH, 'run_scanners.py');

/**
 * Dark Pool Scanner - Insider Movement/Oracle Analysis
 */
export async function scanDarkPool(params: { symbol: string }): Promise<any> {
  const { symbol } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} dark_pool ${symbol}`;
  
  try {
    console.log(`🔮 Scanning dark pool activity for ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Dark pool scanner stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Dark pool scanner error:', error);
    return {
      error: `Dark pool scan failed: ${error.message}`,
      symbol,
    };
  }
}

/**
 * TTM Squeeze Scanner - Volatility Compression Detection
 */
export async function scanTTMSqueeze(params: { symbol: string }): Promise<any> {
  const { symbol } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} ttm_squeeze ${symbol}`;
  
  try {
    console.log(`🔴 Scanning TTM Squeeze for ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        TWELVEDATA_API_KEY: process.env.TWELVEDATA_API_KEY || '5e7a5daaf41d46a8966963106ebef210',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('TTM Squeeze scanner stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('TTM Squeeze scanner error:', error);
    return {
      error: `TTM Squeeze scan failed: ${error.message}`,
      symbol,
    };
  }
}

/**
 * Options Flow Scanner - Bear to Bull Pressure Analysis
 */
export async function scanOptionsFlow(params: { symbol: string }): Promise<any> {
  const { symbol } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} options_flow ${symbol}`;
  
  try {
    console.log(`📊 Scanning options flow for ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Options flow scanner stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Options flow scanner error:', error);
    return {
      error: `Options flow scan failed: ${error.message}`,
      symbol,
    };
  }
}

/**
 * Breakout Detector - Multi-Signal Breakout Analysis
 */
export async function scanBreakout(params: { symbol: string }): Promise<any> {
  const { symbol } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} breakout ${symbol}`;
  
  try {
    console.log(`🚀 Scanning breakout signals for ${symbol}...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000,
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        TWELVEDATA_API_KEY: process.env.TWELVEDATA_API_KEY || '5e7a5daaf41d46a8966963106ebef210',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Breakout detector stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Breakout detector error:', error);
    return {
      error: `Breakout scan failed: ${error.message}`,
      symbol,
    };
  }
}


/**
 * Market-Wide TTM Squeeze Scanner - Scans entire market for squeeze setups
 */
export async function scanMarketTTMSqueeze(params: { maxStocks?: number }): Promise<any> {
  const { maxStocks = 100 } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} market_ttm_squeeze ${maxStocks}`;
  
  try {
    console.log(`🔴 Scanning entire market for TTM Squeeze setups (${maxStocks} stocks)...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 50 * 1024 * 1024, // 50MB buffer for large results
      timeout: 600000, // 10 minute timeout for market-wide scan
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        TWELVEDATA_API_KEY: process.env.TWELVEDATA_API_KEY || '5e7a5daaf41d46a8966963106ebef210',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Market TTM Squeeze scanner stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Market TTM Squeeze scanner error:', error);
    return {
      error: `Market TTM Squeeze scan failed: ${error.message}`,
      status: 'error',
    };
  }
}

/**
 * Market-Wide Breakout Scanner - Scans entire market for breakout setups
 */
export async function scanMarketBreakout(params: { maxStocks?: number }): Promise<any> {
  const { maxStocks = 100 } = params;
  const command = `${PYTHON_BIN} ${SCANNER_SCRIPT} market_breakout ${maxStocks}`;
  
  try {
    console.log(`🚀 Scanning entire market for Breakout setups (${maxStocks} stocks)...`);
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 50 * 1024 * 1024, // 50MB buffer for large results
      timeout: 600000, // 10 minute timeout for market-wide scan
      cwd: PYTHON_SYSTEM_PATH,
      env: {
        ...process.env,
        PYTHONPATH: '',
        PYTHONHOME: '',
        LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
        TWELVEDATA_API_KEY: process.env.TWELVEDATA_API_KEY || '5e7a5daaf41d46a8966963106ebef210',
      },
    });
    
    if (stderr && !stderr.includes('INFO') && !stderr.includes('WARNING')) {
      console.error('Market Breakout scanner stderr:', stderr);
    }
    
    return JSON.parse(stdout);
  } catch (error: any) {
    console.error('Market Breakout scanner error:', error);
    return {
      error: `Market Breakout scan failed: ${error.message}`,
      status: 'error',
    };
  }
}
