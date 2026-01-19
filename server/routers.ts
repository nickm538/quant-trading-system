import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router, protectedProcedure } from "./_core/trpc";
import { z } from "zod";
import { analyzeStock, analyzeOptions, scanMarket, checkPythonSystem, getGreeksHeatmap, analyzeInstitutionalOptions, scanUltimateOptions, analyzeUltimateOptions, analyzeFundamentals, getEducation, sadieChat, sadieChatWithImage, sadieAnalyze, sadieClearHistory, scanDarkPool, scanTTMSqueeze, scanOptionsFlow, scanBreakout, scanMarketTTMSqueeze, scanMarketBreakout, getMarketIntelligence, scanMarketDarkPool } from "./python_executor";
import { exec } from "child_process";
import { promisify } from "util";
import * as path from "path";
import { getAllModelsSummary } from "./db_ml";
import { getCachedAnalysis, setCachedAnalysis } from "./db";

const execAsync = promisify(exec);

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  trading: router({
    // Stock analysis endpoint with caching
    analyzeStock: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
          monte_carlo_sims: z.number().int().min(1000).max(50000).optional().default(20000),
          forecast_days: z.number().int().min(1).max(365).optional().default(30),
          bankroll: z.number().positive().optional().default(1000),
        })
      )
      .mutation(async ({ input }) => {
        const cacheKey = input.symbol.toUpperCase();
        
        // Try to get from cache first
        const cached = await getCachedAnalysis(cacheKey);
        if (cached) {
          console.log(`[Cache HIT] Returning cached analysis for ${cacheKey}`);
          return { ...cached, fromCache: true };
        }
        
        console.log(`[Cache MISS] Fetching fresh analysis for ${cacheKey}`);
        
        // Cache miss - fetch fresh data
        const result = await analyzeStock(input);
        
        // Store in cache (3 minute TTL for real-time trading)
        await setCachedAnalysis(cacheKey, result, 3);
        
        return { ...result, fromCache: false };
      }),
    
    // Options analysis endpoint (legacy)
    analyzeOptions: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
          min_delta: z.number().min(0).max(1).optional().default(0.3),
          max_delta: z.number().min(0).max(1).optional().default(0.6),
          min_days: z.number().int().min(1).optional().default(7),
        })
      )
      .mutation(async ({ input }) => {
        return await analyzeOptions(input);
      }),
    
    // Institutional-grade options analysis endpoint (NEW)
    analyzeInstitutionalOptions: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        return await analyzeInstitutionalOptions(input);
      }),
    
    // Greeks heatmap endpoint
    getGreeksHeatmap: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
          num_strikes: z.number().int().min(5).max(30).optional().default(15),
          num_expirations: z.number().int().min(2).max(12).optional().default(6),
        })
      )
      .mutation(async ({ input }) => {
        return await getGreeksHeatmap(input);
      }),
    
    // Market scanner endpoint
    scanMarket: publicProcedure
      .input(
        z.object({
          top_n: z.number().int().min(5).max(50).optional().default(20),
        })
      )
      .mutation(async ({ input }) => {
        return await scanMarket(input);
      }),
    
    // Ultimate Options Scanner - Market-wide scan for best opportunities
    scanUltimateOptions: publicProcedure
      .input(
        z.object({
          max_results: z.number().int().min(5).max(20).optional().default(10),
          option_type: z.enum(['call', 'put', 'both']).optional().default('both'),
        })
      )
      .mutation(async ({ input }) => {
        return await scanUltimateOptions(input);
      }),
    
    // Ultimate Options Analyzer - Deep analysis of a single symbol
    analyzeUltimateOptions: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
          option_type: z.enum(['call', 'put', 'both']).optional().default('both'),
        })
      )
      .mutation(async ({ input }) => {
        return await analyzeUltimateOptions(input);
      }),
    
    // Legacy Options scanner endpoint (redirects to Ultimate)
    scanOptions: publicProcedure
      .input(
        z.object({
          max_results: z.number().int().min(5).max(20).optional().default(10),
        })
      )
      .mutation(async ({ input }) => {
        // Redirect to Ultimate Options Scanner
        return await scanUltimateOptions({ max_results: input.max_results, option_type: 'both' });
      }),
    
    healthCheck: publicProcedure.query(async () => {
      const isHealthy = await checkPythonSystem();
      return { healthy: isHealthy, timestamp: new Date().toISOString() };
    }),
    
    // Get ML prediction for a stock
    getMLPrediction: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
          horizon_days: z.number().int().min(1).max(90).optional().default(30),
        })
      )
      .mutation(async ({ input }) => {
        console.log(`\n🧠 ML PREDICTION REQUEST: ${input.symbol} (${input.horizon_days} days)`);
        try {
          const pythonPath = 'python3.11';
          const scriptPath = path.join(process.cwd(), 'python_system/ml/prediction_engine.py');
          
          // Construct DATABASE_URL
          let databaseUrl = process.env.DATABASE_URL;
          if (!databaseUrl && process.env.MYSQLHOST) {
            const host = process.env.MYSQLHOST;
            const port = process.env.MYSQLPORT || '3306';
            const user = process.env.MYSQLUSER || 'root';
            const password = process.env.MYSQLPASSWORD || '';
            const database = process.env.MYSQLDATABASE || 'railway';
            databaseUrl = `mysql://${user}:${password}@${host}:${port}/${database}`;
          }
          
          console.log(`🐍 Executing Python: ${pythonPath} ${scriptPath} ${input.symbol.toUpperCase()} ${input.horizon_days}`);
          console.log(`📊 DATABASE_URL set: ${databaseUrl ? 'YES' : 'NO'}`);
          
          const { stdout, stderr } = await execAsync(
            `${pythonPath} ${scriptPath} ${input.symbol.toUpperCase()} ${input.horizon_days}`,
            { 
              maxBuffer: 10 * 1024 * 1024, 
              timeout: 60000, // 1min timeout
              env: {
                ...process.env,
                DATABASE_URL: databaseUrl || '',
                PYTHONPATH: '',
                PYTHONHOME: '',
                LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
              },
            }
          );
          
          if (stderr) {
            console.log(`⚠️  Python stderr: ${stderr}`);
          }
          
          console.log(`✅ Python stdout (first 500 chars): ${stdout.substring(0, 500)}`);
          
          // Parse JSON output from Python
          const result = JSON.parse(stdout);
          console.log(`📊 Result success: ${result.success}`);
          if (!result.success) {
            console.log(`❌ Result error: ${result.error}`);
          }
          return result;
        } catch (error: any) {
          console.error('❌ ML Prediction error:', error);
          console.error('❌ Error stack:', error.stack);
          return {
            success: false,
            error: error.message,
            symbol: input.symbol,
          };
        }
      }),
  }),

  ml: router({
    // Get all trained models summary
    getModels: publicProcedure.query(async () => {
      return await getAllModelsSummary();
    }),
    
    // Train models on the 15 selected stocks
    trainModels: publicProcedure.mutation(async () => {
      try {
        const pythonPath = 'python3.11';
        const scriptPath = path.join(process.cwd(), 'python_system/ml/backtest_and_train.py');
        
        // Construct DATABASE_URL from Railway's MySQL variables if not already set
        let databaseUrl = process.env.DATABASE_URL;
        if (!databaseUrl && process.env.MYSQLHOST) {
          const host = process.env.MYSQLHOST;
          const port = process.env.MYSQLPORT || '3306';
          const user = process.env.MYSQLUSER || 'root';
          const password = process.env.MYSQLPASSWORD || '';
          const database = process.env.MYSQLDATABASE || 'railway';
          databaseUrl = `mysql://${user}:${password}@${host}:${port}/${database}`;
        }
        
        const { stdout, stderr } = await execAsync(
          `${pythonPath} ${scriptPath}`,
          { 
            maxBuffer: 10 * 1024 * 1024, 
            timeout: 300000, // 5min timeout
            env: {
              ...process.env,
              DATABASE_URL: databaseUrl || '',
              PYTHONPATH: '',
              PYTHONHOME: '',
              LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
            },
          }
        );
        
        return {
          success: true,
          output: stdout,
          errors: stderr || null,
        };
      } catch (error: any) {
        return {
          success: false,
          output: error.stdout || '',
          errors: error.stderr || error.message,
        };
      }
    }),
    
    // Validate pending predictions (fetch actual prices and calculate errors)
    validatePredictions: publicProcedure.mutation(async () => {
      try {
        const pythonPath = 'python3.11';
        const scriptPath = path.join(process.cwd(), 'python_system/ml/continuous_learning_scheduler.py');
        
        // Construct DATABASE_URL from Railway's MySQL variables if not already set
        let databaseUrl = process.env.DATABASE_URL;
        if (!databaseUrl && process.env.MYSQLHOST) {
          const host = process.env.MYSQLHOST;
          const port = process.env.MYSQLPORT || '3306';
          const user = process.env.MYSQLUSER || 'root';
          const password = process.env.MYSQLPASSWORD || '';
          const database = process.env.MYSQLDATABASE || 'railway';
          databaseUrl = `mysql://${user}:${password}@${host}:${port}/${database}`;
        }
        
        const { stdout, stderr } = await execAsync(
          `${pythonPath} ${scriptPath} validate`,
          { 
            maxBuffer: 10 * 1024 * 1024, 
            timeout: 120000, // 2min timeout
            env: {
              ...process.env,
              DATABASE_URL: databaseUrl || '',
              PYTHONPATH: '',
              PYTHONHOME: '',
              LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH || '',
            },
          }
        );
        
        return {
          success: true,
          output: stdout,
          errors: stderr || null,
        };
      } catch (error: any) {
        return {
          success: false,
          output: error.stdout || '',
          errors: error.stderr || error.message,
        };
      }
    }),
    
    // Comprehensive fundamentals analysis with educational content
    analyzeFundamentals: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        return await analyzeFundamentals({ symbol: input.symbol.toUpperCase() });
      }),
    
    // Get trading education content
    getEducation: publicProcedure
      .input(
        z.object({
          topic: z.string().optional(),
        })
      )
      .query(async ({ input }) => {
        return await getEducation({ topic: input.topic });
      }),
    
    // Sadie AI Chatbot - The Ultimate Financial Intelligence Assistant
    sadieChat: publicProcedure
      .input(
        z.object({
          message: z.string().min(1).max(10000),
        })
      )
      .mutation(async ({ input }) => {
        console.log('🐿️ Sadie AI chat request:', input.message.substring(0, 100) + '...');
        return await sadieChat({ message: input.message });
      }),
    
    // Sadie AI Chat with Image - Vision-enabled analysis
    sadieChatWithImage: publicProcedure
      .input(
        z.object({
          message: z.string().min(1).max(10000),
          images: z.array(z.string()).min(1).max(5),  // Up to 5 base64 images
        })
      )
      .mutation(async ({ input }) => {
        console.log('🐿️📷 Sadie AI Vision request:', input.message.substring(0, 100) + '...', `(${input.images.length} images)`);
        return await sadieChatWithImage({ message: input.message, images: input.images });
      }),
    
    // Sadie AI Quick Analysis
    sadieAnalyze: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        console.log('🐿️ Sadie AI analyzing:', input.symbol);
        return await sadieAnalyze({ symbol: input.symbol.toUpperCase() });
      }),
    
    // Clear Sadie conversation history
    sadieClearHistory: publicProcedure
      .mutation(async () => {
        return await sadieClearHistory();
      }),
  }),

  // ============================================================================
  // NEW SCANNER MODULES (from financial-analysis-system)
  // ============================================================================
  scanners: router({
    // Dark Pool Scanner - Insider Movement/Oracle Analysis
    darkPool: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        console.log('🔮 Dark Pool scan for:', input.symbol);
        return await scanDarkPool({ symbol: input.symbol.toUpperCase() });
      }),

    // TTM Squeeze Scanner - Volatility Compression Detection
    ttmSqueeze: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        console.log('🔴 TTM Squeeze scan for:', input.symbol);
        return await scanTTMSqueeze({ symbol: input.symbol.toUpperCase() });
      }),

    // Options Flow Scanner - Bear to Bull Pressure Analysis
    optionsFlow: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        console.log('📊 Options Flow scan for:', input.symbol);
        return await scanOptionsFlow({ symbol: input.symbol.toUpperCase() });
      }),

    // Breakout Detector - Multi-Signal Breakout Analysis
    breakout: publicProcedure
      .input(
        z.object({
          symbol: z.string().min(1).max(10),
        })
      )
      .mutation(async ({ input }) => {
        console.log('🚀 Breakout scan for:', input.symbol);
        return await scanBreakout({ symbol: input.symbol.toUpperCase() });
      }),

    // Market-Wide TTM Squeeze Scanner - Scans entire market
    marketTTMSqueeze: publicProcedure
      .input(
        z.object({
          maxStocks: z.number().int().min(10).max(500).optional().default(400),
        })
      )
      .mutation(async ({ input }) => {
        console.log(`🔴 Market-wide TTM Squeeze scan (${input.maxStocks} stocks)...`);
        return await scanMarketTTMSqueeze({ maxStocks: input.maxStocks });
      }),

    // Market-Wide Breakout Scanner - Scans entire market
    marketBreakout: publicProcedure
      .input(
        z.object({
          maxStocks: z.number().int().min(10).max(500).optional().default(400),
        })
      )
      .mutation(async ({ input }) => {
        console.log(`Market-wide Breakout scan (${input.maxStocks} stocks)...`);
        return await scanMarketBreakout({ maxStocks: input.maxStocks });
      }),

    // Market Intelligence - VIX, regime, sentiment, catalysts
    marketIntelligence: publicProcedure
      .mutation(async () => {
        console.log('Fetching market intelligence...');
        return await getMarketIntelligence();
      }),

    // Market-Wide Dark Pool Scanner - finds stocks with highest dark pool activity
    marketDarkPool: publicProcedure
      .input(
        z.object({
          limit: z.number().int().min(10).max(500).optional().default(400),
        })
      )
      .mutation(async ({ input }) => {
        console.log(`Scanning market for dark pool activity (${input.limit} stocks)...`);
        return await scanMarketDarkPool({ maxStocks: input.limit });
      }),
  }),
});

export type AppRouter = typeof appRouter;
// Force rebuild Tue Jan  6 17:04:59 EST 2026
