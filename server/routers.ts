import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router, protectedProcedure } from "./_core/trpc";
import { z } from "zod";
import { analyzeStock, analyzeOptions, scanMarket, checkPythonSystem, getGreeksHeatmap } from "./python_executor";
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
    
    // Options analysis endpoint
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
    
    healthCheck: publicProcedure.query(async () => {
      const isHealthy = await checkPythonSystem();
      return { healthy: isHealthy, timestamp: new Date().toISOString() };
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
        const { stdout, stderr } = await execAsync(
          `${pythonPath} ${scriptPath}`,
          { 
            maxBuffer: 10 * 1024 * 1024, 
            timeout: 300000, // 5min timeout
            env: {
              ...process.env,
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
  }),
});

export type AppRouter = typeof appRouter;
