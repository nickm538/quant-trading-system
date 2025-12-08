import { eq, lt } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, users, analysisCache, InsertAnalysisCache } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// Cache helper functions

/**
 * Get cached analysis result if not expired
 */
export async function getCachedAnalysis(symbol: string): Promise<any | null> {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db
      .select()
      .from(analysisCache)
      .where(eq(analysisCache.stockSymbol, symbol))
      .limit(1);

    if (result.length === 0) return null;

    const cached = result[0];
    const now = new Date();

    // Check if expired
    if (cached.expiresAt < now) {
      // Delete expired cache
      await db.delete(analysisCache).where(eq(analysisCache.stockSymbol, symbol));
      return null;
    }

    // Update hit count and last accessed
    await db
      .update(analysisCache)
      .set({
        hitCount: cached.hitCount + 1,
        lastAccessedAt: now,
      })
      .where(eq(analysisCache.stockSymbol, symbol));

    return JSON.parse(cached.analysisData);
  } catch (error) {
    console.error("[Cache] Failed to get cached analysis:", error);
    return null;
  }
}

/**
 * Store analysis result in cache
 * @param symbol Stock symbol
 * @param data Analysis result
 * @param ttlMinutes Time to live in minutes (default: 10)
 */
export async function setCachedAnalysis(
  symbol: string,
  data: any,
  ttlMinutes: number = 10
): Promise<void> {
  const db = await getDb();
  if (!db) return;

  try {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + ttlMinutes * 60 * 1000);

    const cacheData: InsertAnalysisCache = {
      stockSymbol: symbol,
      analysisData: JSON.stringify(data),
      cachedAt: now,
      expiresAt: expiresAt,
      hitCount: 0,
      lastAccessedAt: now,
    };

    await db
      .insert(analysisCache)
      .values(cacheData)
      .onDuplicateKeyUpdate({
        set: {
          analysisData: JSON.stringify(data),
          cachedAt: now,
          expiresAt: expiresAt,
          hitCount: 0,
          lastAccessedAt: now,
        },
      });
  } catch (error) {
    console.error("[Cache] Failed to set cached analysis:", error);
  }
}

/**
 * Clear expired cache entries
 */
export async function clearExpiredCache(): Promise<void> {
  const db = await getDb();
  if (!db) return;

  try {
    const now = new Date();
    await db.delete(analysisCache).where(lt(analysisCache.expiresAt, now));
  } catch (error) {
    console.error("[Cache] Failed to clear expired cache:", error);
  }
}
