/**
 * Database helpers for ML model storage and retrieval
 */

import { eq, desc } from "drizzle-orm";
import { getDb } from "./db";
import { trainedModels, modelPredictions, retrainingHistory } from "../drizzle/schema";

/**
 * Store a trained model in the database
 */
export async function storeTrainedModel(model: {
  stockSymbol: string;
  modelType: "xgboost" | "lightgbm" | "lstm" | "ensemble";
  version: string;
  modelPath: string;
  trainingAccuracy: number; // 0-1 scale (e.g., 0.8532)
  validationAccuracy: number;
  testAccuracy: number;
  mse: number;
  mae: number;
  r2Score: number;
  hyperparameters: Record<string, any>;
  featureImportance: Record<string, number>;
  trainingStartDate: Date;
  trainingEndDate: Date;
  trainingDataPoints: number;
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  const [inserted] = await db.insert(trainedModels).values({
    stockSymbol: model.stockSymbol,
    modelType: model.modelType,
    version: model.version,
    modelPath: model.modelPath,
    // Convert percentages to integers (e.g., 0.8532 -> 8532)
    trainingAccuracy: Math.round(model.trainingAccuracy * 10000),
    validationAccuracy: Math.round(model.validationAccuracy * 10000),
    testAccuracy: Math.round(model.testAccuracy * 10000),
    mse: Math.round(model.mse * 10000),
    mae: Math.round(model.mae * 10000),
    r2Score: Math.round(model.r2Score * 10000),
    hyperparameters: JSON.stringify(model.hyperparameters),
    featureImportance: JSON.stringify(model.featureImportance),
    trainingStartDate: model.trainingStartDate,
    trainingEndDate: model.trainingEndDate,
    trainingDataPoints: model.trainingDataPoints,
    isActive: "active",
  });

  return inserted;
}

/**
 * Get all trained models for a symbol
 */
export async function getTrainedModels(stockSymbol: string) {
  const db = await getDb();
  if (!db) return [];

  return await db
    .select()
    .from(trainedModels)
    .where(eq(trainedModels.stockSymbol, stockSymbol))
    .orderBy(desc(trainedModels.trainedAt));
}

/**
 * Get the latest active model for a symbol
 */
export async function getLatestModel(stockSymbol: string, modelType?: string) {
  const db = await getDb();
  if (!db) return null;

  const models = await db
    .select()
    .from(trainedModels)
    .where(eq(trainedModels.stockSymbol, stockSymbol))
    .orderBy(desc(trainedModels.trainedAt))
    .limit(1);

  return models[0] || null;
}

/**
 * Store a model prediction
 */
export async function storePrediction(prediction: {
  modelId: number;
  stockSymbol: string;
  targetDate: Date;
  predictedPrice: number; // Dollar amount (e.g., 150.50)
  predictedLow: number;
  predictedHigh: number;
  confidence: number; // 0-1 scale (e.g., 0.85)
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  const [inserted] = await db.insert(modelPredictions).values({
    modelId: prediction.modelId,
    stockSymbol: prediction.stockSymbol,
    predictionDate: new Date(),
    targetDate: prediction.targetDate,
    // Convert dollars to cents (e.g., 150.50 -> 15050)
    predictedPrice: Math.round(prediction.predictedPrice * 100),
    predictedLow: Math.round(prediction.predictedLow * 100),
    predictedHigh: Math.round(prediction.predictedHigh * 100),
    confidence: Math.round(prediction.confidence * 100),
    status: "pending",
  });

  return inserted;
}

/**
 * Log a retraining event
 */
export async function logRetraining(event: {
  stockSymbol: string;
  oldModelId: number | null;
  newModelId: number;
  triggerReason: "scheduled" | "accuracy_degradation" | "regime_change" | "manual" | "new_data_available";
  oldAccuracy: number | null; // 0-1 scale
  newAccuracy: number; // 0-1 scale
  improvementPct: number | null; // 0-1 scale (e.g., 0.05 = 5% improvement)
}) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");

  await db.insert(retrainingHistory).values({
    stockSymbol: event.stockSymbol,
    oldModelId: event.oldModelId,
    newModelId: event.newModelId,
    triggerReason: event.triggerReason,
    // Convert percentages to integers
    oldAccuracy: event.oldAccuracy ? Math.round(event.oldAccuracy * 10000) : null,
    newAccuracy: Math.round(event.newAccuracy * 10000),
    improvementPct: event.improvementPct ? Math.round(event.improvementPct * 10000) : null,
  });
}

/**
 * Get all models summary
 */
export async function getAllModelsSummary() {
  const db = await getDb();
  if (!db) return [];

  const models = await db
    .select({
      id: trainedModels.id,
      stockSymbol: trainedModels.stockSymbol,
      modelType: trainedModels.modelType,
      testAccuracy: trainedModels.testAccuracy,
      trainedAt: trainedModels.trainedAt,
      isActive: trainedModels.isActive,
    })
    .from(trainedModels)
    .orderBy(desc(trainedModels.trainedAt));

  // Convert integers back to percentages
  return models.map(m => ({
    ...m,
    testAccuracy: m.testAccuracy / 10000, // Convert back to 0-1 scale
  }));
}
