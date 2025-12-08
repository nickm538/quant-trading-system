CREATE TABLE `model_predictions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`model_id` int NOT NULL,
	`stock_symbol` varchar(10) NOT NULL,
	`prediction_date` timestamp NOT NULL,
	`target_date` timestamp NOT NULL,
	`predicted_price` int NOT NULL,
	`predicted_low` int NOT NULL,
	`predicted_high` int NOT NULL,
	`confidence` int NOT NULL,
	`actual_price` int,
	`actual_low` int,
	`actual_high` int,
	`price_error` int,
	`percentage_error` int,
	`status` enum('pending','validated','failed') NOT NULL DEFAULT 'pending',
	`created_at` timestamp NOT NULL DEFAULT (now()),
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `model_predictions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `retraining_history` (
	`id` int AUTO_INCREMENT NOT NULL,
	`old_model_id` int,
	`new_model_id` int NOT NULL,
	`stock_symbol` varchar(10) NOT NULL,
	`trigger_reason` enum('scheduled','accuracy_degradation','regime_change','manual','new_data_available') NOT NULL,
	`old_accuracy` int,
	`new_accuracy` int NOT NULL,
	`improvement_pct` int,
	`retrained_at` timestamp NOT NULL DEFAULT (now()),
	`notes` text,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `retraining_history_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `trained_models` (
	`id` int AUTO_INCREMENT NOT NULL,
	`stock_symbol` varchar(10) NOT NULL,
	`model_type` enum('xgboost','lightgbm','lstm','ensemble') NOT NULL,
	`version` varchar(50) NOT NULL,
	`model_path` text NOT NULL,
	`trained_at` timestamp NOT NULL DEFAULT (now()),
	`training_start_date` timestamp NOT NULL,
	`training_end_date` timestamp NOT NULL,
	`training_data_points` int NOT NULL,
	`training_accuracy` int NOT NULL,
	`validation_accuracy` int NOT NULL,
	`test_accuracy` int NOT NULL,
	`mse` int NOT NULL,
	`mae` int NOT NULL,
	`r2_score` int NOT NULL,
	`hyperparameters` text NOT NULL,
	`feature_importance` text,
	`is_active` enum('active','inactive','deprecated') NOT NULL DEFAULT 'active',
	`created_at` timestamp NOT NULL DEFAULT (now()),
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `trained_models_id` PRIMARY KEY(`id`)
);
