CREATE TABLE `analysis_cache` (
	`id` int AUTO_INCREMENT NOT NULL,
	`stock_symbol` varchar(10) NOT NULL,
	`analysis_data` text NOT NULL,
	`cached_at` timestamp NOT NULL DEFAULT (now()),
	`expires_at` timestamp NOT NULL,
	`hit_count` int NOT NULL DEFAULT 0,
	`last_accessed_at` timestamp NOT NULL DEFAULT (now()),
	`created_at` timestamp NOT NULL DEFAULT (now()),
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `analysis_cache_id` PRIMARY KEY(`id`),
	CONSTRAINT `analysis_cache_stock_symbol_unique` UNIQUE(`stock_symbol`)
);
