# Railway Environment Variables for Sadie AI v3.0

## Required Environment Variables

Configure these in Railway's Variables tab for the Sadie AI chatbot to function properly.

### Primary LLM Stack

| Variable | Description | Required | How to Get |
|----------|-------------|----------|------------|
| `GEMINI_API_KEY` | Google Gemini 2.5 Pro API key (primary LLM) | **YES** | [Google AI Studio](https://aistudio.google.com/apikey) |
| `PERPLEXITY_API_KEY` | Perplexity Sonar Pro for real-time research | **YES** | [Perplexity API](https://docs.perplexity.ai/) |
| `OPENROUTER_API_KEY` | OpenRouter for fallback models | **YES** | [OpenRouter](https://openrouter.ai/keys) |
| `FIRECRAWL_API_KEY` | Firecrawl web scraper for live options/news | **YES** | [Firecrawl](https://firecrawl.dev/) |

### Financial Data APIs

| Variable | Description | Required | How to Get |
|----------|-------------|----------|------------|
| `KEY` (or `FINNHUB_API_KEY`) | Finnhub API for real-time quotes | **YES** | [Finnhub](https://finnhub.io/) |
| `POLYGON_API_KEY` | Polygon.io for market data | Recommended | [Polygon](https://polygon.io/) |
| `TWELVEDATA_API_KEY` | TwelveData for technical indicators | Recommended | [TwelveData](https://twelvedata.com/) |
| `FINANCIAL_DATASETS_API_KEY` | FinancialDatasets.ai premium data | Recommended | [FinancialDatasets.ai](https://financialdatasets.ai/) |

### Database (if using ML features)

| Variable | Description | Required | How to Get |
|----------|-------------|----------|------------|
| `DATABASE_URL` | Full database connection string | For ML | Railway MySQL addon |
| `DB_HOST` | Database host | For ML | Railway MySQL addon |
| `DB_PORT` | Database port (default: 3306) | For ML | Railway MySQL addon |
| `DB_USER` | Database username | For ML | Railway MySQL addon |
| `DB_PASSWORD` | Database password | For ML | Railway MySQL addon |
| `DB_NAME` | Database name | For ML | Railway MySQL addon |

## Example Railway Configuration

```bash
# Primary LLM Stack (REQUIRED)
GEMINI_API_KEY=your_gemini_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Financial Data APIs (REQUIRED)
KEY=your_finnhub_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
TWELVEDATA_API_KEY=your_twelvedata_api_key_here
FINANCIAL_DATASETS_API_KEY=your_financial_datasets_api_key_here
```

## Model Configuration

Sadie AI v3.0 uses the following model hierarchy:

1. **Primary**: Google Gemini 2.5 Pro (`google/gemini-2.5-pro-preview-06-05`)
   - Main reasoning engine for all queries
   - Multimodal vision for chart analysis
   
2. **Research**: Perplexity Sonar Pro (`sonar-pro`)
   - Real-time web research
   - Grounded facts with citations
   
3. **Live Data**: Firecrawl
   - Options chains from Yahoo Finance
   - Analyst ratings and price targets
   - Breaking news and catalysts
   
4. **Fallbacks**: OpenRouter
   - GPT-4o, Claude 3.5 Sonnet
   - Used when primary models are unavailable

## Data Injection Pipeline

All responses include verified, timestamped data from:
- Finnhub (real-time quotes, company profiles)
- Polygon.io (historical data, options flow)
- TwelveData (technical indicators)
- FinancialDatasets.ai (financials, SEC filings)
- Firecrawl (live web scraping)
- yfinance (backup quotes, options chains)

**Zero hallucinations guaranteed** - LLM only uses injected data.

## Troubleshooting

If Sadie returns errors:
1. Check all required API keys are set in Railway Variables
2. Verify API keys are valid and have sufficient credits
3. Check Railway logs for specific error messages
4. Ensure `KEY` is set (Finnhub uses this variable name)

## Version History

- **v3.0** (Current): Gemini 2.5 Pro + Firecrawl + Perplexity stack
- **v2.0**: GPT-4o + Perplexity stack
- **v1.0**: OpenAI GPT-4 only
