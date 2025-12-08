# Deployment Instructions for Quant Trading System

## Cloudflare Pages Deployment

### Prerequisites
- GitHub repository: https://github.com/nickm538/quant-trading-system
- Cloudflare account with Pages access

### Step-by-Step Deployment

#### 1. Create D1 Database
```bash
# Via Cloudflare Dashboard:
1. Go to Workers & Pages → D1
2. Click "Create database"
3. Name: quant-trading-db
4. Copy the database ID
```

#### 2. Deploy to Cloudflare Pages

**Option A: Via Dashboard (Recommended)**

1. Go to https://dash.cloudflare.com/
2. Navigate to **Workers & Pages**
3. Click **Create application** → **Pages** → **Connect to Git**
4. Select your GitHub repository: `nickm538/quant-trading-system`
5. Configure build settings:
   - **Framework preset**: None
   - **Build command**: `pnpm build`
   - **Build output directory**: `dist/public`
   - **Root directory**: `/`
6. Add environment variables:
   - `NODE_ENV` = `production`
7. Click **Save and Deploy**

**Option B: Via Wrangler CLI**

```bash
# Install dependencies
pnpm install

# Build the project
pnpm build

# Deploy to Pages
pnpm wrangler pages deploy dist/public --project-name=quant-trading-system
```

#### 3. Configure D1 Database Binding

1. Go to your Pages project settings
2. Navigate to **Settings** → **Functions** → **D1 database bindings**
3. Add binding:
   - **Variable name**: `DB`
   - **D1 database**: Select `quant-trading-db`
4. Save changes

#### 4. Initialize Database Schema

After deployment, run migrations:

```bash
# Apply schema to D1
pnpm wrangler d1 execute quant-trading-db --file=drizzle/0000_init.sql
pnpm wrangler d1 execute quant-trading-db --file=drizzle/0001_ml_training.sql
```

Or via the dashboard:
1. Go to D1 database → Console
2. Copy and paste SQL from `drizzle/schema.ts`
3. Execute

#### 5. Set Environment Variables

In Cloudflare Pages project settings, add:

```
NODE_ENV=production
DATABASE_URL=<your-d1-connection-string>
```

### Python Backend Considerations

**Important**: Cloudflare Workers/Pages don't support Python natively. You have two options:

#### Option 1: Use Cloudflare Workers with WebAssembly
- Convert Python analysis to JavaScript/TypeScript
- Use libraries like `technicalindicators` for TA
- Implement Monte Carlo in JavaScript

#### Option 2: Hybrid Architecture (Recommended)
- Deploy frontend to Cloudflare Pages
- Deploy Python backend to:
  - **Railway**: https://railway.app
  - **Render**: https://render.com
  - **Fly.io**: https://fly.io
- Connect via API calls from Pages to Python backend

### Current Architecture

```
Frontend (React + Vite)
    ↓
Node.js Backend (tRPC)
    ↓
Python Analysis System
    ↓
yfinance + TA-Lib + ML
```

### Recommended Production Architecture

```
Cloudflare Pages (Frontend)
    ↓
Cloudflare Workers (API Gateway)
    ↓
Railway/Render (Python Backend)
    ↓
D1 Database (Cloudflare)
```

## Alternative: Full Deployment to Railway

If you want everything in one place with Python support:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

Railway will automatically:
- Detect Node.js and Python
- Install dependencies
- Run migrations
- Provide a production URL

## Post-Deployment

1. Test the deployment at your Cloudflare Pages URL
2. Configure custom domain (optional)
3. Set up monitoring and analytics
4. Enable Cloudflare CDN and security features

## Troubleshooting

### Python Execution Issues
- Cloudflare Workers don't support Python
- Consider rewriting analysis in JavaScript or use hybrid architecture

### Database Connection Issues
- Ensure D1 binding is correctly configured
- Check database ID in wrangler.toml matches actual D1 database

### Build Failures
- Verify all dependencies are in package.json
- Check build logs for specific errors
- Ensure Node.js version compatibility (18+)

## Support

For issues, check:
- Cloudflare Pages docs: https://developers.cloudflare.com/pages
- Cloudflare D1 docs: https://developers.cloudflare.com/d1
- GitHub repository: https://github.com/nickm538/quant-trading-system
