import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Lightbulb, 
  Target,
  Users,
  History,
  Brain
} from "lucide-react";
import { Streamdown } from "streamdown";

interface ExpertReasoningProps {
  analysis: any;
}

export function ExpertReasoningDisplay({ analysis }: ExpertReasoningProps) {
  const reasoning = analysis?.expert_reasoning;
  const patternRecognition = analysis?.pattern_recognition;
  
  // Debug logging
  console.log('[ExpertReasoningDisplay] Analysis:', !!analysis);
  console.log('[ExpertReasoningDisplay] Reasoning:', !!reasoning);
  console.log('[ExpertReasoningDisplay] Legendary traders:', !!reasoning?.legendary_trader_perspectives);
  if (reasoning?.legendary_trader_perspectives) {
    console.log('[ExpertReasoningDisplay] Trader keys:', Object.keys(reasoning.legendary_trader_perspectives));
    console.log('[ExpertReasoningDisplay] Trader count:', Object.keys(reasoning.legendary_trader_perspectives).length);
  }
  
  if (!reasoning) {
    return null;
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 70) return "text-green-600";
    if (confidence > 40) return "text-yellow-600";
    return "text-red-600";
  };

  const getSignalColor = (signal: string) => {
    if (signal === "BUY" || signal === "STRONG_BUY") return "bg-green-600";
    if (signal === "SELL" || signal === "STRONG_SELL") return "bg-red-600";
    return "bg-gray-600";
  };

  return (
    <div className="space-y-6">
      {/* Primary Thesis */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <CardTitle>Expert Analysis</CardTitle>
            </div>
            <Badge className={getSignalColor(analysis.signal)}>
              {analysis.signal}
            </Badge>
          </div>
          <CardDescription>
            Confidence: <span className={getConfidenceColor(analysis.confidence)}>
              {analysis.confidence.toFixed(1)}%
            </span>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <Streamdown>{reasoning.primary_thesis}</Streamdown>
          </div>
        </CardContent>
      </Card>

      {/* Tabbed Detailed Analysis */}
      <Tabs defaultValue="factors" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="factors">
            <Target className="h-4 w-4 mr-2" />
            Factors
          </TabsTrigger>
          <TabsTrigger value="execution">
            <Lightbulb className="h-4 w-4 mr-2" />
            Execution
          </TabsTrigger>
          <TabsTrigger value="traders">
            <Users className="h-4 w-4 mr-2" />
            Legends
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Supporting & Risk Factors */}
        <TabsContent value="factors" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <CardTitle className="text-lg">Supporting Factors</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {reasoning.supporting_factors?.map((factor: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-green-600 mt-1">✓</span>
                    <span className="text-sm">{factor}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
                <CardTitle className="text-lg">Risk Factors</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {reasoning.risk_factors?.map((risk: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-yellow-600 mt-1">⚠</span>
                    <span className="text-sm">{risk}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          {reasoning.catalysts && reasoning.catalysts.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  <CardTitle className="text-lg">Potential Catalysts</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {reasoning.catalysts.map((catalyst: string, idx: number) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">→</span>
                      <span className="text-sm">{catalyst}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Execution Strategy */}
        <TabsContent value="execution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Market Regime</CardTitle>
            </CardHeader>
            <CardContent>
              <Alert>
                <AlertDescription>
                  <Streamdown>{reasoning.market_regime}</Streamdown>
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Execution Strategy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <Streamdown>{reasoning.execution_strategy}</Streamdown>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Confidence Explanation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <Streamdown>{reasoning.confidence_explanation}</Streamdown>
              </div>
            </CardContent>
          </Card>

          {reasoning.alternative_scenarios && (
            <Card>
              <CardHeader>
                <CardTitle>Alternative Scenarios</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Bull Case */}
                  <div className="border-l-4 border-green-600 pl-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-green-600">Bull Case</h4>
                      <Badge variant="outline" className="text-green-600">
                        {reasoning.alternative_scenarios.bull_case.probability}% probability
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Target: ${reasoning.alternative_scenarios.bull_case.target.toFixed(2)} 
                      ({reasoning.alternative_scenarios.bull_case.return_pct > 0 ? '+' : ''}
                      {reasoning.alternative_scenarios.bull_case.return_pct.toFixed(1)}%)
                    </p>
                    <p className="text-sm">{reasoning.alternative_scenarios.bull_case.description}</p>
                  </div>

                  {/* Base Case */}
                  <div className="border-l-4 border-blue-600 pl-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-blue-600">Base Case</h4>
                      <Badge variant="outline" className="text-blue-600">
                        {reasoning.alternative_scenarios.base_case.probability}% probability
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Target: ${reasoning.alternative_scenarios.base_case.target.toFixed(2)} 
                      ({reasoning.alternative_scenarios.base_case.return_pct > 0 ? '+' : ''}
                      {reasoning.alternative_scenarios.base_case.return_pct.toFixed(1)}%)
                    </p>
                    <p className="text-sm">{reasoning.alternative_scenarios.base_case.description}</p>
                  </div>

                  {/* Bear Case */}
                  <div className="border-l-4 border-red-600 pl-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-red-600">Bear Case</h4>
                      <Badge variant="outline" className="text-red-600">
                        {reasoning.alternative_scenarios.bear_case.probability}% probability
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Target: ${reasoning.alternative_scenarios.bear_case.target.toFixed(2)} 
                      ({reasoning.alternative_scenarios.bear_case.return_pct > 0 ? '+' : ''}
                      {reasoning.alternative_scenarios.bear_case.return_pct.toFixed(1)}%)
                    </p>
                    <p className="text-sm">{reasoning.alternative_scenarios.bear_case.description}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Legendary Traders */}
        <TabsContent value="traders" className="space-y-4">
          {reasoning.trader_consensus && (
            <Alert>
              <Users className="h-4 w-4" />
              <AlertDescription>
                <Streamdown>{reasoning.trader_consensus}</Streamdown>
              </AlertDescription>
            </Alert>
          )}

          {reasoning.legendary_trader_perspectives && (
            <div className="grid gap-4">
              {Object.entries(reasoning.legendary_trader_perspectives).map(([trader, advice]: [string, any]) => {
                const traderNames: Record<string, string> = {
                  warren_buffett: "Warren Buffett",
                  george_soros: "George Soros",
                  stanley_druckenmiller: "Stanley Druckenmiller",
                  peter_lynch: "Peter Lynch",
                  paul_tudor_jones: "Paul Tudor Jones",
                  jesse_livermore: "Jesse Livermore"
                };

                return (
                  <Card key={trader}>
                    <CardHeader>
                      <CardTitle className="text-lg">{traderNames[trader]}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="prose prose-sm max-w-none dark:prose-invert">
                        <Streamdown>{advice}</Streamdown>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>

        {/* Historical Patterns */}
        <TabsContent value="history" className="space-y-4">
          {patternRecognition?.historical_analogy && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <History className="h-5 w-5 text-primary" />
                  <CardTitle>Historical Analogy</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <Streamdown>{patternRecognition.historical_analogy}</Streamdown>
                </div>
              </CardContent>
            </Card>
          )}

          {patternRecognition?.regime_match && (
            <Card>
              <CardHeader>
                <CardTitle>Market Regime Match</CardTitle>
                <CardDescription>
                  Match Confidence: {patternRecognition.regime_match.match_score?.toFixed(0)}%
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <p><strong>Period:</strong> {patternRecognition.regime_match.period}</p>
                  <p><strong>Description:</strong> {patternRecognition.regime_match.description}</p>
                  <p><strong>Outcome:</strong> {patternRecognition.regime_match.outcome}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {patternRecognition?.pattern_prediction && (
            <Card>
              <CardHeader>
                <CardTitle>Pattern Prediction</CardTitle>
                <CardDescription>
                  Based on {patternRecognition.pattern_prediction.sample_size} similar historical patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Expected Return</p>
                    <p className="text-2xl font-bold">
                      {(patternRecognition.pattern_prediction.expected_return * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Probability Up</p>
                    <p className="text-2xl font-bold text-green-600">
                      {(patternRecognition.pattern_prediction.probability_up * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Days to Outcome</p>
                    <p className="text-2xl font-bold">
                      {patternRecognition.pattern_prediction.avg_days_to_outcome}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Confidence</p>
                    <p className="text-2xl font-bold">
                      {patternRecognition.confidence?.toFixed(0)}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
