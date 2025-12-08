import { HelpCircle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { getIndicatorTooltip } from "@/lib/indicatorTooltips";

interface IndicatorTooltipProps {
  indicatorKey: string;
  value?: number | string;
  className?: string;
}

/**
 * Reusable tooltip component for indicators
 * Shows indicator meaning, current value, and normal ranges
 */
export function IndicatorTooltip({ indicatorKey, value, className = "" }: IndicatorTooltipProps) {
  const tooltip = getIndicatorTooltip(indicatorKey);
  
  if (!tooltip) return null;
  
  // Format value for display
  const formattedValue = value !== undefined 
    ? (typeof value === 'number' ? value.toFixed(2) : value)
    : 'N/A';
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <HelpCircle 
            className={`inline-block h-4 w-4 text-muted-foreground hover:text-foreground cursor-help transition-colors ${className}`}
          />
        </TooltipTrigger>
        <TooltipContent className="max-w-sm p-4 space-y-2">
          <div>
            <p className="font-semibold text-sm">{tooltip.name}</p>
            <p className="text-xs text-muted-foreground mt-1">{tooltip.description}</p>
          </div>
          
          {value !== undefined && (
            <div className="pt-2 border-t">
              <p className="text-xs">
                <span className="font-medium">Current Value:</span> {formattedValue}
              </p>
            </div>
          )}
          
          <div className="pt-2 border-t space-y-1">
            <p className="text-xs">
              <span className="font-medium">Interpretation:</span> {tooltip.interpretation}
            </p>
            <p className="text-xs text-green-600 dark:text-green-400">
              <span className="font-medium">Good:</span> {tooltip.goodRange}
            </p>
            <p className="text-xs text-red-600 dark:text-red-400">
              <span className="font-medium">Bad:</span> {tooltip.badRange}
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
