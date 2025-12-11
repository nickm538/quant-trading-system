/**
 * Alerts Panel Component
 * Displays real-time alerts from TTM Squeeze monitoring system
 */

import React from 'react';
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Bell, AlertTriangle, Info, AlertCircle, TrendingUp, TrendingDown, Clock, Target, StopCircle } from "lucide-react";
import { Alert } from '@/hooks/useSqueezeStream';

interface AlertsPanelProps {
  alerts: Alert[];
  maxAlerts?: number;
}

export function AlertsPanel({ alerts, maxAlerts = 50 }: AlertsPanelProps) {
  
  // Get icon for alert type
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'SQUEEZE_FIRE':
        return <TrendingUp className="w-4 h-4" />;
      case 'SQUEEZE_ACTIVE':
        return <AlertCircle className="w-4 h-4" />;
      case 'EXIT_SIGNAL':
        return <StopCircle className="w-4 h-4" />;
      case 'PROFIT_TARGET':
        return <Target className="w-4 h-4" />;
      case 'STOP_LOSS':
        return <AlertTriangle className="w-4 h-4" />;
      case 'MOMENTUM_REVERSAL':
        return <TrendingDown className="w-4 h-4" />;
      case 'TIME_DECAY_WARNING':
        return <Clock className="w-4 h-4" />;
      default:
        return <Info className="w-4 h-4" />;
    }
  };

  // Get color for priority
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'bg-red-500 text-white border-red-600';
      case 'high':
        return 'bg-orange-500 text-white border-orange-600';
      case 'medium':
        return 'bg-yellow-500 text-white border-yellow-600';
      case 'low':
        return 'bg-blue-500 text-white border-blue-600';
      default:
        return 'bg-gray-500 text-white border-gray-600';
    }
  };

  // Get background color for alert card
  const getAlertBgColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800';
      case 'high':
        return 'bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800';
      case 'medium':
        return 'bg-yellow-50 dark:bg-yellow-950 border-yellow-200 dark:border-yellow-800';
      case 'low':
        return 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800';
      default:
        return 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-800';
    }
  };

  // Format timestamp
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  // Get relative time
  const getRelativeTime = (timestamp: string) => {
    const now = new Date().getTime();
    const then = new Date(timestamp).getTime();
    const diff = Math.floor((now - then) / 1000); // seconds

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const displayedAlerts = alerts.slice(0, maxAlerts);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="w-5 h-5" />
          Real-Time Alerts
          {alerts.length > 0 && (
            <Badge className="ml-auto">{alerts.length}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {alerts.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <Bell className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No alerts yet</p>
            <p className="text-sm">Alerts will appear here when squeeze events occur</p>
          </div>
        ) : (
          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-2">
              {displayedAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border ${getAlertBgColor(alert.priority)}`}
                >
                  <div className="flex items-start gap-3">
                    {/* Icon */}
                    <div className={`p-2 rounded-full ${getPriorityColor(alert.priority)}`}>
                      {getAlertIcon(alert.type)}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      {/* Header */}
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="font-semibold text-gray-900 dark:text-gray-100">
                          {alert.symbol}
                        </span>
                        <div className="flex items-center gap-2">
                          <Badge className={getPriorityColor(alert.priority)} variant="outline">
                            {alert.priority.toUpperCase()}
                          </Badge>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {getRelativeTime(alert.timestamp)}
                          </span>
                        </div>
                      </div>

                      {/* Message */}
                      <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                        {alert.message}
                      </p>

                      {/* Timestamp */}
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTime(alert.timestamp)}
                      </p>

                      {/* Additional data */}
                      {alert.data && (
                        <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 space-y-1">
                          {alert.data.momentum !== undefined && (
                            <div>Momentum: {alert.data.momentum.toFixed(2)}</div>
                          )}
                          {alert.data.squeeze_bars !== undefined && (
                            <div>Squeeze Bars: {alert.data.squeeze_bars}</div>
                          )}
                          {alert.data.price !== undefined && (
                            <div>Price: ${alert.data.price.toFixed(2)}</div>
                          )}
                          {alert.data.reason && (
                            <div>Reason: {alert.data.reason}</div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
