import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  ChartBarIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  CurrencyDollarIcon,
  EyeIcon,
  BellIcon
} from '@heroicons/react/24/outline';

const Analytics: React.FC = () => {
  const [aiSignals, setAiSignals] = useState<any[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<any>({});
  const [marketSentiment, setMarketSentiment] = useState<any>({});
  const [volumeAnalysis, setVolumeAnalysis] = useState<any>({});

  // Mock data for demonstration
  useEffect(() => {
    // AI Signals data
    const mockSignals = [
      {
        symbol: 'RELIANCE',
        signal: 'BUY',
        confidence: 0.85,
        priceTarget: 2500,
        stopLoss: 2400,
        timestamp: '2024-01-15T10:30:00Z',
        reasoning: 'Strong technical indicators and positive sentiment'
      },
      {
        symbol: 'TCS',
        signal: 'HOLD',
        confidence: 0.65,
        priceTarget: 3900,
        stopLoss: 3800,
        timestamp: '2024-01-15T10:25:00Z',
        reasoning: 'Mixed signals, wait for confirmation'
      },
      {
        symbol: 'HDFC',
        signal: 'SELL',
        confidence: 0.78,
        priceTarget: 1600,
        stopLoss: 1700,
        timestamp: '2024-01-15T10:20:00Z',
        reasoning: 'Negative momentum and high volatility'
      }
    ];

    setAiSignals(mockSignals);

    // Risk metrics
    setRiskMetrics({
      portfolioValue: 125000,
      var1d: 2500,
      var5d: 5000,
      sharpeRatio: 1.25,
      maxDrawdown: 0.08,
      beta: 1.1,
      concentrationRisk: 0.15,
      correlationRisk: 0.35
    });

    // Market sentiment
    setMarketSentiment({
      overall: 'BULLISH',
      score: 0.75,
      newsSentiment: 'POSITIVE',
      socialSentiment: 'NEUTRAL',
      technicalSentiment: 'BULLISH'
    });

    // Volume analysis
    setVolumeAnalysis({
      averageVolume: 1500000,
      currentVolume: 2100000,
      volumeRatio: 1.4,
      volumeTrend: 'INCREASING',
      highVolumeSpikes: 3,
      volumeSignals: ['HIGH_VOLUME', 'VOLUME_BREAKOUT']
    });
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-600 bg-green-100';
      case 'SELL': return 'text-red-600 bg-red-100';
      case 'HOLD': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Mock chart data
  const performanceData = [
    { date: '2024-01-01', value: 100000 },
    { date: '2024-01-02', value: 102500 },
    { date: '2024-01-03', value: 101800 },
    { date: '2024-01-04', value: 103200 },
    { date: '2024-01-05', value: 105000 },
    { date: '2024-01-06', value: 104500 },
    { date: '2024-01-07', value: 106800 },
    { date: '2024-01-08', value: 108200 },
    { date: '2024-01-09', value: 107500 },
    { date: '2024-01-10', value: 109000 },
    { date: '2024-01-11', value: 111200 },
    { date: '2024-01-12', value: 110800 },
    { date: '2024-01-13', value: 112500 },
    { date: '2024-01-14', value: 114000 },
    { date: '2024-01-15', value: 115500 }
  ];

  const sectorAllocation = [
    { name: 'Technology', value: 35, color: '#3B82F6' },
    { name: 'Banking', value: 25, color: '#10B981' },
    { name: 'Energy', value: 20, color: '#F59E0B' },
    { name: 'Healthcare', value: 15, color: '#EF4444' },
    { name: 'Others', value: 5, color: '#8B5CF6' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics & AI Insights</h1>
          <p className="text-gray-600">Advanced market analysis and AI-powered recommendations</p>
        </div>
        <div className="flex space-x-2">
          <button className="btn-secondary">
            <BellIcon className="h-5 w-5 mr-2" />
            Set Alerts
          </button>
          <button className="btn-primary">
            <ChartBarIcon className="h-5 w-5 mr-2" />
            Generate Report
          </button>
        </div>
      </div>

      {/* AI Signals */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">AI Trading Signals</h2>
        <div className="space-y-4">
          {aiSignals.map((signal, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="text-lg font-semibold text-gray-900">{signal.symbol}</div>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(signal.signal)}`}>
                    {signal.signal}
                  </div>
                  <div className={`text-sm font-medium ${getConfidenceColor(signal.confidence)}`}>
                    {formatPercentage(signal.confidence)} Confidence
                  </div>
                </div>
                <div className="text-right text-sm text-gray-600">
                  <div>Target: {formatCurrency(signal.priceTarget)}</div>
                  <div>Stop Loss: {formatCurrency(signal.stopLoss)}</div>
                </div>
              </div>
              <div className="mt-2 text-sm text-gray-600">
                {signal.reasoning}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Metrics</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-gray-600">Portfolio Value:</span>
              <span className="font-medium">{formatCurrency(riskMetrics.portfolioValue)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">1-Day VaR:</span>
              <span className="font-medium text-red-600">{formatCurrency(riskMetrics.var1d)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">5-Day VaR:</span>
              <span className="font-medium text-red-600">{formatCurrency(riskMetrics.var5d)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Sharpe Ratio:</span>
              <span className="font-medium text-green-600">{riskMetrics.sharpeRatio}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Max Drawdown:</span>
              <span className="font-medium text-red-600">{formatPercentage(riskMetrics.maxDrawdown)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Beta:</span>
              <span className="font-medium">{riskMetrics.beta}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Sentiment</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-gray-600">Overall Sentiment:</span>
              <span className={`font-medium ${marketSentiment.overall === 'BULLISH' ? 'text-green-600' : 'text-red-600'}`}>
                {marketSentiment.overall}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Sentiment Score:</span>
              <span className="font-medium">{formatPercentage(marketSentiment.score)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">News Sentiment:</span>
              <span className="font-medium">{marketSentiment.newsSentiment}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Social Sentiment:</span>
              <span className="font-medium">{marketSentiment.socialSentiment}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Technical Sentiment:</span>
              <span className="font-medium">{marketSentiment.technicalSentiment}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Performance</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => formatCurrency(value)}
              />
              <Tooltip 
                formatter={(value: any) => [formatCurrency(value), 'Portfolio Value']}
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Volume Analysis and Sector Allocation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Volume Analysis</h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-gray-600">Average Volume:</span>
              <span className="font-medium">{volumeAnalysis.averageVolume?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Current Volume:</span>
              <span className="font-medium">{volumeAnalysis.currentVolume?.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Volume Ratio:</span>
              <span className="font-medium text-green-600">{volumeAnalysis.volumeRatio}x</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Volume Trend:</span>
              <span className="font-medium text-green-600">{volumeAnalysis.volumeTrend}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">High Volume Spikes:</span>
              <span className="font-medium">{volumeAnalysis.highVolumeSpikes}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sector Allocation</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sectorAllocation}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name} ${value}%`}
                >
                  {sectorAllocation.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
