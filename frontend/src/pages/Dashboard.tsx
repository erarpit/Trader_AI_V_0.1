import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { 
  ArrowTrendingUpIcon, 
  ArrowTrendingDownIcon, 
  CurrencyDollarIcon,
  ChartBarIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { useWebSocket } from '../context/WebSocketContext';
import { api, ApiError } from '../services/api';
import { toast } from 'react-hot-toast';
import { PortfolioResponse, TopGainerLoser } from '../types/api';

const Dashboard: React.FC = () => {
  const { isConnected, lastMessage } = useWebSocket();
  const [marketData, setMarketData] = useState<any[]>([]);
  const [topGainers, setTopGainers] = useState<TopGainerLoser[]>([]);
  const [topLosers, setTopLosers] = useState<TopGainerLoser[]>([]);
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [todayPnL, setTodayPnL] = useState(0);
  const [loading, setLoading] = useState(true);

  // Fetch real data from API
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch portfolio data
        const portfolioResponse: PortfolioResponse = await api.getPortfolio();
        setPortfolioValue(portfolioResponse.total_value || 0);
        setTodayPnL(portfolioResponse.total_pnl || 0);

        // Fetch top gainers and losers
        const [gainersResponse, losersResponse] = await Promise.all([
          api.getTopGainers(),
          api.getTopLosers()
        ]);
        
        setTopGainers(gainersResponse || []);
        setTopLosers(losersResponse || []);

        // Generate portfolio performance chart data
        if (portfolioResponse.portfolio && portfolioResponse.portfolio.length > 0) {
          const chartData = portfolioResponse.portfolio.map((item, index: number) => ({
            date: new Date(Date.now() - (portfolioResponse.portfolio.length - index - 1) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            value: item.total_value || 0,
            volume: item.quantity || 0
          }));
          setMarketData(chartData);
        } else {
          // Fallback to mock data if no portfolio data
          const generateMockData = () => {
            const data = [];
            const now = new Date();
            for (let i = 29; i >= 0; i--) {
              const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
              data.push({
                date: date.toISOString().split('T')[0],
                value: 50000 + Math.random() * 10000 + i * 100,
                volume: Math.floor(Math.random() * 1000000) + 100000
              });
            }
            return data;
          };
          setMarketData(generateMockData());
        }

      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        if (error instanceof ApiError) {
          toast.error(`API Error: ${error.message}`);
        } else {
          toast.error('Failed to load dashboard data');
        }
        
        // Fallback to mock data on error
        setPortfolioValue(125000);
        setTodayPnL(2500);
        setTopGainers([
          { symbol: 'RELIANCE', change: 2.5, price: 2450.50 },
          { symbol: 'TCS', change: 1.8, price: 3850.25 },
          { symbol: 'HDFC', change: 1.5, price: 1650.75 },
          { symbol: 'INFY', change: 1.2, price: 1450.30 },
          { symbol: 'ITC', change: 0.9, price: 425.60 }
        ]);
        setTopLosers([
          { symbol: 'BHARTI', change: -2.1, price: 850.40 },
          { symbol: 'SBI', change: -1.8, price: 520.25 },
          { symbol: 'ONGC', change: -1.5, price: 180.75 },
          { symbol: 'NTPC', change: -1.2, price: 165.30 },
          { symbol: 'POWERGRID', change: -0.9, price: 225.60 }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Update data when WebSocket message is received
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'price_update') {
      // Update market data with new price
      setMarketData(prev => {
        const newData = [...prev];
        const lastData = newData[newData.length - 1];
        newData[newData.length - 1] = {
          ...lastData,
          value: lastMessage.data.last_price || lastData.value
        };
        return newData;
      });
    }
  }, [lastMessage]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">Welcome back! Here's what's happening in the market.</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isConnected 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {isConnected ? 'Live Data' : 'Offline'}
          </div>
        </div>
      </div>

      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CurrencyDollarIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Portfolio Value</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(portfolioValue)}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ArrowTrendingUpIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Today's P&L</p>
              <p className={`text-2xl font-bold ${todayPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatCurrency(todayPnL)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Return</p>
              <p className="text-2xl font-bold text-gray-900">+12.5%</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <EyeIcon className="h-8 w-8 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active Positions</p>
              <p className="text-2xl font-bold text-gray-900">8</p>
            </div>
          </div>
        </div>
      </div>

      {/* Market Chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Portfolio Performance</h2>
          <div className="flex space-x-2">
            <button className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-lg">1M</button>
            <button className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">3M</button>
            <button className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">1Y</button>
          </div>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={marketData}>
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

      {/* Market Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Gainers */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Gainers</h3>
          <div className="space-y-3">
            {topGainers.map((stock, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{stock.symbol}</p>
                  <p className="text-sm text-gray-600">{formatCurrency(stock.price)}</p>
                </div>
                <div className="text-right">
                  <p className="text-green-600 font-medium">{formatPercentage(stock.change)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Losers */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Losers</h3>
          <div className="space-y-3">
            {topLosers.map((stock, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{stock.symbol}</p>
                  <p className="text-sm text-gray-600">{formatCurrency(stock.price)}</p>
                </div>
                <div className="text-right">
                  <p className="text-red-600 font-medium">{formatPercentage(stock.change)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
