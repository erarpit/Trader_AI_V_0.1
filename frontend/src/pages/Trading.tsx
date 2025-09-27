import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { 
  MagnifyingGlassIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ArrowUpIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline';
import { useWebSocket } from '../context/WebSocketContext';
import { api, ApiError } from '../services/api';
import { toast } from 'react-hot-toast';
import { QuoteData, HistoricalDataPoint, OrderRequest } from '../types/api';

const Trading: React.FC = () => {
  const { isConnected, subscribeToSymbol, unsubscribeFromSymbol, lastMessage } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [chartData, setChartData] = useState<HistoricalDataPoint[]>([]);
  const [orderType, setOrderType] = useState<'BUY' | 'SELL'>('BUY');
  const [quantity, setQuantity] = useState(1);
  const [orderPrice, setOrderPrice] = useState(0);
  const [orderSide, setOrderSide] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [loading, setLoading] = useState(false);

  // Fetch real data for selected symbol
  useEffect(() => {
    const fetchSymbolData = async () => {
      if (!selectedSymbol) return;
      
      try {
        setLoading(true);
        
        // Fetch current quote
        const quoteData: QuoteData = await api.getQuote(selectedSymbol);
        setCurrentPrice(quoteData.last_price || 0);
        setPriceChange(quoteData.change_percent || 0);
        setOrderPrice(quoteData.last_price || 0);

        // Fetch historical data for chart
        const historicalData: HistoricalDataPoint[] = await api.getHistoricalData(selectedSymbol);
        if (historicalData && historicalData.length > 0) {
          setChartData(historicalData);
        } else {
          // Fallback to mock data
          const generateCandlestickData = (): HistoricalDataPoint[] => {
            const data: HistoricalDataPoint[] = [];
            const basePrice = quoteData.last_price || 2450;
            let currentPrice = basePrice;
            
            for (let i = 29; i >= 0; i--) {
              const date = new Date();
              date.setDate(date.getDate() - i);
              
              const open = currentPrice;
              const close = open + (Math.random() - 0.5) * 50;
              const high = Math.max(open, close) + Math.random() * 20;
              const low = Math.min(open, close) - Math.random() * 20;
              
              data.push({
                date: date.toISOString().split('T')[0],
                open: open,
                high: high,
                low: low,
                close: close,
                volume: Math.floor(Math.random() * 1000000) + 100000
              });
              
              currentPrice = close;
            }
            return data;
          };
          setChartData(generateCandlestickData());
        }

      } catch (error) {
        console.error('Error fetching symbol data:', error);
        if (error instanceof ApiError) {
          toast.error(`API Error: ${error.message}`);
        } else {
          toast.error('Failed to load symbol data');
        }
        
        // Fallback to mock data
        setCurrentPrice(2450.50);
        setPriceChange(2.5);
        setOrderPrice(2450.50);
      } finally {
        setLoading(false);
      }
    };

    fetchSymbolData();
  }, [selectedSymbol]);

  // Update price from WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'price_update' && lastMessage.symbol === selectedSymbol) {
      const newPrice = lastMessage.data.last_price;
      if (newPrice) {
        setCurrentPrice(newPrice);
        if (orderSide === 'MARKET') {
          setOrderPrice(newPrice);
        }
      }
    }
  }, [lastMessage, selectedSymbol, orderSide]);

  // Subscribe to symbol updates
  useEffect(() => {
    if (selectedSymbol) {
      subscribeToSymbol(selectedSymbol);
      return () => unsubscribeFromSymbol(selectedSymbol);
    }
  }, [selectedSymbol, subscribeToSymbol, unsubscribeFromSymbol]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const handlePlaceOrder = async () => {
    try {
      setLoading(true);
      
      const orderData: OrderRequest = {
        symbol: selectedSymbol,
        order_type: orderType,
        quantity,
        price: orderSide === 'MARKET' ? currentPrice : orderPrice,
        order_side: orderSide
      };

      const result = await api.placeOrder(orderData);
      
      toast.success(result.message || 'Order placed successfully!');
      
      // Reset form
      setQuantity(1);
      setOrderPrice(currentPrice);
      
    } catch (error) {
      console.error('Error placing order:', error);
      if (error instanceof ApiError) {
        toast.error(`Order failed: ${error.message}`);
      } else {
        toast.error('Failed to place order');
      }
    } finally {
      setLoading(false);
    }
  };

  const popularStocks = [
    'RELIANCE', 'TCS', 'HDFC', 'INFY', 'ITC', 'BHARTI', 'SBI', 'ONGC'
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trading Terminal</h1>
          <p className="text-gray-600">Real-time trading with advanced charts and analysis</p>
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

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Panel - Stock Search and Info */}
        <div className="lg:col-span-1 space-y-6">
          {/* Stock Search */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Search Stocks</h3>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by symbol..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            {/* Popular Stocks */}
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-600 mb-2">Popular Stocks</p>
              <div className="space-y-1">
                {popularStocks.map((symbol) => (
                  <button
                    key={symbol}
                    onClick={() => setSelectedSymbol(symbol)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                      selectedSymbol === symbol
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {symbol}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Current Stock Info */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Stock Info</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Symbol:</span>
                <span className="font-medium">{selectedSymbol}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Current Price:</span>
                <span className="font-medium">{formatCurrency(currentPrice)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Change:</span>
                <span className={`font-medium ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatPercentage(priceChange)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Volume:</span>
                <span className="font-medium">1.2M</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - Chart */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{selectedSymbol} Chart</h3>
              <div className="flex space-x-2">
                <button className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-lg">1D</button>
                <button className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">1W</button>
                <button className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">1M</button>
              </div>
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
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
                    formatter={(value: any) => [formatCurrency(value), 'Price']}
                    labelFormatter={(label) => new Date(label).toLocaleDateString()}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#3B82F6" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Panel - Order Placement */}
        <div className="lg:col-span-1">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Place Order</h3>
            
            {/* Order Type */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Order Type</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderType('BUY')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    orderType === 'BUY'
                      ? 'bg-green-100 text-green-700 border-2 border-green-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <ArrowUpIcon className="h-4 w-4 inline mr-1" />
                  BUY
                </button>
                <button
                  onClick={() => setOrderType('SELL')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    orderType === 'SELL'
                      ? 'bg-red-100 text-red-700 border-2 border-red-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <ArrowDownIcon className="h-4 w-4 inline mr-1" />
                  SELL
                </button>
              </div>
            </div>

            {/* Order Side */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Order Side</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderSide('MARKET')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    orderSide === 'MARKET'
                      ? 'bg-blue-100 text-blue-700 border-2 border-blue-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Market
                </button>
                <button
                  onClick={() => setOrderSide('LIMIT')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    orderSide === 'LIMIT'
                      ? 'bg-blue-100 text-blue-700 border-2 border-blue-300'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Limit
                </button>
              </div>
            </div>

            {/* Quantity */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Quantity</label>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            {/* Price (for limit orders) */}
            {orderSide === 'LIMIT' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Price</label>
                <input
                  type="number"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  step="0.01"
                />
              </div>
            )}

            {/* Order Summary */}
            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Total Value:</span>
                <span className="font-medium">
                  {formatCurrency((orderSide === 'MARKET' ? currentPrice : orderPrice) * quantity)}
                </span>
              </div>
            </div>

            {/* Place Order Button */}
            <button
              onClick={handlePlaceOrder}
              className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
                orderType === 'BUY'
                  ? 'bg-green-600 hover:bg-green-700'
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {orderType} {quantity} {selectedSymbol}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Trading;
