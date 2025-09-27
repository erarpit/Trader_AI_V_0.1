import React, { useState, useEffect } from 'react';
import { 
  BriefcaseIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  EyeIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { api, ApiError } from '../services/api';
import { toast } from 'react-hot-toast';
import { PortfolioResponse, PortfolioItem } from '../types/api';

const Portfolio: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioItem[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [totalPnL, setTotalPnL] = useState(0);
  const [totalPnLPercentage, setTotalPnLPercentage] = useState(0);
  const [loading, setLoading] = useState(true);

  // Fetch real portfolio data
  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        setLoading(true);
        const response: PortfolioResponse = await api.getPortfolio();
        
        if (response.portfolio && response.portfolio.length > 0) {
          setPortfolio(response.portfolio);
          setTotalValue(response.total_value || 0);
          setTotalPnL(response.total_pnl || 0);
          
          // Calculate PnL percentage
          const investedValue = totalValue - totalPnL;
          const pnlPercentage = investedValue > 0 ? (totalPnL / investedValue) * 100 : 0;
          setTotalPnLPercentage(pnlPercentage);
        } else {
          // Fallback to mock data if no portfolio
          const mockPortfolio: PortfolioItem[] = [
            {
              id: 1,
              symbol: 'RELIANCE',
              company_name: 'Reliance Industries Ltd',
              quantity: 50,
              average_price: 2400.00,
              current_price: 2450.50,
              pnl: 2525.00,
              pnl_percentage: 2.10,
              total_value: 122525.00,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            },
            {
              id: 2,
              symbol: 'TCS',
              company_name: 'Tata Consultancy Services Ltd',
              quantity: 25,
              average_price: 3800.00,
              current_price: 3850.25,
              pnl: 1256.25,
              pnl_percentage: 1.32,
              total_value: 96256.25,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            }
          ];
          
          setPortfolio(mockPortfolio);
          const total = mockPortfolio.reduce((sum, item) => sum + item.total_value, 0);
          const pnl = mockPortfolio.reduce((sum, item) => sum + item.pnl, 0);
          const pnlPercentage = total > 0 ? (pnl / (total - pnl)) * 100 : 0;
          
          setTotalValue(total);
          setTotalPnL(pnl);
          setTotalPnLPercentage(pnlPercentage);
        }

      } catch (error) {
        console.error('Error fetching portfolio data:', error);
        if (error instanceof ApiError) {
          toast.error(`API Error: ${error.message}`);
        } else {
          toast.error('Failed to load portfolio data');
        }
        
        // Fallback to empty portfolio
        setPortfolio([]);
        setTotalValue(0);
        setTotalPnL(0);
        setTotalPnLPercentage(0);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolioData();
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
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const handleRemovePosition = (id: number) => {
    setPortfolio(prev => prev.filter(item => item.id !== id));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Portfolio</h1>
          <p className="text-gray-600">Manage your investments and track performance</p>
        </div>
        <button className="btn-primary">
          <ChartBarIcon className="h-5 w-5 mr-2" />
          Analyze Portfolio
        </button>
      </div>

      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CurrencyDollarIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Value</p>
              <p className="text-2xl font-bold text-gray-900">{formatCurrency(totalValue)}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ArrowTrendingUpIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total P&L</p>
              <p className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatCurrency(totalPnL)}
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
              <p className={`text-2xl font-bold ${totalPnLPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(totalPnLPercentage)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <EyeIcon className="h-8 w-8 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Positions</p>
              <p className="text-2xl font-bold text-gray-900">{portfolio.length}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Holdings</h2>
          <div className="flex space-x-2">
            <button className="btn-secondary">Export</button>
            <button className="btn-primary">Add Position</button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Company
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Quantity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolio.map((position) => (
                <tr key={position.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{position.symbol}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{position.company_name || position.symbol}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{position.quantity}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.average_price)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.current_price)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <div className={`text-sm font-medium ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatCurrency(position.pnl)}
                      </div>
                      <div className={`text-xs ${position.pnl_percentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatPercentage(position.pnl_percentage)}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.total_value)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-900">Edit</button>
                      <button 
                        onClick={() => position.id && handleRemovePosition(position.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Portfolio Performance Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Performance</h2>
        <div className="h-80 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">Portfolio performance chart will be displayed here</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
