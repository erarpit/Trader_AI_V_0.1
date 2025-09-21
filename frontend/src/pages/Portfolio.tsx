import React, { useState, useEffect } from 'react';
import { 
  BriefcaseIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  EyeIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const Portfolio: React.FC = () => {
  const [portfolio, setPortfolio] = useState<any[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [totalPnL, setTotalPnL] = useState(0);
  const [totalPnLPercentage, setTotalPnLPercentage] = useState(0);

  // Mock portfolio data
  useEffect(() => {
    const mockPortfolio = [
      {
        id: 1,
        symbol: 'RELIANCE',
        companyName: 'Reliance Industries Ltd',
        quantity: 50,
        averagePrice: 2400.00,
        currentPrice: 2450.50,
        pnl: 2525.00,
        pnlPercentage: 2.10,
        totalValue: 122525.00
      },
      {
        id: 2,
        symbol: 'TCS',
        companyName: 'Tata Consultancy Services Ltd',
        quantity: 25,
        averagePrice: 3800.00,
        currentPrice: 3850.25,
        pnl: 1256.25,
        pnlPercentage: 1.32,
        totalValue: 96256.25
      },
      {
        id: 3,
        symbol: 'HDFC',
        companyName: 'HDFC Bank Ltd',
        quantity: 75,
        averagePrice: 1600.00,
        currentPrice: 1650.75,
        pnl: 3806.25,
        pnlPercentage: 3.17,
        totalValue: 123806.25
      },
      {
        id: 4,
        symbol: 'INFY',
        companyName: 'Infosys Ltd',
        quantity: 100,
        averagePrice: 1400.00,
        currentPrice: 1450.30,
        pnl: 5030.00,
        pnlPercentage: 3.59,
        totalValue: 145030.00
      },
      {
        id: 5,
        symbol: 'ITC',
        companyName: 'ITC Ltd',
        quantity: 200,
        averagePrice: 420.00,
        currentPrice: 425.60,
        pnl: 1120.00,
        pnlPercentage: 1.33,
        totalValue: 85120.00
      }
    ];

    setPortfolio(mockPortfolio);
    
    // Calculate totals
    const total = mockPortfolio.reduce((sum, item) => sum + item.totalValue, 0);
    const pnl = mockPortfolio.reduce((sum, item) => sum + item.pnl, 0);
    const pnlPercentage = total > 0 ? (pnl / (total - pnl)) * 100 : 0;
    
    setTotalValue(total);
    setTotalPnL(pnl);
    setTotalPnLPercentage(pnlPercentage);
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
              <TrendingUpIcon className="h-8 w-8 text-green-600" />
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
                    <div className="text-sm text-gray-900">{position.companyName}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{position.quantity}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.averagePrice)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.currentPrice)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <div className={`text-sm font-medium ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatCurrency(position.pnl)}
                      </div>
                      <div className={`text-xs ${position.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatPercentage(position.pnlPercentage)}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{formatCurrency(position.totalValue)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-900">Edit</button>
                      <button 
                        onClick={() => handleRemovePosition(position.id)}
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
