/**
 * API service for communicating with the backend
 */

import {
  QuoteData,
  HistoricalDataPoint,
  MarketStatus,
  TopGainerLoser,
  PortfolioResponse,
  OrdersResponse,
  OrderRequest,
  OrderResponse,
  AIAnalysis,
  RiskMetrics
} from '../types/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, errorData.detail || 'Request failed');
  }
  return response.json();
}

export const api = {
  // Real-time data endpoints
  async getQuote(symbol: string, exchange: string = 'NSE'): Promise<QuoteData> {
    const response = await fetch(`${API_BASE_URL}/realtime/quote/${symbol}?exchange=${exchange}`);
    return handleResponse<QuoteData>(response);
  },

  async getHistoricalData(symbol: string, exchange: string = 'NSE', fromDate?: string, toDate?: string): Promise<HistoricalDataPoint[]> {
    const params = new URLSearchParams({ exchange });
    if (fromDate) params.append('from_date', fromDate);
    if (toDate) params.append('to_date', toDate);
    
    const response = await fetch(`${API_BASE_URL}/realtime/historical/${symbol}?${params}`);
    return handleResponse<HistoricalDataPoint[]>(response);
  },

  async getMarketStatus(): Promise<MarketStatus> {
    const response = await fetch(`${API_BASE_URL}/realtime/market-status`);
    return handleResponse<MarketStatus>(response);
  },

  async getTopGainers(exchange: string = 'NSE'): Promise<TopGainerLoser[]> {
    const response = await fetch(`${API_BASE_URL}/realtime/top-gainers?exchange=${exchange}`);
    return handleResponse<TopGainerLoser[]>(response);
  },

  async getTopLosers(exchange: string = 'NSE'): Promise<TopGainerLoser[]> {
    const response = await fetch(`${API_BASE_URL}/realtime/top-losers?exchange=${exchange}`);
    return handleResponse<TopGainerLoser[]>(response);
  },

  // Trading endpoints
  async getPortfolio(userId: number = 1): Promise<PortfolioResponse> {
    const response = await fetch(`${API_BASE_URL}/trading/portfolio?user_id=${userId}`);
    return handleResponse<PortfolioResponse>(response);
  },

  async getOrders(userId: number = 1): Promise<OrdersResponse> {
    const response = await fetch(`${API_BASE_URL}/trading/orders?user_id=${userId}`);
    return handleResponse<OrdersResponse>(response);
  },

  async placeOrder(orderData: OrderRequest, userId: number = 1): Promise<OrderResponse> {
    const response = await fetch(`${API_BASE_URL}/trading/place-order?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(orderData),
    });
    return handleResponse<OrderResponse>(response);
  },

  async cancelOrder(orderId: number, userId: number = 1): Promise<OrderResponse> {
    const response = await fetch(`${API_BASE_URL}/trading/cancel-order/${orderId}?user_id=${userId}`, {
      method: 'DELETE',
    });
    return handleResponse<OrderResponse>(response);
  },

  // AI Analysis endpoints
  async getAIAnalysis(symbol: string): Promise<AIAnalysis> {
    const response = await fetch(`${API_BASE_URL}/ai/analyze/${symbol}`);
    return handleResponse<AIAnalysis>(response);
  },

  async getTradingSignals(): Promise<AIAnalysis[]> {
    const response = await fetch(`${API_BASE_URL}/ai/signals`);
    return handleResponse<AIAnalysis[]>(response);
  },

  // Risk Management endpoints
  async getRiskMetrics(userId: number = 1): Promise<RiskMetrics> {
    const response = await fetch(`${API_BASE_URL}/risk/metrics?user_id=${userId}`);
    return handleResponse<RiskMetrics>(response);
  },

  async getPortfolioRisk(userId: number = 1): Promise<RiskMetrics> {
    const response = await fetch(`${API_BASE_URL}/risk/portfolio?user_id=${userId}`);
    return handleResponse<RiskMetrics>(response);
  },
};

export { ApiError };
