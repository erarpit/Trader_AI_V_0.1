/**
 * Type definitions for API responses
 */

export interface QuoteData {
  symbol: string;
  company_name: string;
  last_price: number;
  change: number;
  change_percent: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  value: number;
  market_cap: number;
  timestamp: string;
}

export interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketStatus {
  nse: any;
  bse: any;
  timestamp: string;
}

export interface TopGainerLoser {
  symbol: string;
  change: number;
  price: number;
}

export interface PortfolioItem {
  id?: number;
  symbol: string;
  company_name?: string;
  quantity: number;
  average_price: number;
  current_price: number;
  pnl: number;
  pnl_percentage: number;
  total_value: number;
  created_at?: string;
  updated_at?: string;
}

export interface PortfolioResponse {
  portfolio: PortfolioItem[];
  total_value: number;
  total_pnl: number;
}

export interface OrderItem {
  id: number;
  symbol: string;
  order_type: string;
  quantity: number;
  price: number;
  order_status: string;
  order_time: string;
  execution_time: string | null;
}

export interface OrdersResponse {
  orders: OrderItem[];
}

export interface OrderRequest {
  symbol: string;
  order_type: string;
  quantity: number;
  price: number;
  order_side: string;
}

export interface OrderResponse {
  order_id: number;
  status: string;
  message: string;
}

export interface AIAnalysis {
  symbol: string;
  analysis_type: string;
  signal: string;
  confidence: number;
  price_target: number;
  stop_loss: number;
  analysis_data: string;
  created_at: string;
}

export interface RiskMetrics {
  portfolio_value: number;
  var_1d: number;
  var_5d: number;
  sharpe_ratio: number;
  max_drawdown: number;
  beta: number;
  calculated_at: string;
}
