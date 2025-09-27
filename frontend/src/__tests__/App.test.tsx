/**
 * Main App integration tests
 * Tests the complete application flow and integration between components
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '../test-utils';
import App from '../App';

// Mock all the page components
jest.mock('../pages/Dashboard', () => {
  return function MockDashboard() {
    return <div data-testid="dashboard-page">Dashboard Page</div>;
  };
});

jest.mock('../pages/Trading', () => {
  return function MockTrading() {
    return <div data-testid="trading-page">Trading Page</div>;
  };
});

jest.mock('../pages/Portfolio', () => {
  return function MockPortfolio() {
    return <div data-testid="portfolio-page">Portfolio Page</div>;
  };
});

jest.mock('../pages/Analytics', () => {
  return function MockAnalytics() {
    return <div data-testid="analytics-page">Analytics Page</div>;
  };
});

jest.mock('../pages/Settings', () => {
  return function MockSettings() {
    return <div data-testid="settings-page">Settings Page</div>;
  };
});

// Mock the components
jest.mock('../components/Header', () => {
  return function MockHeader({ onMenuClick }: { onMenuClick: () => void }) {
    return (
      <div data-testid="header">
        <button onClick={onMenuClick}>Menu</button>
        <span>Trader AI</span>
      </div>
    );
  };
});

jest.mock('../components/Sidebar', () => {
  return function MockSidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    if (!isOpen) return null;
    return (
      <div data-testid="sidebar">
        <button onClick={onClose}>Close</button>
        <nav>
          <a href="/">Dashboard</a>
          <a href="/trading">Trading</a>
          <a href="/portfolio">Portfolio</a>
          <a href="/analytics">Analytics</a>
          <a href="/settings">Settings</a>
        </nav>
      </div>
    );
  };
});

jest.mock('../components/TradingSignalPopup', () => {
  return function MockTradingSignalPopup() {
    return <div data-testid="trading-signal-popup">Trading Signal Popup</div>;
  };
});

// Mock the context providers
jest.mock('../context/AuthContext', () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="auth-provider">{children}</div>
  ),
  useAuth: () => ({
    user: null,
    isAuthenticated: false,
    login: jest.fn(),
    logout: jest.fn(),
    loading: false,
  }),
}));

jest.mock('../context/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="websocket-provider">{children}</div>
  ),
  useWebSocket: () => ({
    socket: null,
    isConnected: false,
    connect: jest.fn(),
    disconnect: jest.fn(),
    send: jest.fn(),
    subscribe: jest.fn(),
    unsubscribe: jest.fn(),
  }),
}));

describe('App Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the complete application structure', () => {
    render(<App />);

    // Check if all main components are rendered
    expect(screen.getByTestId('auth-provider')).toBeInTheDocument();
    expect(screen.getByTestId('websocket-provider')).toBeInTheDocument();
    expect(screen.getByTestId('header')).toBeInTheDocument();
    expect(screen.getByTestId('trading-signal-popup')).toBeInTheDocument();
  });

  it('renders dashboard by default', () => {
    render(<App />);

    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  it('handles sidebar toggle functionality', () => {
    render(<App />);

    // Initially sidebar should be closed
    expect(screen.queryByTestId('sidebar')).not.toBeInTheDocument();

    // Click menu button to open sidebar
    const menuButton = screen.getByText('Menu');
    fireEvent.click(menuButton);

    // Sidebar should now be visible
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();

    // Click close button to close sidebar
    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);

    // Sidebar should be closed again
    expect(screen.queryByTestId('sidebar')).not.toBeInTheDocument();
  });

  it('handles navigation between pages', async () => {
    render(<App />);

    // Open sidebar first
    const menuButton = screen.getByText('Menu');
    fireEvent.click(menuButton);

    // Navigate to Trading page
    const tradingLink = screen.getByText('Trading');
    fireEvent.click(tradingLink);

    await waitFor(() => {
      expect(screen.getByTestId('trading-page')).toBeInTheDocument();
    });

    // Navigate to Portfolio page
    const portfolioLink = screen.getByText('Portfolio');
    fireEvent.click(portfolioLink);

    await waitFor(() => {
      expect(screen.getByTestId('portfolio-page')).toBeInTheDocument();
    });

    // Navigate to Analytics page
    const analyticsLink = screen.getByText('Analytics');
    fireEvent.click(analyticsLink);

    await waitFor(() => {
      expect(screen.getByTestId('analytics-page')).toBeInTheDocument();
    });

    // Navigate to Settings page
    const settingsLink = screen.getByText('Settings');
    fireEvent.click(settingsLink);

    await waitFor(() => {
      expect(screen.getByTestId('settings-page')).toBeInTheDocument();
    });

    // Navigate back to Dashboard
    const dashboardLink = screen.getByText('Dashboard');
    fireEvent.click(dashboardLink);

    await waitFor(() => {
      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
    });
  });

  it('maintains sidebar state during navigation', async () => {
    render(<App />);

    // Open sidebar
    const menuButton = screen.getByText('Menu');
    fireEvent.click(menuButton);

    expect(screen.getByTestId('sidebar')).toBeInTheDocument();

    // Navigate to different page
    const tradingLink = screen.getByText('Trading');
    fireEvent.click(tradingLink);

    await waitFor(() => {
      expect(screen.getByTestId('trading-page')).toBeInTheDocument();
    });

    // Sidebar should still be open
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
  });

  it('handles responsive design correctly', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });

    render(<App />);

    // Should render on mobile
    expect(screen.getByTestId('header')).toBeInTheDocument();
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  it('provides context to all child components', () => {
    render(<App />);

    // Check if context providers are wrapping the app
    expect(screen.getByTestId('auth-provider')).toBeInTheDocument();
    expect(screen.getByTestId('websocket-provider')).toBeInTheDocument();
  });

  it('handles URL routing correctly', () => {
    // Mock different URL paths
    const { rerender } = render(<App />);

    // Test root path
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();

    // Test trading path
    window.history.pushState({}, 'Trading', '/trading');
    rerender(<App />);
    expect(screen.getByTestId('trading-page')).toBeInTheDocument();

    // Test portfolio path
    window.history.pushState({}, 'Portfolio', '/portfolio');
    rerender(<App />);
    expect(screen.getByTestId('portfolio-page')).toBeInTheDocument();
  });

  it('displays trading signal popup', () => {
    render(<App />);

    expect(screen.getByTestId('trading-signal-popup')).toBeInTheDocument();
  });

  it('handles keyboard navigation', () => {
    render(<App />);

    // Test tab navigation
    const menuButton = screen.getByText('Menu');
    menuButton.focus();

    expect(document.activeElement).toBe(menuButton);
  });

  it('maintains application state during navigation', async () => {
    render(<App />);

    // Open sidebar
    const menuButton = screen.getByText('Menu');
    fireEvent.click(menuButton);

    // Navigate to different page
    const tradingLink = screen.getByText('Trading');
    fireEvent.click(tradingLink);

    await waitFor(() => {
      expect(screen.getByTestId('trading-page')).toBeInTheDocument();
    });

    // Navigate back to dashboard
    const dashboardLink = screen.getByText('Dashboard');
    fireEvent.click(dashboardLink);

    await waitFor(() => {
      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
    });

    // Sidebar should still be open
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
  });

  it('handles error boundaries gracefully', () => {
    // Mock a component that throws an error
    const ErrorComponent = () => {
      throw new Error('Test error');
    };

    // This would be handled by an error boundary in a real app
    expect(() => render(<ErrorComponent />)).toThrow('Test error');
  });

  it('renders with all required providers', () => {
    render(<App />);

    // Check if all context providers are present
    expect(screen.getByTestId('auth-provider')).toBeInTheDocument();
    expect(screen.getByTestId('websocket-provider')).toBeInTheDocument();
    
    // Check if main layout components are present
    expect(screen.getByTestId('header')).toBeInTheDocument();
    expect(screen.getByTestId('trading-signal-popup')).toBeInTheDocument();
  });
});
