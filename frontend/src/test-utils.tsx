/**
 * Custom render function with providers for testing
 * Provides consistent test environment with all necessary context providers
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { WebSocketProvider } from './context/WebSocketContext';

// Mock the toast functionality
jest.mock('react-hot-toast', () => ({
  Toaster: () => <div data-testid="toaster" />,
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    loading: jest.fn(),
    dismiss: jest.fn(),
  },
}));

// Mock the WebSocket context to avoid actual WebSocket connections in tests
const MockWebSocketProvider = ({ children }: { children: React.ReactNode }) => {
  const mockWebSocketValue = {
    isConnected: false,
    lastMessage: null,
    sendMessage: jest.fn(),
    subscribeToSymbol: jest.fn(),
    unsubscribeFromSymbol: jest.fn(),
  };

  return (
    <WebSocketProvider>
      {children}
    </WebSocketProvider>
  );
};

// Mock the Auth context
const MockAuthProvider = ({ children }: { children: React.ReactNode }) => {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
};

// Custom render function that includes all providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      <MockAuthProvider>
        <MockWebSocketProvider>
          {children}
        </MockWebSocketProvider>
      </MockAuthProvider>
    </BrowserRouter>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';

// Override render method
export { customRender as render };

// Helper function to render with specific providers
export const renderWithAuth = (
  ui: ReactElement,
  authValue?: any,
  options?: Omit<RenderOptions, 'wrapper'>
) => {
  const MockAuthProviderWithValue = ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <AuthProvider>
        <MockWebSocketProvider>
          {children}
        </MockWebSocketProvider>
      </AuthProvider>
    </BrowserRouter>
  );

  return render(ui, { wrapper: MockAuthProviderWithValue, ...options });
};

export const renderWithWebSocket = (
  ui: ReactElement,
  wsValue?: any,
  options?: Omit<RenderOptions, 'wrapper'>
) => {
  const MockWebSocketProviderWithValue = ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <MockAuthProvider>
        <WebSocketProvider>
          {children}
        </WebSocketProvider>
      </MockAuthProvider>
    </BrowserRouter>
  );

  return render(ui, { wrapper: MockWebSocketProviderWithValue, ...options });
};
