import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import './App.css';

// Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import TradingSignalPopup from './components/TradingSignalPopup';
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Portfolio from './pages/Portfolio';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

// Context
import { WebSocketProvider } from './context/WebSocketContext';
import { AuthProvider } from './context/AuthContext';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <AuthProvider>
      <WebSocketProvider>
        <Router>
          <div className="min-h-screen bg-gray-100">
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
            
            <div className="lg:pl-64">
              <Header onMenuClick={() => setSidebarOpen(true)} />
              
              <main className="py-6">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/trading" element={<Trading />} />
                    <Route path="/portfolio" element={<Portfolio />} />
                    <Route path="/analytics" element={<Analytics />} />
                    <Route path="/settings" element={<Settings />} />
                  </Routes>
                </div>
              </main>
            </div>
            
            <TradingSignalPopup />
            
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
              }}
            />
          </div>
        </Router>
      </WebSocketProvider>
    </AuthProvider>
  );
}

export default App;
