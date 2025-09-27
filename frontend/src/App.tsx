import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Portfolio from './pages/Portfolio';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleMenuClick = () => {
    setSidebarOpen(true);
  };

  const handleSidebarClose = () => {
    setSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <Sidebar isOpen={sidebarOpen} onClose={handleSidebarClose} />
        
        {/* Main Content */}
        <div className="flex-1 flex flex-col lg:ml-64">
          <Header onMenuClick={handleMenuClick} />
          
          <main className="flex-1 p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/trading" element={<Trading />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
