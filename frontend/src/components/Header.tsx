import React from 'react';
import { 
  Bars3Icon, 
  BellIcon, 
  MagnifyingGlassIcon,
  WifiIcon,
  SignalSlashIcon
} from '@heroicons/react/24/outline';
import { useWebSocket } from '../context/WebSocketContext';

interface HeaderProps {
  onMenuClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick }) => {
  const { isConnected } = useWebSocket();

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side */}
          <div className="flex items-center">
            <button
              onClick={onMenuClick}
              className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100"
            >
              <Bars3Icon className="h-6 w-6" />
            </button>
            
            {/* Search bar */}
            <div className="hidden md:block ml-4">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  placeholder="Search stocks..."
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                />
              </div>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* Connection status */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <WifiIcon className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-green-600 font-medium">Live</span>
                </>
              ) : (
                <>
                  <SignalSlashIcon className="h-5 w-5 text-red-500" />
                  <span className="text-sm text-red-600 font-medium">Offline</span>
                </>
              )}
            </div>

            {/* Notifications */}
            <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg">
              <BellIcon className="h-6 w-6" />
            </button>

            {/* User menu */}
            <div className="flex items-center space-x-3">
              <div className="hidden md:block text-right">
                <p className="text-sm font-medium text-gray-900">Demo User</p>
                <p className="text-xs text-gray-500">demo@traderai.com</p>
              </div>
              <div className="h-8 w-8 bg-blue-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">D</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
