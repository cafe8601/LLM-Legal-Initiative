import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Scale } from 'lucide-react';
import { NavItem } from '../types';

const navItems: NavItem[] = [
  { label: '기능', path: '/features' },
  { label: '작동 원리', path: '/how-it-works' },
  { label: '요금제', path: '/pricing' },
  { label: '소개', path: '/about' },
  { label: '문의하기', path: '/contact' },
];

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <nav className="fixed w-full z-50 bg-white/90 backdrop-blur-md border-b border-neutral-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <Link to="/" className="flex items-center space-x-2">
            <div className="p-2 bg-primary-main rounded-lg">
              <Scale className="h-6 w-6 text-white" />
            </div>
            <span className="font-heading font-bold text-xl text-primary-dark">Legal Council</span>
          </Link>

          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-sm font-medium transition-colors hover:text-secondary-main ${
                  location.pathname === item.path ? 'text-secondary-main' : 'text-neutral-600'
                }`}
              >
                {item.label}
              </Link>
            ))}
            <Link
              to="/expert-chat"
              className="px-5 py-2.5 rounded-lg bg-primary-main text-white font-medium text-sm hover:bg-primary-dark transition-all shadow-md hover:shadow-lg"
            >
              상담 시작
            </Link>
          </div>

          <div className="md:hidden flex items-center">
            <button onClick={toggleMenu} className="text-neutral-600 hover:text-primary-main p-2">
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden bg-white border-t border-neutral-100">
          <div className="px-4 pt-2 pb-6 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setIsOpen(false)}
                className={`block px-3 py-3 rounded-md text-base font-medium ${
                  location.pathname === item.path
                    ? 'bg-neutral-50 text-secondary-main'
                    : 'text-neutral-600 hover:bg-neutral-50 hover:text-primary-main'
                }`}
              >
                {item.label}
              </Link>
            ))}
            <div className="pt-4">
              <Link
                to="/expert-chat"
                onClick={() => setIsOpen(false)}
                className="block w-full text-center px-4 py-3 rounded-lg bg-primary-main text-white font-bold"
              >
                상담 시작
              </Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;