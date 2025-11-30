import React from 'react';
import { LucideIcon } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'gold';
  size?: 'sm' | 'md' | 'lg';
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  ...props 
}) => {
  const baseStyles = "inline-flex items-center justify-center rounded-lg font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";
  
  const variants = {
    primary: "bg-primary-main text-white hover:bg-primary-dark shadow-md hover:shadow-lg focus:ring-primary-main",
    secondary: "bg-secondary-main text-white hover:bg-secondary-dark shadow-md hover:shadow-lg focus:ring-secondary-main",
    outline: "border-2 border-primary-main text-primary-main hover:bg-primary-main hover:text-white focus:ring-primary-main",
    gold: "bg-accent-gold text-primary-dark hover:bg-yellow-400 shadow-md hover:shadow-xl focus:ring-accent-gold font-bold",
  };

  const sizes = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-5 py-2.5 text-base",
    lg: "px-8 py-3.5 text-lg",
  };

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`} 
      {...props}
    >
      {children}
    </button>
  );
};

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({ icon: Icon, title, description }) => {
  return (
    <div className="bg-white p-6 rounded-xl border border-neutral-200 shadow-sm hover:shadow-xl hover:border-secondary-main transition-all duration-300 group h-full">
      <div className="w-14 h-14 rounded-lg bg-gradient-to-br from-primary-light to-primary-main flex items-center justify-center mb-6 shadow-inner group-hover:scale-110 transition-transform">
        <Icon className="h-7 w-7 text-white" />
      </div>
      <h3 className="text-xl font-bold text-primary-dark mb-3">{title}</h3>
      <p className="text-neutral-600 leading-relaxed">{description}</p>
    </div>
  );
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'success' | 'warning' | 'neutral';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'neutral', className = '' }) => {
  const variants = {
    success: "bg-green-100 text-green-800",
    warning: "bg-yellow-100 text-yellow-800",
    neutral: "bg-neutral-100 text-neutral-800",
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
};