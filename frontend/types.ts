import React from 'react';

export interface NavItem {
  label: string;
  path: string;
}

export interface Feature {
  title: string;
  description: string;
  icon: React.ComponentType<any>;
}

export interface Testimonial {
  content: string;
  author: string;
  role: string;
  company: string;
}

export interface PricingTier {
  name: string;
  price: string;
  target: string;
  features: string[];
  recommended?: boolean;
  cta: string;
}

export interface ProcessStep {
  step: number;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
}