import React from 'react';
import { Link } from 'react-router-dom';
import { Scale, Mail, MapPin, Github, Twitter, Linkedin } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-neutral-900 text-white pt-16 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Scale className="h-6 w-6 text-accent-gold" />
              <span className="font-heading font-bold text-xl">LLM Legal Council</span>
            </div>
            <p className="text-neutral-400 text-sm leading-relaxed">
              4개의 최신 AI 모델이 집단 지성으로 분석하고, 상호 검증하며, 
              근거 있는 법률 의견을 제시하는 차세대 법률 자문 플랫폼입니다.
            </p>
          </div>

          <div>
            <h4 className="font-bold text-lg mb-4 text-neutral-100">제품</h4>
            <ul className="space-y-2 text-neutral-400">
              <li><Link to="/features" className="hover:text-secondary-main transition-colors">주요 기능</Link></li>
              <li><Link to="/how-it-works" className="hover:text-secondary-main transition-colors">작동 원리</Link></li>
              <li><Link to="/pricing" className="hover:text-secondary-main transition-colors">요금제</Link></li>
              <li><Link to="/docs" className="hover:text-secondary-main transition-colors">API 문서</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-lg mb-4 text-neutral-100">회사</h4>
            <ul className="space-y-2 text-neutral-400">
              <li><Link to="/about" className="hover:text-secondary-main transition-colors">소개</Link></li>
              <li><Link to="/blog" className="hover:text-secondary-main transition-colors">블로그</Link></li>
              <li><Link to="/careers" className="hover:text-secondary-main transition-colors">채용</Link></li>
              <li><Link to="/contact" className="hover:text-secondary-main transition-colors">문의하기</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-lg mb-4 text-neutral-100">연락처</h4>
            <ul className="space-y-3 text-neutral-400">
              <li className="flex items-center space-x-3">
                <Mail className="h-5 w-5 text-secondary-main" />
                <span>support@legalcouncil.ai</span>
              </li>
              <li className="flex items-center space-x-3">
                <MapPin className="h-5 w-5 text-secondary-main" />
                <span>서울 강남구 테헤란로 123</span>
              </li>
              <li className="flex space-x-4 pt-2">
                <a href="#" className="hover:text-accent-gold transition-colors"><Twitter className="h-5 w-5" /></a>
                <a href="#" className="hover:text-accent-gold transition-colors"><Linkedin className="h-5 w-5" /></a>
                <a href="#" className="hover:text-accent-gold transition-colors"><Github className="h-5 w-5" /></a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800 pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-neutral-500">
          <p>© 2024 LLM Legal Advisory Council. All rights reserved.</p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#" className="hover:text-white transition-colors">이용약관</a>
            <a href="#" className="hover:text-white transition-colors">개인정보처리방침</a>
            <a href="#" className="hover:text-white transition-colors">면책조항</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;