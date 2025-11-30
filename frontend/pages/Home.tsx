import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, Users, Search, Brain, CheckCircle, ArrowRight, Activity, Gavel, Scale, Database } from 'lucide-react';
import { Button, FeatureCard } from '../components/UIComponents';

const Home: React.FC = () => {
  return (
    <div className="overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center bg-gradient-to-br from-primary-dark via-primary-main to-neutral-900 text-white overflow-hidden">
        {/* Animated Background Pattern */}
        <div className="absolute inset-0 opacity-10 pointer-events-none">
          <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:32px_32px]"></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 relative z-10 w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-8 animate-fade-in-up">
              <div className="inline-flex items-center px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 text-secondary-light text-sm font-medium">
                <span className="w-2 h-2 rounded-full bg-accent-gold mr-2 animate-pulse"></span>
                LLM Legal Advisory Council v3.0
              </div>
              <h1 className="text-5xl lg:text-7xl font-bold font-heading leading-tight">
                AI 법률 자문<br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-secondary-main to-white">위원회의</span>
                <br />집단 지성
              </h1>
              <p className="text-lg lg:text-xl text-neutral-300 max-w-2xl leading-relaxed">
                4개의 최신 AI 모델이 당신의 법률 문제를 다각도로 분석하고, 상호 검증하며, 
                판례와 법령에 기반한 가장 정확한 자문을 제공합니다.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/consultation">
                  <Button variant="gold" size="lg" className="w-full sm:w-auto">
                    지금 상담하기
                  </Button>
                </Link>
                <Link to="/how-it-works">
                  <Button variant="outline" size="lg" className="w-full sm:w-auto text-white border-white/30 hover:bg-white/10 hover:text-white">
                    작동 원리 보기
                  </Button>
                </Link>
              </div>
              
              <div className="pt-8 flex items-center space-x-8 text-neutral-400 text-sm font-medium">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-accent-success" />
                  <span>10,000+ 상담 완료</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-accent-success" />
                  <span>95% 인용 정확도</span>
                </div>
              </div>
            </div>

            {/* Visual Orchestrator / Council Visualization */}
            <div className="relative hidden lg:block h-[500px]">
              {/* Central Table/Platform */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full border border-white/10 bg-white/5 backdrop-blur-md shadow-2xl flex items-center justify-center z-10">
                 <div className="text-center">
                    <Scale className="w-16 h-16 text-accent-gold mx-auto mb-2 opacity-80" />
                    <span className="text-xs tracking-widest text-accent-gold uppercase">Chairman</span>
                 </div>
              </div>

              {/* Orbiting AI Nodes */}
              {/* Node 1: GPT */}
              <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/5 animate-spin-slow">
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-blue-600/20 border border-blue-400/50 backdrop-blur-md flex items-center justify-center animate-float shadow-[0_0_15px_rgba(59,130,246,0.5)]">
                  <span className="text-xs font-bold text-blue-200">GPT-5</span>
                </div>
              </div>

              {/* Node 2: Claude */}
              <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/5 rotate-90">
                 <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-purple-600/20 border border-purple-400/50 backdrop-blur-md flex items-center justify-center shadow-[0_0_15px_rgba(168,85,247,0.5)]">
                  <span className="text-xs font-bold text-purple-200">Claude</span>
                </div>
              </div>

              {/* Node 3: Gemini */}
              <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/5 rotate-180">
                 <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-teal-600/20 border border-teal-400/50 backdrop-blur-md flex items-center justify-center shadow-[0_0_15px_rgba(20,184,166,0.5)]">
                  <span className="text-xs font-bold text-teal-200">Gemini</span>
                </div>
              </div>

               {/* Node 4: Grok */}
               <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/5 -rotate-90">
                 <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-neutral-600/20 border border-neutral-400/50 backdrop-blur-md flex items-center justify-center shadow-[0_0_15px_rgba(255,255,255,0.3)]">
                  <span className="text-xs font-bold text-neutral-200">Grok</span>
                </div>
              </div>
              
              {/* Connecting Lines */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-30">
                 <circle cx="50%" cy="50%" r="200" fill="none" stroke="currentColor" strokeDasharray="4 4" />
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* Trust & Stats Section */}
      <div className="bg-white border-b border-neutral-200">
         <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-2 gap-8 md:grid-cols-4 text-center">
               <div>
                  <div className="text-3xl font-bold text-primary-main mb-1">4 LLMs</div>
                  <div className="text-sm text-neutral-500 font-medium">Council Composition</div>
               </div>
               <div>
                  <div className="text-3xl font-bold text-primary-main mb-1">7 Years</div>
                  <div className="text-sm text-neutral-500 font-medium">Case Law DB</div>
               </div>
               <div>
                  <div className="text-3xl font-bold text-primary-main mb-1">Peer Review</div>
                  <div className="text-sm text-neutral-500 font-medium">Blind Validation</div>
               </div>
               <div>
                  <div className="text-3xl font-bold text-primary-main mb-1">24/7</div>
                  <div className="text-sm text-neutral-500 font-medium">Availability</div>
               </div>
            </div>
         </div>
      </div>

      {/* How It Works Preview */}
      <section className="py-24 bg-neutral-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-primary-dark mb-4">3단계 집단 지성 프로세스</h2>
            <p className="text-lg text-neutral-600">
              단일 AI의 한계를 넘어, 여러 전문가가 머리를 맞대듯 3단계의 정교한 검증 과정을 거칩니다.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connecting Line (Desktop) */}
            <div className="hidden md:block absolute top-12 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-neutral-300 to-transparent -z-10"></div>

            <div className="bg-white p-8 rounded-2xl shadow-lg border border-neutral-100 relative group hover:-translate-y-2 transition-transform duration-300">
               <div className="w-16 h-16 bg-primary-main rounded-full flex items-center justify-center text-white text-2xl font-bold mb-6 mx-auto shadow-md ring-4 ring-white">1</div>
               <h3 className="text-xl font-bold text-center mb-3">개별 의견 수집</h3>
               <p className="text-neutral-600 text-center text-sm leading-relaxed">
                 GPT, Claude, Gemini, Grok이 각자의 관점과 전문 DB를 활용하여 독립적으로 사건을 분석합니다.
               </p>
            </div>

            <div className="bg-white p-8 rounded-2xl shadow-lg border border-neutral-100 relative group hover:-translate-y-2 transition-transform duration-300">
               <div className="w-16 h-16 bg-secondary-main rounded-full flex items-center justify-center text-white text-2xl font-bold mb-6 mx-auto shadow-md ring-4 ring-white">2</div>
               <h3 className="text-xl font-bold text-center mb-3">익명 상호 평가</h3>
               <p className="text-neutral-600 text-center text-sm leading-relaxed">
                 서로의 모델명을 가린 채(Blind) 타 모델의 의견을 법률적 정확성과 논리성을 기준으로 엄격히 평가합니다.
               </p>
            </div>

            <div className="bg-white p-8 rounded-2xl shadow-lg border border-neutral-100 relative group hover:-translate-y-2 transition-transform duration-300">
               <div className="w-16 h-16 bg-accent-gold rounded-full flex items-center justify-center text-white text-2xl font-bold mb-6 mx-auto shadow-md ring-4 ring-white">3</div>
               <h3 className="text-xl font-bold text-center mb-3">의장(Chairman) 합성</h3>
               <p className="text-neutral-600 text-center text-sm leading-relaxed">
                 평가 점수가 가장 높은 의견들을 종합하여, 의장 AI가 최종적으로 권위 있는 단일 법률 메모를 작성합니다.
               </p>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-end mb-12">
             <div className="max-w-2xl">
                <h2 className="text-3xl md:text-4xl font-bold text-primary-dark mb-4">핵심 기능</h2>
                <p className="text-lg text-neutral-600">법률 자문의 새로운 기준을 제시하는 6가지 핵심 기술</p>
             </div>
             <Link to="/features" className="hidden md:flex items-center text-secondary-main font-bold hover:text-secondary-dark transition-colors mt-4 md:mt-0">
                전체 기능 보기 <ArrowRight className="ml-2 w-5 h-5" />
             </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard 
              icon={Users}
              title="멀티 LLM Council"
              description="단일 모델의 환각(Hallucination) 위험을 제거하고 다양한 법리적 해석을 제공합니다."
            />
            <FeatureCard 
              icon={Activity}
              title="블라인드 피어 리뷰"
              description="모델 간의 편향을 제거하기 위해 익명으로 상호 평가하여 최적의 답변을 선별합니다."
            />
            <FeatureCard 
              icon={Search}
              title="법률 RAG 시스템"
              description="최신 법령, 판례, 법률 문헌 DB를 실시간으로 참조하여 근거(Reference)를 명시합니다."
            />
            <FeatureCard 
              icon={Brain}
              title="3-Tier 메모리"
              description="세션 기억, 단기 기억(7일), 장기 기억(5년)을 구분하여 문맥에 맞는 연속성 있는 자문을 제공합니다."
            />
            <FeatureCard 
              icon={Gavel}
              title="의장 합성 (Synthesis)"
              description="상충되는 의견을 법률적 위계에 따라 조율하여 하나의 완결된 법률 문서로 정리합니다."
            />
            <FeatureCard 
              icon={Database}
              title="투명성 리포트"
              description="어떤 모델이 어떤 의견을 냈고, 어떤 근거로 채택되었는지 모든 의사결정 과정을 공개합니다."
            />
          </div>
          
          <div className="mt-8 text-center md:hidden">
             <Link to="/features" className="inline-flex items-center text-secondary-main font-bold hover:text-secondary-dark">
                전체 기능 보기 <ArrowRight className="ml-2 w-5 h-5" />
             </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-dark relative overflow-hidden">
         <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1589829085413-56de8ae18c73?auto=format&fit=crop&q=80&w=2000&opacity=0.1')] bg-cover bg-center opacity-10 mix-blend-overlay"></div>
         <div className="max-w-4xl mx-auto px-4 relative z-10 text-center">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">지금 바로 전문적인 자문을 받아보세요</h2>
            <p className="text-xl text-neutral-300 mb-10">
               복잡한 법률 문제, 더 이상 혼자 고민하지 마세요. <br className="hidden md:block"/>
               AI 법률 자문 위원회가 명쾌한 해답을 드립니다.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
               <Link to="/consultation">
                  <Button variant="gold" size="lg" className="w-full sm:w-auto min-w-[200px]">
                     무료로 시작하기
                  </Button>
               </Link>
               <Link to="/pricing">
                  <Button variant="outline" size="lg" className="w-full sm:w-auto border-white text-white hover:bg-white hover:text-primary-dark">
                     요금제 보기
                  </Button>
               </Link>
            </div>
         </div>
      </section>
    </div>
  );
};

export default Home;