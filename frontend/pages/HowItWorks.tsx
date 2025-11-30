import React from 'react';
import { Button } from '../components/UIComponents';
import { Link } from 'react-router-dom';

const HowItWorks: React.FC = () => {
  return (
    <div className="bg-white min-h-screen">
      <section className="bg-neutral-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">작동 원리 (Workflow)</h1>
          <p className="text-xl text-neutral-400 max-w-3xl mx-auto">
            당신의 질문이 입력되면 0.1초 만에 위원회가 소집됩니다. <br/>
            복잡해 보이는 내부 과정을 투명하게 공개합니다.
          </p>
        </div>
      </section>

      <section className="max-w-5xl mx-auto px-4 py-20">
        <div className="space-y-24 relative">
          {/* Vertical Line */}
          <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-neutral-200 -z-10 transform -translate-x-1/2"></div>

          {/* Step 1 */}
          <div className="relative flex flex-col md:flex-row items-center justify-between gap-8">
             <div className="md:w-5/12 text-right order-2 md:order-1">
                <h3 className="text-2xl font-bold text-primary-main mb-2">Step 1. 안건 상정 및 검색</h3>
                <p className="text-neutral-600">
                   사용자가 법률 질문을 입력하면, AI가 핵심 쟁점을 파악(Issue Spotting)하고 
                   관련된 법령과 판례를 RAG 엔진을 통해 실시간으로 검색합니다.
                </p>
             </div>
             <div className="w-12 h-12 bg-primary-main rounded-full flex items-center justify-center text-white font-bold text-xl border-4 border-white shadow-lg z-10 order-1 md:order-2">1</div>
             <div className="md:w-5/12 order-3 bg-neutral-50 p-6 rounded-lg border border-neutral-100">
                <div className="text-xs text-neutral-400 mb-2">SYSTEM LOG</div>
                <div className="font-mono text-xs text-green-600">
                   {'>'}Query received<br/>
                   {'>'}Extracting keywords: "임대차 계약", "묵시적 갱신"<br/>
                   {'>'}RAG Searching... found 12 precedents.
                </div>
             </div>
          </div>

          {/* Step 2 */}
          <div className="relative flex flex-col md:flex-row items-center justify-between gap-8">
             <div className="md:w-5/12 order-3 md:order-1 bg-neutral-50 p-6 rounded-lg border border-neutral-100">
                <div className="grid grid-cols-2 gap-2">
                   <div className="bg-white p-2 text-center text-xs rounded shadow-sm border">GPT-5: 의견 작성 중...</div>
                   <div className="bg-white p-2 text-center text-xs rounded shadow-sm border">Claude: 의견 작성 중...</div>
                   <div className="bg-white p-2 text-center text-xs rounded shadow-sm border">Gemini: 의견 작성 중...</div>
                   <div className="bg-white p-2 text-center text-xs rounded shadow-sm border">Grok: 의견 작성 중...</div>
                </div>
             </div>
             <div className="w-12 h-12 bg-secondary-main rounded-full flex items-center justify-center text-white font-bold text-xl border-4 border-white shadow-lg z-10 order-1 md:order-2">2</div>
             <div className="md:w-5/12 order-2 md:order-3">
                <h3 className="text-2xl font-bold text-primary-main mb-2">Step 2. 독립적 의견 제시</h3>
                <p className="text-neutral-600">
                   4개의 모델이 검색된 자료를 바탕으로 각자의 법적 견해를 작성합니다. 
                   이때 서로의 의견을 볼 수 없으며(Blind), 독립적으로 판단합니다.
                </p>
             </div>
          </div>

          {/* Step 3 */}
          <div className="relative flex flex-col md:flex-row items-center justify-between gap-8">
             <div className="md:w-5/12 text-right order-2 md:order-1">
                <h3 className="text-2xl font-bold text-primary-main mb-2">Step 3. 상호 검증 (Peer Review)</h3>
                <p className="text-neutral-600">
                   작성된 4개의 의견을 서로 교차하여 평가합니다. 논리적 비약은 없는지, 
                   인용된 판례가 적절한지를 비판적으로 검토하고 점수(0~100점)를 부여합니다.
                </p>
             </div>
             <div className="w-12 h-12 bg-accent-gold rounded-full flex items-center justify-center text-white font-bold text-xl border-4 border-white shadow-lg z-10 order-1 md:order-2">3</div>
             <div className="md:w-5/12 order-3 bg-neutral-50 p-6 rounded-lg border border-neutral-100 flex items-center justify-center">
                 <div className="w-full bg-white rounded h-2 mb-2 overflow-hidden">
                    <div className="bg-blue-500 h-full w-3/4"></div>
                 </div>
             </div>
          </div>

          {/* Step 4 */}
          <div className="relative flex flex-col md:flex-row items-center justify-between gap-8">
             <div className="md:w-5/12 order-3 md:order-1 bg-neutral-50 p-6 rounded-lg border border-neutral-100 shadow-sm border-l-4 border-l-primary-main">
                <div className="text-xs text-neutral-400 mb-1">FINAL OUTPUT</div>
                <div className="font-serif text-sm text-neutral-800">
                   "검토 결과, 귀하의 경우 주택임대차보호법 제6조에 따라 묵시적 갱신이 인정될 소지가 높습니다. 다만..."
                </div>
             </div>
             <div className="w-12 h-12 bg-primary-dark rounded-full flex items-center justify-center text-white font-bold text-xl border-4 border-white shadow-lg z-10 order-1 md:order-2">4</div>
             <div className="md:w-5/12 order-2 md:order-3">
                <h3 className="text-2xl font-bold text-primary-main mb-2">Step 4. 의장 합성 및 최종 전달</h3>
                <p className="text-neutral-600">
                   '의장(Chairman)' 역할을 맡은 모델이 높은 점수를 받은 의견들을 통합하여, 
                   사용자가 이해하기 쉬운 최종 법률 메모 형태로 정리해 전달합니다.
                </p>
             </div>
          </div>

        </div>

        <div className="mt-24 text-center">
           <Link to="/contact">
             <Button variant="gold" size="lg">내 사건 분석 요청하기</Button>
           </Link>
        </div>
      </section>
    </div>
  );
};

export default HowItWorks;