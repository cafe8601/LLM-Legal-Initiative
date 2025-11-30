import React from 'react';
import { Users, EyeOff, BookOpen, Brain, Zap, FileSearch } from 'lucide-react';
import { Badge, Button } from '../components/UIComponents';
import { Link } from 'react-router-dom';

const Features: React.FC = () => {
  return (
    <div className="bg-neutral-50 min-h-screen pb-24">
      <div className="bg-primary-main py-20 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <Badge variant="success">Technology</Badge>
          <h1 className="text-4xl md:text-5xl font-bold mt-6 mb-6">AI 법률 자문의 새로운 기준</h1>
          <p className="text-xl text-neutral-200 max-w-2xl mx-auto">
            LLM Legal Advisory Council은 단순한 챗봇이 아닙니다. <br/>
            4개의 최정상급 AI 모델과 독자적인 검증 아키텍처가 결합된 전문가 시스템입니다.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-10">
        <div className="space-y-16">
          {/* Feature 1 */}
          <div className="bg-white rounded-2xl p-8 md:p-12 shadow-xl border border-neutral-100 flex flex-col md:flex-row gap-12 items-center">
            <div className="flex-1 order-2 md:order-1">
              <div className="bg-blue-100 w-14 h-14 rounded-lg flex items-center justify-center mb-6">
                <Users className="w-8 h-8 text-primary-main" />
              </div>
              <h2 className="text-3xl font-bold text-primary-dark mb-4">멀티 LLM Council</h2>
              <p className="text-neutral-600 mb-6 leading-relaxed">
                하나의 AI 모델에 의존하는 것은 위험합니다. 우리는 GPT-5.1, Claude 3.5 Sonnet, Gemini 1.5 Pro, Grok 4를 동시에 구동하여 '위원회'를 구성합니다. 각 모델은 모두 동일한 '법률 자문 전문가' 자격으로 참여하며, 단일 모델이 놓칠 수 있는 관점을 보완합니다.
              </p>
              <ul className="space-y-3">
                <li className="flex items-center text-sm font-medium text-neutral-700">
                  <span className="w-2 h-2 rounded-full bg-secondary-main mr-3"></span>
                  4명의 전문가에 의한 다각적 법리 해석
                </li>
                <li className="flex items-center text-sm font-medium text-neutral-700">
                  <span className="w-2 h-2 rounded-full bg-secondary-main mr-3"></span>
                  할루시네이션(환각) 교차 검증
                </li>
                <li className="flex items-center text-sm font-medium text-neutral-700">
                  <span className="w-2 h-2 rounded-full bg-secondary-main mr-3"></span>
                  독립적 의견 작성 및 상호 비평
                </li>
              </ul>
            </div>
            <div className="flex-1 order-1 md:order-2 bg-neutral-100 rounded-xl p-8 h-80 flex items-center justify-center">
               <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
                  <div className="bg-white p-4 rounded-lg shadow-sm text-center">
                     <div className="w-8 h-8 bg-blue-500 rounded-full mx-auto mb-2"></div>
                     <div className="font-bold text-sm">GPT</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow-sm text-center">
                     <div className="w-8 h-8 bg-purple-500 rounded-full mx-auto mb-2"></div>
                     <div className="font-bold text-sm">Claude</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow-sm text-center">
                     <div className="w-8 h-8 bg-teal-500 rounded-full mx-auto mb-2"></div>
                     <div className="font-bold text-sm">Gemini</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow-sm text-center">
                     <div className="w-8 h-8 bg-neutral-800 rounded-full mx-auto mb-2"></div>
                     <div className="font-bold text-sm">Grok</div>
                  </div>
               </div>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="bg-white rounded-2xl p-8 md:p-12 shadow-xl border border-neutral-100 flex flex-col md:flex-row gap-12 items-center">
            <div className="flex-1 bg-neutral-100 rounded-xl p-8 h-80 flex items-center justify-center">
               <div className="relative">
                  <div className="bg-white px-8 py-6 rounded-lg shadow-md border border-neutral-200 z-10 relative">
                     <div className="flex gap-1 mb-2">
                        <span className="w-4 h-4 text-accent-gold">★</span>
                        <span className="w-4 h-4 text-accent-gold">★</span>
                        <span className="w-4 h-4 text-accent-gold">★</span>
                        <span className="w-4 h-4 text-accent-gold">★</span>
                        <span className="w-4 h-4 text-neutral-300">★</span>
                     </div>
                     <div className="text-sm font-bold text-neutral-800">Review Score: 4.2/5.0</div>
                     <div className="text-xs text-neutral-500 mt-1">Evaluated by Anonymous Peers</div>
                  </div>
                  <div className="absolute top-4 left-4 w-full h-full bg-neutral-200 rounded-lg -z-0"></div>
               </div>
            </div>
            <div className="flex-1">
              <div className="bg-teal-100 w-14 h-14 rounded-lg flex items-center justify-center mb-6">
                <EyeOff className="w-8 h-8 text-secondary-main" />
              </div>
              <h2 className="text-3xl font-bold text-primary-dark mb-4">익명 블라인드 평가</h2>
              <p className="text-neutral-600 mb-6 leading-relaxed">
                편향을 방지하기 위해 각 AI 모델은 다른 모델이 작성한 답변을 '누가 썼는지 모르는 상태'에서 평가합니다. 논리적 완결성, 법적 근거의 적절성, 실용성을 기준으로 점수를 매기며, 점수가 낮은 답변은 최종 합성 과정에서 배제됩니다.
              </p>
              <Link to="/how-it-works">
                <Button variant="outline" size="sm">평가 알고리즘 자세히 보기</Button>
              </Link>
            </div>
          </div>

           {/* Feature 3 */}
           <div className="bg-white rounded-2xl p-8 md:p-12 shadow-xl border border-neutral-100 flex flex-col md:flex-row gap-12 items-center">
            <div className="flex-1 order-2 md:order-1">
              <div className="bg-yellow-100 w-14 h-14 rounded-lg flex items-center justify-center mb-6">
                <BookOpen className="w-8 h-8 text-accent-gold" />
              </div>
              <h2 className="text-3xl font-bold text-primary-dark mb-4">법률 RAG 시스템</h2>
              <p className="text-neutral-600 mb-6 leading-relaxed">
                일반적인 LLM은 최신 법령이나 구체적인 판례를 모를 수 있습니다. 우리의 RAG(Retrieval-Augmented Generation) 엔진은 질문이 입력되는 즉시 국가법령정보센터와 판례 DB를 검색하여 관련 문서를 AI에게 제공합니다. 이를 통해 '그럴듯한 말'이 아닌 '근거 있는 답변'을 생성합니다.
              </p>
              <div className="flex gap-2">
                 <Badge variant="neutral">대한민국 법령</Badge>
                 <Badge variant="neutral">대법원 판례</Badge>
                 <Badge variant="neutral">하급심 판례</Badge>
                 <Badge variant="neutral">법학 논문</Badge>
              </div>
            </div>
            <div className="flex-1 order-1 md:order-2 bg-neutral-100 rounded-xl p-8 h-80 flex items-center justify-center">
               <FileSearch className="w-32 h-32 text-neutral-300" />
            </div>
          </div>

        </div>
      </div>
      
      <div className="mt-24 text-center">
        <h3 className="text-2xl font-bold mb-6">지금 바로 기능을 체험해보세요</h3>
        <Link to="/contact">
           <Button variant="primary" size="lg">무료 체험 시작하기</Button>
        </Link>
      </div>
    </div>
  );
};

export default Features;