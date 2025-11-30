import React from 'react';
import { Button } from '../components/UIComponents';
import { Link } from 'react-router-dom';

const About: React.FC = () => {
  return (
    <div className="bg-white min-h-screen">
      <div className="relative h-[400px] bg-primary-dark flex items-center justify-center overflow-hidden">
         <div className="absolute inset-0 bg-neutral-900 opacity-50 z-0"></div>
         <div className="relative z-10 text-center px-4">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">법률 자문의<br/>민주화를 꿈꿉니다</h1>
            <p className="text-xl text-neutral-300 max-w-2xl mx-auto">
               누구나 쉽고, 빠르고, 정확하게 법률 전문가의 도움을 받을 수 있는 세상을 만듭니다.
            </p>
         </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
         <div className="prose prose-lg mx-auto text-neutral-600">
            <h2 className="text-3xl font-bold text-primary-dark mb-6">우리의 미션</h2>
            <p className="mb-8">
               법은 우리 사회를 지탱하는 약속이지만, 그 내용은 너무나 복잡하고 접근하기 어렵습니다. 
               높은 비용과 심리적 장벽으로 인해 많은 개인과 소규모 기업들이 정당한 권리를 포기하거나 
               불리한 계약을 체결하곤 합니다.
            </p>
            <p className="mb-12">
               LLM Legal Advisory Council은 최첨단 AI 기술을 활용하여 이러한 정보 불균형을 해소하고자 합니다. 
               우리는 AI가 인간 변호사를 대체하는 것이 아니라, 법률 서비스의 효율성을 극대화하여 
               더 많은 사람이 법의 보호를 받을 수 있도록 돕는 도구가 되어야 한다고 믿습니다.
            </p>

            <h2 className="text-3xl font-bold text-primary-dark mb-6">왜 '위원회(Council)'인가요?</h2>
            <p className="mb-8">
               "집단 지성은 한 명의 천재보다 똑똑하다."<br/>
               현재의 AI 모델들은 놀라울 정도로 똑똑하지만, 여전히 완벽하지 않습니다. 때로는 없는 사실을 지어내기도 하고(Hallucination), 
               편향된 시각을 가질 수도 있습니다.
            </p>
            <p className="mb-8">
               우리는 이 문제를 해결하기 위해 단일 모델 대신 <strong>GPT, Claude, Gemini, Grok</strong>이라는 
               서로 다른 성격의 4가지 최상위 모델을 하나의 테이블에 앉혔습니다. 이들은 서로 토론하고, 
               비판하고, 검증하며 인간 전문가 그룹처럼 작동합니다. 이것이 우리가 높은 정확도와 신뢰성을 
               자랑할 수 있는 이유입니다.
            </p>
         </div>

         <div className="mt-16 border-t border-neutral-200 pt-16 text-center">
            <h3 className="text-2xl font-bold mb-6">함께 혁신을 만들어가세요</h3>
            <p className="text-neutral-600 mb-8">
               우리는 법률 테크(LegalTech)의 최전선에서 끊임없이 도전하고 있습니다.
            </p>
            <div className="flex justify-center gap-4">
               <Link to="/contact">
                  <Button variant="primary">제휴/투자 문의</Button>
               </Link>
               <Button variant="outline">채용 공고 보기</Button>
            </div>
         </div>
      </div>
    </div>
  );
};

export default About;