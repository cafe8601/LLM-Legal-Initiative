import React from 'react';
import { Check, X } from 'lucide-react';
import { Button, Badge } from '../components/UIComponents';
import { PricingTier } from '../types';

const tiers: PricingTier[] = [
  {
    name: "Basic",
    price: "₩0",
    target: "개인 / 초기 검토",
    features: [
      "월 3회 무료 상담",
      "기본 RAG 검색 (법령 한정)",
      "4-LLM Council 요약본 제공",
      "세션 단위 메모리",
      "커뮤니티 지원"
    ],
    cta: "무료로 시작",
    recommended: false
  },
  {
    name: "Professional",
    price: "₩99,000",
    target: "스타트업 / 중소기업",
    features: [
      "무제한 상담",
      "전체 RAG DB (판례, 논문 포함)",
      "상세 분석 리포트 & 원본 의견 열람",
      "단기/장기 메모리 (히스토리 기억)",
      "우선 이메일 지원",
      "법률 문서 초안 작성 기능"
    ],
    cta: "14일 무료 체험",
    recommended: true
  },
  {
    name: "Enterprise",
    price: "문의",
    target: "로펌 / 대기업",
    features: [
      "맞춤형 전용 인스턴스",
      "사내 데이터(Private RAG) 연동",
      "API 액세스 & SSO 통합",
      "SLA 보장 (99.9%)",
      "전담 어카운트 매니저",
      "온프레미스 설치 옵션"
    ],
    cta: "영업팀 문의",
    recommended: false
  }
];

const Pricing: React.FC = () => {
  return (
    <div className="bg-neutral-50 min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h1 className="text-4xl font-bold text-primary-dark mb-4">합리적인 비용으로<br/>최고의 법률 자문을 경험하세요</h1>
          <p className="text-xl text-neutral-600">
            변호사 상담 비용의 1/10 수준으로 24시간 언제든 전문가 수준의 법률 검토를 받을 수 있습니다.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {tiers.map((tier) => (
            <div 
              key={tier.name}
              className={`bg-white rounded-2xl p-8 border ${
                tier.recommended 
                  ? 'border-secondary-main shadow-xl ring-2 ring-secondary-main ring-opacity-50 relative' 
                  : 'border-neutral-200 shadow-sm'
              } flex flex-col`}
            >
              {tier.recommended && (
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <span className="bg-secondary-main text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide">
                    Most Popular
                  </span>
                </div>
              )}
              
              <div className="mb-8">
                <h3 className="text-lg font-medium text-neutral-500 mb-2">{tier.name}</h3>
                <div className="flex items-baseline gap-1">
                  <span className="text-4xl font-bold text-primary-dark">{tier.price}</span>
                  {tier.price !== "문의" && <span className="text-neutral-400">/월</span>}
                </div>
                <p className="text-sm text-neutral-400 mt-2">{tier.target}</p>
              </div>

              <ul className="space-y-4 mb-8 flex-1">
                {tier.features.map((feature, i) => (
                  <li key={i} className="flex items-start">
                    <Check className="w-5 h-5 text-secondary-main mr-3 flex-shrink-0" />
                    <span className="text-neutral-600 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button 
                variant={tier.recommended ? 'primary' : 'outline'} 
                className="w-full"
              >
                {tier.cta}
              </Button>
            </div>
          ))}
        </div>

        <div className="mt-16 bg-white rounded-xl p-8 border border-neutral-200">
          <h3 className="text-xl font-bold mb-4">자주 묻는 질문</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
             <div>
                <h4 className="font-bold text-primary-main mb-2">Q. 환불 정책은 어떻게 되나요?</h4>
                <p className="text-neutral-600 text-sm">서비스 불만족 시 7일 이내 전액 환불을 보장합니다. 단, 상담 사용량이 10회 미만인 경우에 한합니다.</p>
             </div>
             <div>
                <h4 className="font-bold text-primary-main mb-2">Q. 법적 효력이 있나요?</h4>
                <p className="text-neutral-600 text-sm">본 서비스는 AI가 제공하는 참고용 자문이며, 변호사법상 법률 감정이나 대리가 아닙니다. 중요한 법적 결정 전에는 반드시 인간 변호사의 확인을 거치시기 바랍니다.</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pricing;