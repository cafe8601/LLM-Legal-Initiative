import React, { useState } from 'react';
import { Mail, MapPin, Phone } from 'lucide-react';
import { Button } from '../components/UIComponents';

const Contact: React.FC = () => {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    // Here you would typically handle the form submission logic
  };

  if (submitted) {
    return (
      <div className="min-h-screen bg-neutral-50 flex items-center justify-center px-4">
        <div className="bg-white p-8 rounded-2xl shadow-xl text-center max-w-md w-full animate-fade-in-up">
           <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-3xl">✅</span>
           </div>
           <h2 className="text-2xl font-bold text-primary-dark mb-4">문의가 접수되었습니다</h2>
           <p className="text-neutral-600 mb-8">
              담당자가 내용을 확인한 후 24시간 이내에 입력하신 이메일로 답변 드리겠습니다.
           </p>
           <Button variant="primary" onClick={() => setSubmitted(false)}>돌아가기</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-50 py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-primary-dark mb-4">문의하기</h1>
          <p className="text-neutral-600">
            서비스 도입, 기술 지원, 제휴 등 궁금한 점을 남겨주세요.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Contact Info */}
          <div className="bg-white p-8 rounded-2xl shadow-sm border border-neutral-200 h-fit">
            <h3 className="text-2xl font-bold mb-8">연락처 정보</h3>
            <div className="space-y-8">
               <div className="flex items-start">
                  <Mail className="w-6 h-6 text-secondary-main mt-1 mr-4" />
                  <div>
                     <div className="font-bold text-lg">Email</div>
                     <p className="text-neutral-600">support@legalcouncil.ai (일반 문의)</p>
                     <p className="text-neutral-600">sales@legalcouncil.ai (기업 도입)</p>
                  </div>
               </div>
               <div className="flex items-start">
                  <Phone className="w-6 h-6 text-secondary-main mt-1 mr-4" />
                  <div>
                     <div className="font-bold text-lg">Phone</div>
                     <p className="text-neutral-600">02-1234-5678</p>
                     <p className="text-xs text-neutral-400">평일 10:00 - 18:00 (주말/공휴일 휴무)</p>
                  </div>
               </div>
               <div className="flex items-start">
                  <MapPin className="w-6 h-6 text-secondary-main mt-1 mr-4" />
                  <div>
                     <div className="font-bold text-lg">Office</div>
                     <p className="text-neutral-600">서울특별시 강남구 테헤란로 123, AI 타워 15층</p>
                  </div>
               </div>
            </div>

            <div className="mt-12 bg-neutral-100 p-6 rounded-lg">
               <h4 className="font-bold mb-2">자주 묻는 질문</h4>
               <ul className="text-sm text-neutral-600 space-y-2 list-disc list-inside">
                  <li>API 키 발급은 어디서 하나요?</li>
                  <li>엔터프라이즈 요금제 가격은?</li>
                  <li>데이터 보안 정책 확인하기</li>
               </ul>
            </div>
          </div>

          {/* Contact Form */}
          <div className="bg-white p-8 rounded-2xl shadow-xl border border-neutral-100">
             <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                   <div>
                      <label htmlFor="name" className="block text-sm font-medium text-neutral-700 mb-1">이름</label>
                      <input type="text" id="name" required className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-secondary-main focus:border-transparent outline-none transition-all" />
                   </div>
                   <div>
                      <label htmlFor="email" className="block text-sm font-medium text-neutral-700 mb-1">이메일</label>
                      <input type="email" id="email" required className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-secondary-main focus:border-transparent outline-none transition-all" />
                   </div>
                </div>

                <div>
                   <label htmlFor="type" className="block text-sm font-medium text-neutral-700 mb-1">문의 유형</label>
                   <select id="type" className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-secondary-main focus:border-transparent outline-none transition-all">
                      <option>일반 문의</option>
                      <option>기업 요금제 도입</option>
                      <option>기술 지원</option>
                      <option>기타</option>
                   </select>
                </div>

                <div>
                   <label htmlFor="message" className="block text-sm font-medium text-neutral-700 mb-1">메시지</label>
                   <textarea id="message" rows={6} required className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-secondary-main focus:border-transparent outline-none transition-all"></textarea>
                </div>

                <div className="flex items-center">
                   <input type="checkbox" id="privacy" required className="w-4 h-4 text-secondary-main border-gray-300 rounded focus:ring-secondary-main" />
                   <label htmlFor="privacy" className="ml-2 text-sm text-neutral-600">
                      <a href="#" className="text-secondary-main hover:underline">개인정보 처리방침</a>에 동의합니다.
                   </label>
                </div>

                <Button type="submit" variant="primary" className="w-full py-3">
                   문의 보내기
                </Button>
             </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;