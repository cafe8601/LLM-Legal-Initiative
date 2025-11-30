import React, { useState, useEffect, useRef } from 'react';
import { Send, Scale, MessageSquare, Users, Sparkles, Bot, ChevronRight, DollarSign, Zap, Shield, Brain, Clock, ArrowLeft, X, Info, TrendingDown } from 'lucide-react';
import { Button, Badge } from '../components/UIComponents';

// Types
interface Expert {
  provider: string;
  display_name: string;
  description: string;
  strengths: string[];
  cost_per_1k_input: number;
  cost_per_1k_output: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface SessionInfo {
  session_id: string;
  expert: string;
  expert_name: string;
  domain: string;
  message_count?: number;
  estimated_cost_krw?: number;
}

type ConsultMode = 'select' | 'council' | 'expert';

// Expert color mapping
const expertColors: Record<string, { bg: string; text: string; border: string }> = {
  openai: { bg: 'bg-blue-500', text: 'text-blue-500', border: 'border-blue-500' },
  anthropic: { bg: 'bg-purple-500', text: 'text-purple-500', border: 'border-purple-500' },
  google: { bg: 'bg-teal-500', text: 'text-teal-500', border: 'border-teal-500' },
  xai: { bg: 'bg-indigo-500', text: 'text-indigo-500', border: 'border-indigo-500' },
};

// Expert icons
const ExpertIcon: React.FC<{ provider: string; className?: string }> = ({ provider, className = '' }) => {
  const colors = expertColors[provider] || expertColors.anthropic;
  return (
    <div className={`${colors.bg} rounded-lg p-2 ${className}`}>
      <Brain className="w-5 h-5 text-white" />
    </div>
  );
};

// Mode Selection Card
const ModeCard: React.FC<{
  mode: 'council' | 'expert';
  title: string;
  description: string;
  features: string[];
  estimatedCost: string;
  savings?: string;
  recommended?: boolean;
  onClick: () => void;
}> = ({ mode, title, description, features, estimatedCost, savings, recommended, onClick }) => (
  <div
    onClick={onClick}
    className={`relative bg-white rounded-2xl border-2 p-6 cursor-pointer transition-all duration-300 hover:shadow-xl hover:-translate-y-1 ${
      recommended ? 'border-accent-gold shadow-lg' : 'border-neutral-200 hover:border-primary-main'
    }`}
  >
    {recommended && (
      <div className="absolute -top-3 left-1/2 -translate-x-1/2">
        <Badge variant="warning" className="px-3 py-1">
          <Sparkles className="w-3 h-3 mr-1" /> 추천
        </Badge>
      </div>
    )}

    <div className="flex items-center space-x-3 mb-4">
      <div className={`p-3 rounded-xl ${mode === 'council' ? 'bg-primary-main' : 'bg-secondary-main'}`}>
        {mode === 'council' ? (
          <Users className="w-6 h-6 text-white" />
        ) : (
          <MessageSquare className="w-6 h-6 text-white" />
        )}
      </div>
      <div>
        <h3 className="text-lg font-bold text-neutral-900">{title}</h3>
        <p className="text-sm text-neutral-500">{description}</p>
      </div>
    </div>

    <ul className="space-y-2 mb-4">
      {features.map((feature, idx) => (
        <li key={idx} className="flex items-center text-sm text-neutral-600">
          <ChevronRight className="w-4 h-4 text-secondary-main mr-2 flex-shrink-0" />
          {feature}
        </li>
      ))}
    </ul>

    <div className="pt-4 border-t border-neutral-100">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs text-neutral-400">예상 비용</span>
          <p className="text-lg font-bold text-neutral-900">{estimatedCost}</p>
        </div>
        {savings && (
          <Badge variant="success" className="flex items-center">
            <TrendingDown className="w-3 h-3 mr-1" />
            {savings} 절감
          </Badge>
        )}
      </div>
    </div>
  </div>
);

// Expert Selection Card
const ExpertCard: React.FC<{
  expert: Expert;
  selected: boolean;
  onClick: () => void;
}> = ({ expert, selected, onClick }) => {
  const colors = expertColors[expert.provider] || expertColors.anthropic;

  return (
    <div
      onClick={onClick}
      className={`relative bg-white rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 ${
        selected
          ? `${colors.border} shadow-lg ring-2 ring-offset-2 ${colors.border.replace('border', 'ring')}`
          : 'border-neutral-200 hover:border-neutral-300 hover:shadow-md'
      }`}
    >
      <div className="flex items-start space-x-3">
        <ExpertIcon provider={expert.provider} />
        <div className="flex-1">
          <h4 className="font-bold text-neutral-900">{expert.display_name}</h4>
          <p className="text-xs text-neutral-500 mt-1">{expert.description}</p>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1">
        {expert.strengths.slice(0, 3).map((strength, idx) => (
          <span
            key={idx}
            className={`text-xs px-2 py-0.5 rounded-full ${colors.bg}/10 ${colors.text}`}
          >
            {strength}
          </span>
        ))}
      </div>

      <div className="mt-3 pt-3 border-t border-neutral-100 flex justify-between items-center">
        <span className="text-xs text-neutral-400">비용</span>
        <span className="text-sm font-medium text-neutral-700">
          ~{Math.round((expert.cost_per_1k_input + expert.cost_per_1k_output) * 2 * 1350)}원/질문
        </span>
      </div>

      {selected && (
        <div className={`absolute top-2 right-2 w-6 h-6 rounded-full ${colors.bg} flex items-center justify-center`}>
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      )}
    </div>
  );
};

// Cost Display Component
const CostDisplay: React.FC<{ session: SessionInfo | null }> = ({ session }) => {
  if (!session) return null;

  return (
    <div className="bg-neutral-50 rounded-lg px-3 py-2 flex items-center space-x-3 text-sm">
      <DollarSign className="w-4 h-4 text-neutral-400" />
      <span className="text-neutral-600">
        예상 비용: <span className="font-medium text-neutral-900">{session.estimated_cost_krw || 0}원</span>
      </span>
      <span className="text-neutral-300">|</span>
      <span className="text-neutral-500">
        {session.message_count || 0}개 메시지
      </span>
    </div>
  );
};

const ExpertChat: React.FC = () => {
  // State
  const [mode, setMode] = useState<ConsultMode>('select');
  const [experts, setExperts] = useState<Expert[]>([]);
  const [selectedExpert, setSelectedExpert] = useState<Expert | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Mock data for experts
  useEffect(() => {
    // In production, this would fetch from /api/v1/expert-chat/experts
    setExperts([
      {
        provider: 'openai',
        display_name: 'GPT-5.1',
        description: 'OpenAI의 최신 추론 모델. 복잡한 법률 분석에 강점.',
        strengths: ['복잡한 추론', '다단계 분석', '논리적 구성'],
        cost_per_1k_input: 0.0025,
        cost_per_1k_output: 0.010,
      },
      {
        provider: 'anthropic',
        display_name: 'Claude 4.5 Sonnet',
        description: 'Anthropic의 Claude 모델. 신중하고 균형잡힌 법률 자문.',
        strengths: ['신중한 분석', '위험 평가', '윤리적 고려'],
        cost_per_1k_input: 0.003,
        cost_per_1k_output: 0.015,
      },
      {
        provider: 'google',
        display_name: 'Gemini 2.5 Pro',
        description: 'Google의 Gemini 모델. 광범위한 지식 기반 자문.',
        strengths: ['광범위한 지식', '비용 효율성', '빠른 응답'],
        cost_per_1k_input: 0.00125,
        cost_per_1k_output: 0.005,
      },
      {
        provider: 'xai',
        display_name: 'Grok 4',
        description: 'xAI의 Grok 모델. 실용적이고 직접적인 법률 조언.',
        strengths: ['실용적 조언', '명확한 설명', '직접적 답변'],
        cost_per_1k_input: 0.002,
        cost_per_1k_output: 0.010,
      },
    ]);
  }, []);

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle mode selection
  const handleModeSelect = (selectedMode: 'council' | 'expert') => {
    if (selectedMode === 'council') {
      // Navigate to council mode (existing Consultation page)
      window.location.href = '#/consultation';
    } else {
      setMode('expert');
    }
  };

  // Start expert chat session
  const startExpertSession = async () => {
    if (!selectedExpert) return;

    setIsLoading(true);

    // Mock session creation
    // In production: POST /api/v1/expert-chat/sessions
    setTimeout(() => {
      setSession({
        session_id: `session_${Date.now()}`,
        expert: selectedExpert.provider,
        expert_name: selectedExpert.display_name,
        domain: 'general_civil',
        message_count: 0,
        estimated_cost_krw: 0,
      });
      setIsLoading(false);
    }, 500);
  };

  // Send message
  const sendMessage = async () => {
    if (!inputValue.trim() || !session || isStreaming) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsStreaming(true);

    // Mock streaming response
    // In production: POST /api/v1/expert-chat/sessions/{session_id}/chat with stream=true
    const mockResponse = generateMockResponse(userMessage.content, selectedExpert?.display_name || 'AI');

    let currentContent = '';
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, assistantMessage]);

    // Simulate streaming
    for (let i = 0; i < mockResponse.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 20));
      currentContent += mockResponse[i];
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...assistantMessage,
          content: currentContent,
        };
        return newMessages;
      });
    }

    setIsStreaming(false);

    // Update session cost
    setSession(prev => prev ? {
      ...prev,
      message_count: (prev.message_count || 0) + 2,
      estimated_cost_krw: (prev.estimated_cost_krw || 0) + Math.round(Math.random() * 50 + 20),
    } : null);
  };

  const generateMockResponse = (query: string, expertName: string): string => {
    const responses = [
      `안녕하세요, ${expertName}입니다. 질문하신 내용에 대해 법률적 관점에서 답변드리겠습니다.\n\n귀하의 질문 "${query.substring(0, 50)}..."에 대해 검토해 보았습니다.\n\n**법률적 분석**\n\n본 사안은 민법상 채권-채무 관계에 해당할 수 있습니다. 구체적인 상황에 따라 다르지만, 일반적으로 다음과 같은 법적 근거가 적용될 수 있습니다:\n\n1. **민법 제390조 (채무불이행과 손해배상)**: 채무자가 채무의 내용에 좇은 이행을 하지 아니한 때에는 채권자는 손해배상을 청구할 수 있습니다.\n\n2. **증거 확보의 중요성**: 계약서, 이메일, 카카오톡 대화 등 모든 관련 자료를 보관하시기 바랍니다.\n\n**권고사항**\n\n- 내용증명을 발송하여 이행을 최고하시는 것이 좋습니다.\n- 소멸시효가 지나지 않았는지 확인이 필요합니다.\n\n더 구체적인 상담이 필요하시면 추가 질문해 주세요.`,

      `${expertName}의 법률 자문입니다.\n\n문의하신 "${query.substring(0, 30)}..." 관련 사안을 검토했습니다.\n\n**핵심 쟁점**\n\n이 문제의 핵심은 당사자 간의 법률관계를 명확히 하는 것입니다. 현행법상 다음 사항들을 고려해야 합니다:\n\n1. 계약의 성립 여부 및 유효성\n2. 이행기의 도래 여부\n3. 상대방의 귀책사유 존재 여부\n\n**대응 방안**\n\n단계별로 접근하시는 것을 권고드립니다:\n\n1단계: 서면으로 이행 최고 (내용증명)\n2단계: 조정 또는 중재 시도\n3단계: 필요시 법적 조치 검토\n\n추가 질문이 있으시면 말씀해 주세요.`,
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetToModeSelect = () => {
    setMode('select');
    setSelectedExpert(null);
    setSession(null);
    setMessages([]);
  };

  // Render Mode Selection
  if (mode === 'select') {
    return (
      <div className="min-h-screen bg-neutral-50 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center p-4 bg-primary-main rounded-2xl shadow-xl mb-6">
              <Scale className="w-10 h-10 text-accent-gold" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
              어떤 방식으로 상담받으시겠어요?
            </h1>
            <p className="text-neutral-500 max-w-xl mx-auto">
              상담 목적과 예산에 맞는 최적의 모드를 선택하세요.
              <br />간단한 질문은 전문가 모드, 중요한 사안은 위원회 모드를 추천드립니다.
            </p>
          </div>

          {/* Mode Cards */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <ModeCard
              mode="council"
              title="AI 법률 자문 위원회"
              description="4개 AI가 협력하여 검증된 답변 제공"
              features={[
                '4개 AI 모델의 다양한 관점',
                '블라인드 피어리뷰로 품질 보증',
                '의장 AI의 종합 답변',
                '높은 정확도와 신뢰성',
              ]}
              estimatedCost="~300원/질문"
              recommended
              onClick={() => handleModeSelect('council')}
            />

            <ModeCard
              mode="expert"
              title="단일 전문가 채팅"
              description="선택한 AI 전문가와 1:1 대화"
              features={[
                '빠른 실시간 응답',
                '자연스러운 대화 흐름',
                '전문가 직접 선택 가능',
                '비용 효율적',
              ]}
              estimatedCost="~50원/질문"
              savings="80%"
              onClick={() => handleModeSelect('expert')}
            />
          </div>

          {/* Info Box */}
          <div className="bg-primary-main/5 border border-primary-main/20 rounded-xl p-4 flex items-start space-x-3">
            <Info className="w-5 h-5 text-primary-main flex-shrink-0 mt-0.5" />
            <div className="text-sm text-neutral-600">
              <p className="font-medium text-neutral-800 mb-1">어떤 모드를 선택해야 할까요?</p>
              <ul className="space-y-1">
                <li>• <strong>위원회 모드</strong>: 분쟁, 계약 검토, 중요한 법률 사안에 적합</li>
                <li>• <strong>전문가 모드</strong>: 일반 질문, 법률 용어 설명, 간단한 절차 안내에 적합</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render Expert Selection (before session starts)
  if (mode === 'expert' && !session) {
    return (
      <div className="min-h-screen bg-neutral-50 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Back Button */}
          <button
            onClick={resetToModeSelect}
            className="flex items-center text-neutral-500 hover:text-neutral-700 mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            모드 선택으로 돌아가기
          </button>

          {/* Header */}
          <div className="text-center mb-10">
            <h1 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-3">
              어떤 전문가와 상담하시겠어요?
            </h1>
            <p className="text-neutral-500">
              각 AI 전문가는 고유한 강점을 가지고 있습니다.
            </p>
          </div>

          {/* Expert Grid */}
          <div className="grid md:grid-cols-2 gap-4 mb-8">
            {experts.map((expert) => (
              <ExpertCard
                key={expert.provider}
                expert={expert}
                selected={selectedExpert?.provider === expert.provider}
                onClick={() => setSelectedExpert(expert)}
              />
            ))}
          </div>

          {/* Start Button */}
          <div className="flex justify-center">
            <Button
              variant="gold"
              size="lg"
              disabled={!selectedExpert || isLoading}
              onClick={startExpertSession}
              className="px-12"
            >
              {isLoading ? (
                <>
                  <Clock className="w-4 h-4 mr-2 animate-spin" />
                  세션 시작 중...
                </>
              ) : (
                <>
                  {selectedExpert?.display_name || '전문가'}와 상담 시작
                  <ChevronRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Render Chat Interface
  return (
    <div className="min-h-screen bg-neutral-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-neutral-200 px-4 py-3 sticky top-16 z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <button
              onClick={resetToModeSelect}
              className="p-2 hover:bg-neutral-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-neutral-500" />
            </button>
            <ExpertIcon provider={selectedExpert?.provider || 'anthropic'} />
            <div>
              <h2 className="font-bold text-neutral-900">{session?.expert_name}</h2>
              <p className="text-xs text-neutral-500">단일 전문가 채팅</p>
            </div>
          </div>
          <CostDisplay session={session} />
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <ExpertIcon provider={selectedExpert?.provider || 'anthropic'} className="mx-auto mb-4 w-16 h-16 rounded-2xl" />
              <h3 className="text-lg font-bold text-neutral-800 mb-2">
                {session?.expert_name}와의 상담을 시작하세요
              </h3>
              <p className="text-neutral-500 mb-6">
                법률 관련 질문을 자유롭게 입력해 주세요.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {['계약서 검토해 주세요', '손해배상 청구 방법', '임대차 분쟁 해결'].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInputValue(suggestion)}
                    className="px-4 py-2 bg-white border border-neutral-200 rounded-full text-sm text-neutral-600 hover:border-primary-main hover:text-primary-main transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-primary-main text-white rounded-tr-sm'
                    : 'bg-white border border-neutral-200 rounded-tl-sm shadow-sm'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="flex items-center space-x-2 mb-2 pb-2 border-b border-neutral-100">
                    <ExpertIcon provider={selectedExpert?.provider || 'anthropic'} className="w-6 h-6 p-1" />
                    <span className="text-xs font-medium text-neutral-500">{session?.expert_name}</span>
                  </div>
                )}
                <div className={`whitespace-pre-wrap text-sm leading-relaxed ${
                  message.role === 'user' ? '' : 'text-neutral-700'
                }`}>
                  {message.content}
                  {isStreaming && idx === messages.length - 1 && message.role === 'assistant' && (
                    <span className="inline-block w-2 h-4 bg-primary-main/50 ml-1 animate-pulse" />
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-neutral-200 px-4 py-4">
        <div className="max-w-4xl mx-auto flex items-end space-x-3">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="법률 관련 질문을 입력하세요..."
            className="flex-1 resize-none border border-neutral-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary-main focus:border-transparent outline-none min-h-[52px] max-h-32"
            rows={1}
          />
          <Button
            variant="gold"
            onClick={sendMessage}
            disabled={!inputValue.trim() || isStreaming}
            className="h-[52px] px-6"
          >
            <Send className="w-5 h-5" />
          </Button>
        </div>
        <p className="text-xs text-neutral-400 text-center mt-2">
          AI 법률 자문은 참고용이며, 정확한 법률 조언을 위해서는 전문 변호사와 상담하세요.
        </p>
      </div>
    </div>
  );
};

export default ExpertChat;
