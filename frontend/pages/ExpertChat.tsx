import React, { useState, useEffect, useRef } from 'react';
import { Send, Scale, MessageSquare, Users, Sparkles, Bot, ChevronRight, DollarSign, Zap, Shield, Brain, Clock, ArrowLeft, X, Info, TrendingDown, RefreshCw, FileText, Briefcase, Copyright, Building2, Gavel, CheckCircle } from 'lucide-react';
import { Button, Badge } from '../components/UIComponents';

// API Base URL
const API_BASE = '/api/v1';

// Types
interface Provider {
  id: string;
  name: string;
  description: string;
  models: {
    drafter: string;
    verifier: string;
  };
}

interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  provider?: string;
  timestamp: Date;
  tokens_used?: number;
  cascade_tier?: string;
}

interface SessionInfo {
  session_id: string;
  user_id: string;
  provider: string;
  provider_name: string;
  domain: string;
  message_count: number;
  total_tokens?: number;
  total_cost?: number;
}

type ConsultMode = 'select' | 'council' | 'chat' | 'domain';

// Legal domains
interface LegalDomainInfo {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  examples: string[];
}

const legalDomains: LegalDomainInfo[] = [
  {
    id: 'general_civil',
    name: '일반/민사',
    description: '일반적인 민사 문제, 손해배상, 채권채무 관계',
    icon: <Scale className="w-5 h-5" />,
    examples: ['손해배상 청구', '채권 추심', '부당이득 반환'],
  },
  {
    id: 'contract',
    name: '계약 검토',
    description: '계약서 작성, 검토, 해석 및 분쟁',
    icon: <FileText className="w-5 h-5" />,
    examples: ['계약서 검토', '계약 해지', '위약금 분쟁'],
  },
  {
    id: 'ip',
    name: '지식재산권',
    description: '특허, 상표, 저작권 등 지식재산 관련',
    icon: <Copyright className="w-5 h-5" />,
    examples: ['상표 출원', '저작권 침해', '특허 분쟁'],
  },
  {
    id: 'labor',
    name: '노무/인사',
    description: '근로계약, 임금, 해고, 산업재해 등',
    icon: <Building2 className="w-5 h-5" />,
    examples: ['부당해고', '임금체불', '퇴직금 계산'],
  },
  {
    id: 'criminal',
    name: '형사/고소',
    description: '형사 고소, 고발, 형사 피해 구제',
    icon: <Gavel className="w-5 h-5" />,
    examples: ['사기 고소', '명예훼손', '폭행/상해'],
  },
];

// Provider color mapping
const providerColors: Record<string, { bg: string; text: string; border: string; light: string }> = {
  claude: { bg: 'bg-purple-500', text: 'text-purple-500', border: 'border-purple-500', light: 'bg-purple-50' },
  gpt: { bg: 'bg-emerald-500', text: 'text-emerald-500', border: 'border-emerald-500', light: 'bg-emerald-50' },
  gemini: { bg: 'bg-blue-500', text: 'text-blue-500', border: 'border-blue-500', light: 'bg-blue-50' },
  grok: { bg: 'bg-orange-500', text: 'text-orange-500', border: 'border-orange-500', light: 'bg-orange-50' },
};

// Provider Icon
const ProviderIcon: React.FC<{ provider: string; className?: string }> = ({ provider, className = '' }) => {
  const colors = providerColors[provider] || providerColors.claude;
  return (
    <div className={`${colors.bg} rounded-lg p-2 ${className}`}>
      <Brain className="w-5 h-5 text-white" />
    </div>
  );
};

// Mode Selection Card
const ModeCard: React.FC<{
  mode: 'council' | 'chat';
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

// Domain Selection Card
const DomainCard: React.FC<{
  domain: LegalDomainInfo;
  selected: boolean;
  onClick: () => void;
}> = ({ domain, selected, onClick }) => (
  <div
    onClick={onClick}
    className={`relative bg-white rounded-xl border-2 p-4 cursor-pointer transition-all duration-200 ${
      selected
        ? 'border-primary-main shadow-lg ring-2 ring-offset-2 ring-primary-main'
        : 'border-neutral-200 hover:border-neutral-300 hover:shadow-md'
    }`}
  >
    <div className="flex items-start space-x-3">
      <div className={`p-2 rounded-lg ${selected ? 'bg-primary-main text-white' : 'bg-neutral-100 text-neutral-600'}`}>
        {domain.icon}
      </div>
      <div className="flex-1">
        <h4 className="font-bold text-neutral-900">{domain.name}</h4>
        <p className="text-xs text-neutral-500 mt-1">{domain.description}</p>
      </div>
    </div>

    <div className="mt-3 pt-3 border-t border-neutral-100">
      <div className="flex flex-wrap gap-1">
        {domain.examples.map((example, idx) => (
          <span key={idx} className="text-xs bg-neutral-100 text-neutral-600 px-2 py-0.5 rounded">
            {example}
          </span>
        ))}
      </div>
    </div>

    {selected && (
      <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-primary-main flex items-center justify-center">
        <CheckCircle className="w-4 h-4 text-white" />
      </div>
    )}
  </div>
);

// Provider Selection Card
const ProviderCard: React.FC<{
  provider: Provider;
  selected: boolean;
  onClick: () => void;
}> = ({ provider, selected, onClick }) => {
  const colors = providerColors[provider.id] || providerColors.claude;

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
        <ProviderIcon provider={provider.id} />
        <div className="flex-1">
          <h4 className="font-bold text-neutral-900">{provider.name}</h4>
          <p className="text-xs text-neutral-500 mt-1">{provider.description}</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-neutral-100">
        <div className="flex justify-between items-center text-xs">
          <span className="text-neutral-400">CascadeFlow</span>
          <div className="flex items-center space-x-1">
            <Zap className="w-3 h-3 text-yellow-500" />
            <span className="text-neutral-600">비용 최적화</span>
          </div>
        </div>
        <div className="mt-2 text-xs text-neutral-500">
          <span className="font-mono bg-neutral-100 px-1 rounded">{provider.models.drafter.split('/')[1]}</span>
          <span className="mx-1">→</span>
          <span className="font-mono bg-neutral-100 px-1 rounded">{provider.models.verifier.split('/')[1]}</span>
        </div>
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

  const costKrw = Math.round((session.total_cost || 0) * 1350);

  return (
    <div className="bg-neutral-50 rounded-lg px-3 py-2 flex items-center space-x-3 text-sm">
      <DollarSign className="w-4 h-4 text-neutral-400" />
      <span className="text-neutral-600">
        비용: <span className="font-medium text-neutral-900">{costKrw}원</span>
      </span>
      <span className="text-neutral-300">|</span>
      <span className="text-neutral-500">
        {session.total_tokens || 0} 토큰
      </span>
    </div>
  );
};

// Cascade Tier Badge
const CascadeTierBadge: React.FC<{ tier?: string }> = ({ tier }) => {
  if (!tier) return null;

  const isVerifier = tier === 'verifier';
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
      isVerifier ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'
    }`}>
      {isVerifier ? '검증됨' : '빠른응답'}
    </span>
  );
};

const ExpertChat: React.FC = () => {
  // State
  const [mode, setMode] = useState<ConsultMode>('select');
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null);
  const [selectedDomain, setSelectedDomain] = useState<string>('general_civil');
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Fetch providers on mount
  useEffect(() => {
    fetchProviders();
  }, []);

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchProviders = async () => {
    try {
      const response = await fetch(`${API_BASE}/chat/providers`);
      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers);
      } else {
        // Fallback to default providers if API fails
        setProviders([
          {
            id: 'claude',
            name: 'Claude (Anthropic)',
            description: '정교한 법률 분석과 논리적 추론에 강점',
            models: {
              drafter: 'anthropic/claude-haiku-4',
              verifier: 'anthropic/claude-sonnet-4',
            },
          },
          {
            id: 'gpt',
            name: 'GPT (OpenAI)',
            description: '다양한 법률 지식과 유연한 응답 생성',
            models: {
              drafter: 'openai/gpt-4o-mini',
              verifier: 'openai/gpt-5.1',
            },
          },
          {
            id: 'gemini',
            name: 'Gemini (Google)',
            description: '최신 정보 통합과 빠른 응답 속도',
            models: {
              drafter: 'google/gemini-flash-latest',
              verifier: 'google/gemini-3-pro-preview',
            },
          },
          {
            id: 'grok',
            name: 'Grok (xAI)',
            description: '실시간 정보 접근과 실용적 조언',
            models: {
              drafter: 'x-ai/grok-4-fast',
              verifier: 'x-ai/grok-4.1',
            },
          },
        ]);
      }
    } catch (err) {
      console.error('Failed to fetch providers:', err);
    }
  };

  // Handle mode selection
  const handleModeSelect = (selectedMode: 'council' | 'chat') => {
    if (selectedMode === 'council') {
      // Navigate to council consultation page (existing functionality)
      window.location.href = '#/council-consultation';
    } else {
      // Go to domain selection for individual chat
      setMode('domain');
    }
  };

  // Handle domain selection - proceed to provider selection
  const handleDomainSelect = () => {
    setMode('chat');
  };

  // Create chat session
  const createSession = async () => {
    if (!selectedProvider) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/chat/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider.id,
          domain: selectedDomain,
        }),
      });

      if (!response.ok) {
        throw new Error('세션 생성에 실패했습니다.');
      }

      const data = await response.json();
      setSession(data);
    } catch (err) {
      // Fallback: Create mock session for demo
      const domainInfo = legalDomains.find(d => d.id === selectedDomain);
      setSession({
        session_id: `session_${Date.now()}`,
        user_id: 'anonymous',
        provider: selectedProvider.id,
        provider_name: selectedProvider.name,
        domain: selectedDomain,
        message_count: 0,
        total_tokens: 0,
        total_cost: 0,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Send message with streaming
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
    setError(null);

    try {
      // Try streaming API first
      const response = await fetch(`${API_BASE}/chat/sessions/${session.session_id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: userMessage.content,
          stream: true,
        }),
      });

      if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
        // Handle SSE streaming
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: '',
          model: selectedProvider?.models.drafter,
          provider: selectedProvider?.id,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);

        let fullContent = '';

        while (reader) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              if (data.startsWith('[ERROR]')) {
                setError(data.replace('[ERROR] ', ''));
                continue;
              }
              fullContent += data;
              setMessages(prev => {
                const newMessages = [...prev];
                newMessages[newMessages.length - 1] = {
                  ...assistantMessage,
                  content: fullContent,
                };
                return newMessages;
              });
            }
          }
        }

        // Update session stats
        setSession(prev => prev ? {
          ...prev,
          message_count: (prev.message_count || 0) + 2,
          total_tokens: (prev.total_tokens || 0) + fullContent.length,
          total_cost: (prev.total_cost || 0) + 0.001,
        } : null);

      } else {
        // Fallback to mock response
        await handleMockResponse(userMessage.content);
      }
    } catch (err) {
      console.error('Send message error:', err);
      // Fallback to mock response on error
      await handleMockResponse(userMessage.content);
    } finally {
      setIsStreaming(false);
    }
  };

  // Mock response handler for demo
  const handleMockResponse = async (query: string) => {
    const providerName = selectedProvider?.name || 'AI 전문가';
    const mockResponse = generateMockResponse(query, providerName);

    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      model: selectedProvider?.models.drafter,
      provider: selectedProvider?.id,
      timestamp: new Date(),
      cascade_tier: Math.random() > 0.7 ? 'verifier' : 'drafter',
    };
    setMessages(prev => [...prev, assistantMessage]);

    // Simulate streaming
    let currentContent = '';
    for (let i = 0; i < mockResponse.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 15));
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

    // Update session
    setSession(prev => prev ? {
      ...prev,
      message_count: (prev.message_count || 0) + 2,
      total_tokens: (prev.total_tokens || 0) + mockResponse.length,
      total_cost: (prev.total_cost || 0) + 0.001,
    } : null);
  };

  const generateMockResponse = (query: string, providerName: string): string => {
    return `안녕하세요, ${providerName}입니다. 질문하신 내용에 대해 법률적 관점에서 답변드리겠습니다.

귀하의 질문 "${query.substring(0, 50)}${query.length > 50 ? '...' : ''}"에 대해 검토해 보았습니다.

**법률적 분석**

본 사안은 민법상 채권-채무 관계에 해당할 수 있습니다. 구체적인 상황에 따라 다르지만, 일반적으로 다음과 같은 법적 근거가 적용될 수 있습니다:

1. **민법 제390조 (채무불이행과 손해배상)**: 채무자가 채무의 내용에 좇은 이행을 하지 아니한 때에는 채권자는 손해배상을 청구할 수 있습니다.

2. **증거 확보의 중요성**: 계약서, 이메일, 카카오톡 대화 등 모든 관련 자료를 보관하시기 바랍니다.

**권고사항**

- 내용증명을 발송하여 이행을 최고하시는 것이 좋습니다.
- 소멸시효가 지나지 않았는지 확인이 필요합니다.

더 구체적인 상담이 필요하시면 추가 질문해 주세요.

---
*이 응답은 CascadeFlow 시스템을 통해 비용 최적화되었습니다.*`;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetToModeSelect = () => {
    setMode('select');
    setSelectedProvider(null);
    setSelectedDomain('general_civil');
    setSession(null);
    setMessages([]);
    setError(null);
  };

  const goBackToDomainSelect = () => {
    setMode('domain');
    setSelectedProvider(null);
  };

  const changeProvider = (provider: Provider) => {
    setSelectedProvider(provider);
    if (session) {
      // Update session provider
      setSession(prev => prev ? {
        ...prev,
        provider: provider.id,
        provider_name: provider.name,
      } : null);
    }
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
              <br />간단한 질문은 개인 채팅, 중요한 사안은 위원회 모드를 추천드립니다.
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
              mode="chat"
              title="개인 법률 상담"
              description="선택한 AI 전문가와 1:1 대화"
              features={[
                '원하는 AI 모델 선택 가능',
                'CascadeFlow 비용 최적화',
                'RAG 기반 법률 지식 검색',
                '실시간 대화형 상담',
              ]}
              estimatedCost="~50원/질문"
              savings="80%"
              onClick={() => handleModeSelect('chat')}
            />
          </div>

          {/* Info Box */}
          <div className="bg-primary-main/5 border border-primary-main/20 rounded-xl p-4 flex items-start space-x-3">
            <Info className="w-5 h-5 text-primary-main flex-shrink-0 mt-0.5" />
            <div className="text-sm text-neutral-600">
              <p className="font-medium text-neutral-800 mb-1">어떤 모드를 선택해야 할까요?</p>
              <ul className="space-y-1">
                <li>• <strong>위원회 모드</strong>: 분쟁, 계약 검토, 중요한 법률 사안에 적합</li>
                <li>• <strong>개인 채팅</strong>: 일반 질문, 법률 용어 설명, 간단한 절차 안내에 적합</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render Domain Selection (after mode selection, before provider selection)
  if (mode === 'domain') {
    const currentDomainInfo = legalDomains.find(d => d.id === selectedDomain);
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
            <div className="inline-flex items-center justify-center p-3 bg-secondary-main rounded-xl shadow-lg mb-4">
              <Briefcase className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-3">
              어떤 법률 분야를 상담하시겠어요?
            </h1>
            <p className="text-neutral-500">
              분야에 맞는 전문 프롬프트가 적용되어 더 정확한 답변을 받으실 수 있습니다.
            </p>
          </div>

          {/* Domain Info Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex items-start space-x-3">
            <Shield className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-blue-800 mb-1">전문 분야별 최적화</p>
              <p className="text-blue-700">
                선택하신 분야에 맞는 법령 데이터베이스와 판례가 우선 검색되며, 해당 분야 전문 프롬프트가 적용됩니다.
              </p>
            </div>
          </div>

          {/* Domain Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            {legalDomains.map((domain) => (
              <DomainCard
                key={domain.id}
                domain={domain}
                selected={selectedDomain === domain.id}
                onClick={() => setSelectedDomain(domain.id)}
              />
            ))}
          </div>

          {/* Selected Domain Summary */}
          {currentDomainInfo && (
            <div className="bg-white border border-neutral-200 rounded-xl p-4 mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-primary-main/10 rounded-lg text-primary-main">
                  {currentDomainInfo.icon}
                </div>
                <div>
                  <p className="text-sm text-neutral-500">선택된 분야</p>
                  <p className="font-bold text-neutral-900">{currentDomainInfo.name}</p>
                </div>
              </div>
            </div>
          )}

          {/* Next Button */}
          <div className="flex justify-center">
            <Button
              variant="gold"
              size="lg"
              onClick={handleDomainSelect}
              className="px-12"
            >
              AI 전문가 선택하기
              <ChevronRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Render Provider Selection (before session starts)
  if (mode === 'chat' && !session) {
    const currentDomainInfo = legalDomains.find(d => d.id === selectedDomain);
    return (
      <div className="min-h-screen bg-neutral-50 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Back Button */}
          <button
            onClick={goBackToDomainSelect}
            className="flex items-center text-neutral-500 hover:text-neutral-700 mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            분야 선택으로 돌아가기
          </button>

          {/* Selected Domain Badge */}
          {currentDomainInfo && (
            <div className="flex justify-center mb-6">
              <div className="inline-flex items-center space-x-2 bg-primary-main/10 text-primary-main px-4 py-2 rounded-full">
                {currentDomainInfo.icon}
                <span className="font-medium">{currentDomainInfo.name}</span>
              </div>
            </div>
          )}

          {/* Header */}
          <div className="text-center mb-10">
            <h1 className="text-2xl md:text-3xl font-bold text-neutral-900 mb-3">
              어떤 AI 전문가와 상담하시겠어요?
            </h1>
            <p className="text-neutral-500">
              각 AI는 CascadeFlow로 비용을 최적화하며 <strong>{currentDomainInfo?.name}</strong> 전문 프롬프트를 사용합니다.
            </p>
          </div>

          {/* CascadeFlow Info */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6 flex items-start space-x-3">
            <Zap className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-yellow-800 mb-1">CascadeFlow 기술</p>
              <p className="text-yellow-700">
                간단한 질문은 빠르고 저렴한 모델(Drafter)로, 복잡한 질문은 고급 모델(Verifier)로 자동 라우팅되어 비용을 최대 80%까지 절감합니다.
              </p>
            </div>
          </div>

          {/* Provider Grid */}
          <div className="grid md:grid-cols-2 gap-4 mb-8">
            {providers.map((provider) => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                selected={selectedProvider?.id === provider.id}
                onClick={() => setSelectedProvider(provider)}
              />
            ))}
          </div>

          {/* Start Button */}
          <div className="flex justify-center">
            <Button
              variant="gold"
              size="lg"
              disabled={!selectedProvider || isLoading}
              onClick={createSession}
              className="px-12"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  세션 시작 중...
                </>
              ) : (
                <>
                  {selectedProvider?.name || 'AI 전문가'}와 상담 시작
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
            <ProviderIcon provider={selectedProvider?.id || 'claude'} />
            <div>
              <h2 className="font-bold text-neutral-900">{session?.provider_name}</h2>
              <p className="text-xs text-neutral-500 flex items-center">
                <Zap className="w-3 h-3 text-yellow-500 mr-1" />
                CascadeFlow 활성화
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {/* Provider Selector */}
            <select
              value={selectedProvider?.id}
              onChange={(e) => {
                const provider = providers.find(p => p.id === e.target.value);
                if (provider) changeProvider(provider);
              }}
              className="text-sm border border-neutral-200 rounded-lg px-2 py-1 bg-white"
            >
              {providers.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <CostDisplay session={session} />
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2">
          <div className="max-w-4xl mx-auto flex items-center text-red-700 text-sm">
            <X className="w-4 h-4 mr-2" />
            {error}
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <ProviderIcon provider={selectedProvider?.id || 'claude'} className="mx-auto mb-4 w-16 h-16 rounded-2xl" />
              <h3 className="text-lg font-bold text-neutral-800 mb-2">
                {session?.provider_name}와의 상담을 시작하세요
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
                  <div className="flex items-center justify-between mb-2 pb-2 border-b border-neutral-100">
                    <div className="flex items-center space-x-2">
                      <ProviderIcon provider={selectedProvider?.id || 'claude'} className="w-6 h-6 p-1" />
                      <span className="text-xs font-medium text-neutral-500">{session?.provider_name}</span>
                    </div>
                    <CascadeTierBadge tier={message.cascade_tier} />
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
