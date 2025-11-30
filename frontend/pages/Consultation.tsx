import React, { useState, useEffect, useRef } from 'react';
import { Send, Scale, FileText, CheckCircle, Loader2, Brain, Search, Users, ShieldCheck, AlertTriangle, BookOpen, Download, Camera, Paperclip, X, History, Clock, ChevronRight, ChevronDown, ChevronUp, Trash2, FileIcon, Mic, MicOff, GripHorizontal, MessageSquare, Sparkles, Eye } from 'lucide-react';
import { Button, Badge } from '../components/UIComponents';

type Step = 'input' | 'analyzing' | 'drafting' | 'reviewing' | 'synthesizing' | 'complete';

const steps = [
  { id: 'analyzing', label: '쟁점 분석', icon: Search },
  { id: 'drafting', label: '의견 작성', icon: Users },
  { id: 'reviewing', label: '상호 검증', icon: ShieldCheck },
  { id: 'synthesizing', label: '최종 합성', icon: Scale },
];

interface ModelOpinion {
  model: string;
  role: string;
  text: string;
  score: number;
  fullAnalysis?: string;
}

interface PeerReview {
  reviewer: string;
  targetModel: string;
  score: number;
  comment: string;
}

interface Citation {
  title: string;
  content: string;
  source: string;
}

interface ConsultationResult {
  confidence: number;
  summary: string;
  conclusion: string;
  details: string[];
  citations: Citation[];
  opinions: ModelOpinion[];
  peerReviews: PeerReview[];
}

// Single Turn of Conversation
interface ConsultationTurn {
  id: string;
  timestamp: string;
  query: string;
  attachments: string[];
  result: ConsultationResult;
}

// Entire Session Record
interface ConsultationRecord {
  id: string;
  startDate: string;
  category: string;
  turns: ConsultationTurn[];
}

interface InputAreaProps {
  compact?: boolean;
  query: string;
  setQuery: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
  category: string;
  setCategory: (value: string) => void;
  isListening: boolean;
  toggleListening: () => void;
  files: File[];
  handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  removeFile: (index: number) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}

const FilePreviewThumbnail: React.FC<{ file: File; onRemove: () => void }> = ({ file, onRemove }) => {
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreview(null);
    }
  }, [file]);

  return (
    <div className="relative group animate-fade-in">
       <div className={`relative flex items-center justify-center bg-white border border-neutral-200 shadow-sm overflow-hidden rounded-xl ${preview ? 'w-20 h-20' : 'w-24 h-20 p-2'}`}>
          {preview ? (
             <img src={preview} alt="preview" className="w-full h-full object-cover" />
          ) : (
             <div className="flex flex-col items-center text-center">
                <FileIcon className="w-6 h-6 text-neutral-400 mb-1" />
                <span className="text-[10px] text-neutral-500 max-w-[80px] truncate leading-tight">{file.name}</span>
                <span className="text-[9px] text-neutral-300 uppercase mt-0.5">{file.name.split('.').pop() || 'FILE'}</span>
             </div>
          )}
       </div>
       <button 
         type="button" 
         onClick={onRemove} 
         className="absolute -top-2 -right-2 bg-neutral-800 text-white rounded-full p-1 shadow-md hover:bg-red-500 transition-colors z-10"
       >
         <X className="w-3 h-3" />
       </button>
    </div>
  );
};

const InputArea: React.FC<InputAreaProps> = ({
  compact = false,
  query,
  setQuery,
  handleSubmit,
  category,
  setCategory,
  isListening,
  toggleListening,
  files,
  handleFileChange,
  removeFile,
  fileInputRef
}) => (
  <form onSubmit={handleSubmit} className={`relative transition-all duration-300 ${compact ? 'bg-white/90 backdrop-blur-md border-t border-neutral-200 p-4 fixed bottom-0 left-0 right-0 z-20 shadow-[0_-5px_20px_rgba(0,0,0,0.05)]' : 'bg-white/90 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-neutral-200'}`}>
      <div className={compact ? 'max-w-5xl mx-auto flex gap-4 items-end' : ''}>
         
         {!compact && (
            <div className="mb-4">
               <label className="block text-sm font-medium text-neutral-700 mb-2">상담 분야</label>
               <div className="flex flex-wrap gap-2">
                  {['일반/민사', '계약 검토', '지식재산권', '노무/인사', '형사/고소'].map((cat) => (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setCategory(cat)}
                      className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                        category === cat 
                          ? 'bg-primary-main text-white font-medium' 
                          : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
               </div>
            </div>
         )}

         <div className={`relative ${compact ? 'flex-grow' : 'mb-4'}`}>
            <textarea 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={compact ? "추가 질문을 입력하세요..." : "예: 프리랜서 개발자인데 클라이언트가 잔금 지급을 2달째 미루고 있습니다..."}
                className={`w-full border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-main focus:border-transparent outline-none resize-none leading-relaxed pr-12 bg-white ${
                  compact ? 'h-14 py-3 pl-4' : 'h-48 p-4 text-base'
                }`}
            ></textarea>
            <button
               type="button"
               onClick={toggleListening}
               className={`absolute right-3 rounded-full transition-all duration-300 p-2 ${
                  compact ? 'top-1/2 transform -translate-y-1/2' : 'bottom-4'
               } ${
                  isListening 
                     ? 'bg-red-500 text-white animate-pulse shadow-lg ring-2 ring-red-300' 
                     : 'text-neutral-400 hover:text-primary-main'
               }`}
            >
               {isListening ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
            </button>
         </div>
         
         {/* Compact Mode File & Send Buttons */}
         {compact && (
           <div className="flex gap-2 h-14 items-center">
              <input 
                 type="file" 
                 multiple 
                 ref={fileInputRef} 
                 className="hidden" 
                 onChange={handleFileChange} 
                 accept="image/*,.pdf"
              />
              <button 
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-3 text-neutral-500 hover:text-primary-main hover:bg-neutral-100 rounded-lg transition-colors"
              >
                <Paperclip className="w-5 h-5" />
              </button>
              <Button type="submit" variant="gold" className="h-full px-6 shadow-md hover:shadow-lg transition-shadow">
                 <Send className="w-5 h-5" />
              </Button>
           </div>
         )}

         {/* Full Mode File & Send */}
         {!compact && (
            <>
              <div className="mb-6">
                 <div className="flex items-center justify-between mb-2">
                   <label className="block text-sm font-medium text-neutral-700">증거 자료 첨부 (선택)</label>
                 </div>
                 <div className="flex gap-2 mb-3">
                   <input 
                     type="file" 
                     multiple 
                     ref={fileInputRef} 
                     className="hidden" 
                     onChange={handleFileChange} 
                     accept="image/*,.pdf"
                   />
                   <button 
                     type="button"
                     onClick={() => fileInputRef.current?.click()}
                     className="flex items-center px-4 py-2 border border-neutral-300 rounded-lg text-sm text-neutral-600 hover:bg-neutral-50 hover:border-neutral-400 transition-colors bg-white"
                   >
                     <Paperclip className="w-4 h-4 mr-2" /> 파일 찾기
                   </button>
                   <button 
                     type="button"
                     onClick={() => fileInputRef.current?.click()}
                     className="flex items-center px-4 py-2 border border-neutral-300 rounded-lg text-sm text-neutral-600 hover:bg-neutral-50 hover:border-neutral-400 transition-colors bg-white"
                   >
                     <Camera className="w-4 h-4 mr-2" /> 사진 촬영
                   </button>
                 </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-neutral-100">
                 <p className="text-xs text-neutral-400 flex items-center">
                    <AlertTriangle className="w-3 h-3 mr-1" />
                    개인정보(실명, 전화번호)는 가급적 제외해주세요.
                 </p>
                 <Button type="submit" variant="gold" size="lg" className="px-8 shadow-md hover:shadow-xl transition-all" disabled={!query.trim()}>
                    자문 요청하기 <Send className="ml-2 w-4 h-4" />
                 </Button>
              </div>
            </>
         )}
      </div>
      
      {/* File Preview (Common) */}
      {files.length > 0 && (
        <div className={`flex flex-wrap gap-3 ${compact ? 'max-w-5xl mx-auto mt-3 px-16' : 'mt-4'}`}>
          {files.map((file, idx) => (
            <FilePreviewThumbnail key={idx} file={file} onRemove={() => removeFile(idx)} />
          ))}
        </div>
      )}
  </form>
);

const Consultation: React.FC = () => {
  // Application State
  const [step, setStep] = useState<Step>('input');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('일반/민사');
  const [files, setFiles] = useState<File[]>([]);
  
  // History & Session State
  const [history, setHistory] = useState<ConsultationRecord[]>([]);
  const [currentRecord, setCurrentRecord] = useState<ConsultationRecord | null>(null);
  
  // UI State
  const [expandedItems, setExpandedItems] = useState<string[]>([]); // Store IDs like "turnId-type-idx"
  const [progress, setProgress] = useState(0);
  const [isListening, setIsListening] = useState(false);
  
  // Draggable Header State
  const [headerTop, setHeaderTop] = useState(64);
  const isDraggingHeader = useRef(false);
  const dragStartY = useRef(0);
  const initialHeaderTop = useRef(64);

  // Simulation State
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  // Load history on mount
  useEffect(() => {
    const saved = localStorage.getItem('consultation_history_v2');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Save history whenever it changes
  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem('consultation_history_v2', JSON.stringify(history));
    }
  }, [history]);

  // Scroll to bottom when new turn is added or step changes
  useEffect(() => {
    if ((step === 'complete' || step !== 'input') && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentRecord?.turns.length, step]);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Global Mouse Events for Dragging Header
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingHeader.current) {
        const deltaY = e.clientY - dragStartY.current;
        const newTop = Math.max(64, initialHeaderTop.current + deltaY);
        setHeaderTop(newTop);
      }
    };

    const handleMouseUp = () => {
      if (isDraggingHeader.current) {
        isDraggingHeader.current = false;
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString().split(' ')[0]}] ${message}`]);
  };

  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    isDraggingHeader.current = true;
    dragStartY.current = e.clientY;
    initialHeaderTop.current = headerTop;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'move';
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(prev => [...prev, ...Array.from(e.target.files || [])]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const toggleItem = (turnId: string, type: 'opinion' | 'citation' | 'review', index: number) => {
    const id = `${turnId}-${type}-${index}`;
    setExpandedItems(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const toggleListening = () => {
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("이 브라우저는 음성 인식을 지원하지 않습니다.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'ko-KR';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (finalTranscript) {
        setQuery(prev => prev + (prev && !prev.endsWith(' ') ? ' ' : '') + finalTranscript);
      }
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error", event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      if (isListening) {
         setIsListening(false);
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  };

  const generateMockResult = (currentQuery: string, currentCategory: string): ConsultationResult => {
    const isFollowUp = currentRecord && currentRecord.turns.length > 0;
    
    // Dynamic content based on query keywords (Mock simulation)
    let summaryText = "귀하의 질의 내용을 종합적으로 검토한 결과, 해당 사안은 법적 보호를 받을 가능성이 높습니다.";
    let conclusionText = `검토 결과, 본 사안은 ${currentCategory} 관련 법령에 의거하여 청구권이 인정될 소지가 다분합니다.`;
    
    if (currentQuery.includes("비용") || currentQuery.includes("얼마")) {
      summaryText = "소송 비용 및 경제적 실익에 대한 분석을 진행하였습니다.";
      conclusionText = "청구 금액 대비 변호사 보수 및 인지대를 고려할 때, 지급명령 신청이 가장 경제적인 수단으로 판단됩니다.";
    } else if (currentQuery.includes("기간") || currentQuery.includes("언제")) {
      summaryText = "소멸시효 및 법적 대응 기간에 대한 검토 결과입니다.";
      conclusionText = "해당 채권의 소멸시효는 3년이므로, 내용증명 발송을 통해 시효를 중단시키는 것이 시급합니다.";
    }

    if (isFollowUp) {
       summaryText = "추가 질의하신 사항에 대해 위원회가 재소집되어 검토하였습니다.";
       conclusionText = "앞선 자문 내용과 결부하여 볼 때, 추가적인 증거 확보가 핵심 쟁점이 될 것입니다.";
    }

    return {
      confidence: Math.floor(Math.random() * 5) + 93, // 93-97%
      summary: summaryText,
      conclusion: conclusionText,
      details: [
        "계약의 성립 요건을 충족하며, 상대방의 귀책 사유가 명백해 보입니다.",
        "관련 판례(대법원 2018다2XXXX)에 따르면 유사한 사례에서 원고 승소 판결이 내려진 바 있습니다.",
        "내용증명 발송을 통해 소멸시효를 중단시키는 조치가 선행되어야 합니다."
      ],
      citations: [
        {
           title: "민법 제390조 (채무불이행과 손해배상)",
           source: "국가법령정보센터",
           content: "채무자가 채무의 내용에 좇은 이행을 하지 아니한 때에는 채권자는 손해배상을 청구할 수 있다. 그러나 채무자의 고의나 과실없이 이행할 수 없게 된 때에는 그러하지 아니하다."
        },
        {
           title: "민법 제750조 (불법행위의 내용)",
           source: "국가법령정보센터",
           content: "고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다."
        },
        {
           title: "상가건물 임대차보호법 제10조",
           source: "국가법령정보센터",
           content: "임대인은 임차인이 임대차기간이 만료되기 6개월 전부터 1개월 전까지 사이에 계약갱신을 요구할 경우 정당한 사유 없이 거절하지 못한다."
        }
      ],
      peerReviews: [
         {
            reviewer: "Claude 4.5",
            targetModel: "GPT-5.1",
            score: 9.8,
            comment: "법리적 해석이 매우 정교하며, 특히 최신 판례를 인용한 논리 구성이 탁월합니다. 다만, 소송 비용에 대한 구체적인 산출 근거가 보강되면 더 좋을 것입니다."
         },
         {
            reviewer: "GPT-5.1",
            targetModel: "Gemini Pro",
            score: 9.4,
            comment: "판결 예측 통계 데이터가 매우 유용합니다. 그러나 해당 통계의 모집단(기간, 법원)을 조금 더 명확히 명시할 필요가 있습니다."
         },
         {
            reviewer: "Grok 4",
            targetModel: "Claude 4.5",
            score: 9.6,
            comment: "입증 책임에 대한 분석이 예리합니다. 카카오톡 대화 내역의 증거 능력을 인정한 부분은 실무적으로 매우 적절한 판단입니다."
         }
      ],
      opinions: [
        { 
          model: "GPT-5.1", 
          role: "Legal Advisor", 
          text: "계약서 조항의 문리적 해석에 집중했을 때, 의뢰인에게 유리한 해석이 가능합니다.", 
          score: 9.2,
          fullAnalysis: "계약서 제5조 제2항의 '검수 완료' 시점에 대한 해석이 쟁점입니다. 통상적인 업계 관행과 대법원 판례(20XX다XXXX)에 비추어 볼 때, 결과물 수령 후 7일 이내에 이의제기가 없으면 검수가 완료된 것으로 보는 묵시적 의사표시가 인정될 가능성이 높습니다."
        },
        { 
          model: "Claude 4.5", 
          role: "Legal Advisor", 
          text: "판례의 태도를 고려할 때 입증 책임의 분배가 핵심 쟁점이 될 것입니다.", 
          score: 9.5,
          fullAnalysis: "유사한 하급심 판례들을 분석한 결과, 단순히 작업물을 전송한 사실만으로는 용역 수행 완료를 입증하기 부족할 수 있습니다. 그러나 의뢰인이 제시한 카카오톡 대화 내용 중 '수고하셨습니다'라는 메시지는 유력한 증거입니다."
        },
        { 
          model: "Gemini Pro", 
          role: "Legal Advisor", 
          text: "최신 하급심 판결 경향을 분석한 결과 승소 가능성이 70% 이상으로 예측됩니다.", 
          score: 8.9,
          fullAnalysis: "최근 3년간의 유사 분쟁 판결 150건을 분석하였습니다. 계약서가 존재하고 조항이 비교적 명확하므로 승소 확률이 높게 산정됩니다. 소송보다는 지급명령 신청이 효율적일 것입니다."
        },
        { 
          model: "Grok 4", 
          role: "Legal Advisor", 
          text: "상대방의 행위는 신의성실의 원칙에 반하는 것으로 볼 여지가 큽니다.", 
          score: 8.5,
          fullAnalysis: "상대방이 잔금 지급을 미루면서 구체적인 수정 요청 없이 시간만 끌고 있는 행태는 민법 제2조 신의성실의 원칙에 위배되는 권리 남용으로 볼 소지가 큽니다."
        }
      ]
    };
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Capture current files and query for this turn
    const turnFiles = [...files];
    const turnQuery = query;

    // Clear input immediately for better UX
    setQuery('');
    setFiles([]);
    setStep('analyzing');
    setLogs([]);
    setProgress(0);
    
    // Simulation Timeline
    setTimeout(() => {
       addLog(`Session ${currentRecord ? 'Update' : 'Init'}...`);
       addLog("Receiving user query and attachments...");
       if (turnFiles.length > 0) {
         addLog(`Processing ${turnFiles.length} attached documents via OCR...`);
       }
       setProgress(10);
    }, 500);

    setTimeout(() => {
       addLog("Processing Natural Language Understanding (NLU)...");
       addLog(`Context: ${currentRecord ? 'Follow-up Question' : 'New Case'}`);
       setProgress(25);
    }, 1500);

    setTimeout(() => {
       addLog("Querying RAG Database...");
       addLog("Retrieved relevant legal precedents.");
       setStep('drafting');
       setProgress(40);
    }, 3500);

    setTimeout(() => {
       addLog("Council Members independently reviewing the case...");
       setProgress(50);
    }, 4500);

    setTimeout(() => {
       addLog("Drafting legal opinions...");
       setProgress(60);
    }, 6000);
    
    setTimeout(() => {
       setStep('reviewing');
       setProgress(75);
    }, 7500);

    setTimeout(() => {
       addLog("Blind Peer Review in progress...");
       setProgress(85);
    }, 9000);

    setTimeout(() => {
       setStep('synthesizing');
       setProgress(90);
    }, 11000);

    setTimeout(() => {
       addLog("Chairman AI: Synthesizing final response...");
       setProgress(95);
    }, 13000);

    setTimeout(() => {
       const mockResult = generateMockResult(turnQuery, category);
       
       const newTurn: ConsultationTurn = {
         id: Date.now().toString(),
         timestamp: new Date().toISOString(),
         query: turnQuery,
         attachments: turnFiles.map(f => f.name),
         result: mockResult
       };

       if (currentRecord) {
         // Append to existing record
         const updatedRecord = {
           ...currentRecord,
           turns: [...currentRecord.turns, newTurn]
         };
         setCurrentRecord(updatedRecord);
         
         // Update history
         const updatedHistory = history.map(h => h.id === updatedRecord.id ? updatedRecord : h);
         setHistory(updatedHistory);
       } else {
         // Create new record
         const newRecord: ConsultationRecord = {
           id: Date.now().toString(),
           startDate: new Date().toISOString(),
           category,
           turns: [newTurn]
         };
         setCurrentRecord(newRecord);
         setHistory([newRecord, ...history]);
       }
       
       addLog("Process complete.");
       setProgress(100);
       setStep('complete');
    }, 15000);
  };

  const handleDeleteHistory = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newHistory = history.filter(h => h.id !== id);
    setHistory(newHistory);
    // If deleted current record, reset
    if (currentRecord && currentRecord.id === id) {
      resetForm();
    }
  };

  const loadHistoryItem = (record: ConsultationRecord) => {
    setCurrentRecord(record);
    setExpandedItems([]);
    setStep('complete');
    setCategory(record.category);
  };

  const resetForm = () => {
    setQuery('');
    setFiles([]);
    setCurrentRecord(null);
    setStep('input');
    setProgress(0);
  };

  // --- Render Components ---

  const renderVisualizer = () => (
    <div className="bg-neutral-900/95 backdrop-blur-sm rounded-2xl p-4 md:p-8 mb-8 text-white relative min-h-[520px] flex flex-col items-center justify-center animate-fade-in shadow-2xl border border-neutral-800 overflow-hidden">
       {/* Background Grid */}
       <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(#ffffff 1px, transparent 1px)', backgroundSize: '24px 24px' }}></div>
       
       {/* Visualizer Content Container */}
       <div className="relative w-full h-full flex flex-col items-center justify-start py-8">
          
          {/* Progress Stepper */}
          <div className="flex justify-between w-full max-w-lg mb-12 px-4 relative z-20">
             {steps.map((s, idx) => {
               const isActive = s.id === step;
               const isPast = steps.findIndex(st => st.id === step) > idx;
               const Icon = s.icon;
               
               return (
                 <div key={s.id} className="flex flex-col items-center relative group">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 z-10 ${isActive ? 'bg-accent-gold border-accent-gold text-primary-dark scale-110 shadow-[0_0_15px_rgba(234,179,8,0.5)]' : isPast ? 'bg-secondary-main border-secondary-main text-white' : 'bg-neutral-800 border-neutral-600 text-neutral-500'}`}>
                       <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-xs mt-2 font-medium transition-colors ${isActive ? 'text-accent-gold' : isPast ? 'text-secondary-main' : 'text-neutral-500'}`}>{s.label}</span>
                    
                    {/* Connector Line */}
                    {idx < steps.length - 1 && (
                       <div className={`absolute top-5 left-1/2 w-full h-0.5 -z-0 transition-colors duration-500 ${isPast ? 'bg-secondary-main' : 'bg-neutral-700'}`} style={{ width: 'calc(100% + 2rem)' }}></div>
                    )}
                 </div>
               );
             })}
          </div>

          {/* Council Table Container */}
          <div className="relative w-full max-w-md aspect-square max-h-[350px]">
             {/* Central Hub (Chairman) */}
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 flex flex-col items-center">
                <div className={`relative w-24 h-24 md:w-32 md:h-32 rounded-full border-4 border-accent-gold/50 flex items-center justify-center bg-neutral-800 transition-all duration-700 ${step === 'synthesizing' ? 'shadow-[0_0_50px_rgba(234,179,8,0.6)] scale-110 border-accent-gold' : 'shadow-xl'}`}>
                   <Scale className={`w-10 h-10 md:w-12 md:h-12 text-accent-gold ${step === 'synthesizing' ? 'animate-pulse' : ''}`} />
                   {/* Progress Ring */}
                   <svg className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none">
                      <circle 
                         cx="50%" cy="50%" r="46%" 
                         stroke="hsl(45, 80%, 50%)" 
                         strokeWidth="3" 
                         fill="none" 
                         strokeDasharray="289" // 2 * pi * 46
                         strokeDashoffset={289 - (289 * progress) / 100} 
                         className="transition-all duration-700 ease-out"
                         strokeLinecap="round"
                      />
                   </svg>
                </div>
                {step === 'synthesizing' && (
                   <div className="mt-4 text-accent-gold font-bold text-sm tracking-widest animate-pulse">SYNTHESIZING</div>
                )}
             </div>

             {/* Connecting Lines (Animated) */}
             <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-50">
               {/* Center Glow */}
               <circle cx="50%" cy="50%" r="20%" fill="url(#centerGlow)" className="opacity-20" />
               <defs>
                  <radialGradient id="centerGlow">
                     <stop offset="0%" stopColor="#EAB308" />
                     <stop offset="100%" stopColor="transparent" />
                  </radialGradient>
               </defs>
               
               {/* Connections to center */}
               <line x1="15%" y1="15%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <line x1="85%" y1="15%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <line x1="15%" y1="85%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <line x1="85%" y1="85%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

               {/* Active Flow Lines */}
               {step !== 'analyzing' && (
                  <>
                     <circle cx="50%" cy="50%" r="0" fill="white" className="animate-ping" style={{ animationDuration: '3s' }} />
                  </>
               )}
               
               {/* Cross Validation Lines (Reviewing Step) */}
               {step === 'reviewing' && (
                  <>
                     <path d="M 20% 20% Q 50% 10% 80% 20%" stroke="#2DD4BF" strokeWidth="2" fill="none" className="animate-pulse" strokeDasharray="5 5"/>
                     <path d="M 80% 20% Q 90% 50% 80% 80%" stroke="#2DD4BF" strokeWidth="2" fill="none" className="animate-pulse" strokeDasharray="5 5"/>
                     <path d="M 80% 80% Q 50% 90% 20% 80%" stroke="#2DD4BF" strokeWidth="2" fill="none" className="animate-pulse" strokeDasharray="5 5"/>
                     <path d="M 20% 80% Q 10% 50% 20% 20%" stroke="#2DD4BF" strokeWidth="2" fill="none" className="animate-pulse" strokeDasharray="5 5"/>
                  </>
               )}
             </svg>

             {/* AI Model Cards (Corner Positions) */}
             {[
                { id: 'gpt', name: 'GPT-5.1', color: 'bg-blue-500', pos: 'top-0 left-0', translate: '-translate-x-1/4 -translate-y-1/4' },
                { id: 'claude', name: 'Claude 4.5', color: 'bg-purple-500', pos: 'top-0 right-0', translate: 'translate-x-1/4 -translate-y-1/4' },
                { id: 'gemini', name: 'Gemini Pro', color: 'bg-teal-500', pos: 'bottom-0 left-0', translate: '-translate-x-1/4 translate-y-1/4' },
                { id: 'grok', name: 'Grok 4', color: 'bg-indigo-500', pos: 'bottom-0 right-0', translate: 'translate-x-1/4 translate-y-1/4' },
             ].map((model, idx) => (
                <div 
                   key={model.id}
                   className={`absolute ${model.pos} transform ${model.translate} z-20 flex flex-col items-center transition-all duration-500`}
                >
                   <div className={`w-16 h-16 md:w-20 md:h-20 rounded-2xl ${model.color}/20 border border-${model.color.split('-')[1]}-400/50 backdrop-blur-md flex items-center justify-center shadow-lg transition-all duration-300 ${step === 'drafting' ? 'scale-110 border-white ring-2 ring-white/50' : ''}`}>
                      <Brain className={`w-8 h-8 md:w-10 md:h-10 text-white ${step === 'drafting' ? 'animate-bounce' : ''}`} />
                   </div>
                   <div className="mt-2 bg-black/50 px-3 py-1 rounded-full border border-white/10 backdrop-blur-sm">
                      <span className="text-xs md:text-sm font-bold text-white">{model.name}</span>
                   </div>
                   {step === 'drafting' && (
                      <div className="absolute -top-6 bg-white text-black text-xs px-2 py-1 rounded shadow animate-pulse whitespace-nowrap">
                         Drafting Opinion...
                      </div>
                   )}
                   {step === 'reviewing' && (
                      <div className="absolute -top-6 bg-secondary-main text-white text-xs px-2 py-1 rounded shadow whitespace-nowrap">
                         Reviewing Peers
                      </div>
                   )}
                </div>
             ))}
          </div>

          <div className="text-center mt-8 relative z-20">
             <div className="flex items-center justify-center space-x-3 mb-2">
                {step === 'analyzing' && <Loader2 className="animate-spin text-accent-gold" />}
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-neutral-400">
                   {steps.find(s => s.id === step)?.label || '처리 중...'}
                </h2>
             </div>
             <p className="text-neutral-400 text-sm max-w-md mx-auto h-6">
                {step === 'analyzing' && "RAG 시스템이 관련 법령 데이터를 로드하고 있습니다."}
                {step === 'drafting' && "4개의 AI 모델이 독립적으로 법률 의견을 작성합니다."}
                {step === 'reviewing' && "작성된 의견을 상호 교차 검증하여 오류를 수정합니다."}
                {step === 'synthesizing' && "최종 의견을 하나의 완결된 문서로 통합하고 있습니다."}
             </p>
          </div>
       </div>

       {/* Progress Bar & Status Footer */}
       <div className="absolute bottom-0 left-0 right-0 bg-neutral-900/80 border-t border-neutral-800 p-4">
          <div className="flex items-center justify-between text-xs font-mono text-neutral-500 mb-2 px-2">
             <span className="flex items-center"><Sparkles className="w-3 h-3 mr-1 text-secondary-main"/> AI ORCHESTRATOR ACTIVE</span>
             <span>PHASE: {step.toUpperCase()}</span>
          </div>
          <div className="w-full h-2 bg-neutral-800 rounded-full overflow-hidden shadow-inner relative">
             <div 
               className="h-full bg-gradient-to-r from-secondary-main to-accent-gold transition-all duration-300 ease-out shadow-[0_0_10px_rgba(234,179,8,0.5)]" 
               style={{ width: `${progress}%` }}
             ></div>
             {/* Glow effect moving across bar */}
             <div className="absolute top-0 bottom-0 w-20 bg-white/20 skew-x-12 animate-[shimmer_2s_infinite] pointer-events-none"></div>
          </div>
          <div className="flex justify-end mt-2">
             <span className="text-accent-gold font-bold font-mono text-lg">{progress}%</span>
          </div>
       </div>
    </div>
  );

  const renderLogs = () => (
    <div className="bg-black text-green-400 p-4 rounded-xl font-mono text-sm h-48 overflow-y-auto border border-neutral-800 shadow-inner scrollbar-hide">
      {logs.map((log, i) => (
        <div key={i} className="mb-1 opacity-80 hover:opacity-100 transition-opacity">
           <span className="text-green-600 mr-2">&gt;</span>
           {log}
        </div>
      ))}
      <div ref={logsEndRef} />
    </div>
  );

  const renderOpinions = (turnId: string, opinions: ModelOpinion[]) => (
    <div className="grid grid-cols-1 gap-4 mb-6">
      {opinions.map((op, idx) => {
        const isExpanded = expandedItems.includes(`${turnId}-opinion-${idx}`);
        
        return (
          <div key={idx} className="bg-white rounded-xl border border-neutral-200 shadow-sm hover:shadow-md transition-all overflow-hidden group">
            <div 
              className="p-4 flex items-center justify-between cursor-pointer bg-neutral-50/50 hover:bg-neutral-50 transition-colors"
              onClick={() => toggleItem(turnId, 'opinion', idx)}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-white text-xs font-bold shadow-sm ${
                  op.model.includes('GPT') ? 'bg-blue-500' :
                  op.model.includes('Claude') ? 'bg-purple-500' :
                  op.model.includes('Gemini') ? 'bg-teal-500' : 'bg-indigo-600'
                }`}>
                  {op.model.substring(0, 1)}
                </div>
                <div>
                  <div className="font-bold text-neutral-800 flex items-center">
                     {op.model}
                     <Badge variant="neutral" className="ml-2 text-[10px] py-0">{op.role}</Badge>
                  </div>
                  <div className="text-xs text-neutral-500 mt-0.5 line-clamp-1">{op.text}</div>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                 <div className="text-right hidden sm:block">
                    <div className="text-xs text-neutral-400">Score</div>
                    <div className="font-bold text-secondary-main">{op.score}/10</div>
                 </div>
                 {isExpanded ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
              </div>
            </div>
            
            {/* Expanded Content */}
            {isExpanded && (
               <div className="p-5 border-t border-neutral-100 bg-white animate-fade-in">
                  <h4 className="font-bold text-sm text-neutral-800 mb-2 flex items-center">
                     <FileText className="w-4 h-4 mr-2 text-neutral-500" /> 상세 법률 검토 의견
                  </h4>
                  <p className="text-neutral-600 text-sm leading-relaxed whitespace-pre-line mb-4">
                     {op.fullAnalysis || op.text}
                  </p>
                  <div className="flex gap-2">
                     <button className="text-xs flex items-center text-neutral-400 hover:text-primary-main transition-colors">
                        <CheckCircle className="w-3 h-3 mr-1" /> 논리적 타당성 검증됨
                     </button>
                     <button className="text-xs flex items-center text-neutral-400 hover:text-primary-main transition-colors">
                        <BookOpen className="w-3 h-3 mr-1" /> 판례 인용 적절함
                     </button>
                  </div>
               </div>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderChatTurn = (turn: ConsultationTurn) => (
    <div key={turn.id} className="mb-16 animate-fade-in-up">
       {/* User Message */}
       <div className="flex justify-end mb-8">
          <div className="max-w-2xl">
             <div className="bg-primary-main text-white p-5 rounded-2xl rounded-tr-sm shadow-md relative">
                <p className="whitespace-pre-wrap leading-relaxed">{turn.query}</p>
                {/* Attachments */}
                {turn.attachments.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-white/20">
                     <div className="text-xs opacity-70 mb-1 flex items-center"><Paperclip className="w-3 h-3 mr-1"/> 첨부 파일</div>
                     <div className="flex flex-wrap gap-2">
                        {turn.attachments.map((file, i) => (
                           <span key={i} className="inline-flex items-center bg-white/20 px-2 py-1 rounded text-xs">
                              <FileIcon className="w-3 h-3 mr-1"/> {file}
                           </span>
                        ))}
                     </div>
                  </div>
                )}
             </div>
             <div className="text-right text-xs text-neutral-400 mt-2 mr-1">
                {new Date(turn.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
             </div>
          </div>
       </div>

       {/* AI Response (Legal Memo) */}
       <div className="flex justify-start w-full">
          <div className="w-full max-w-4xl bg-white rounded-xl shadow-xl border border-neutral-200 overflow-hidden">
             {/* Header */}
             <div className="bg-neutral-50 border-b border-neutral-200 p-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="flex items-center space-x-3">
                   <div className="p-2 bg-primary-dark rounded-lg">
                      <Scale className="h-6 w-6 text-accent-gold" />
                   </div>
                   <div>
                      <h2 className="text-lg font-bold text-primary-dark font-heading">AI Legal Advisory Council</h2>
                      <p className="text-xs text-neutral-500">Official Legal Memorandum • Ref: #{turn.id.slice(-6)}</p>
                   </div>
                </div>
                <div className="flex items-center space-x-2">
                   <Badge variant="success">검토 완료</Badge>
                   <Button variant="outline" size="sm" className="h-8 text-xs border-neutral-300 text-neutral-600 hover:bg-neutral-100">
                      <Download className="w-3 h-3 mr-1" /> PDF 저장
                   </Button>
                </div>
             </div>

             {/* Body */}
             <div className="p-8">
                {/* Summary Box */}
                <div className="bg-primary-main/5 border border-primary-main/10 rounded-xl p-6 mb-8">
                   <h3 className="text-primary-dark font-bold mb-2 flex items-center">
                      <Sparkles className="w-4 h-4 mr-2 text-primary-main"/> 
                      Executive Summary
                   </h3>
                   <p className="text-neutral-700 leading-relaxed font-medium">
                      {turn.result.summary}
                   </p>
                </div>

                <div className="space-y-8">
                   <section>
                      <h3 className="text-lg font-bold text-neutral-900 border-l-4 border-secondary-main pl-3 mb-4">
                         1. 법률적 검토 결과 (Conclusion)
                      </h3>
                      <p className="text-neutral-700 leading-relaxed pl-4">
                         {turn.result.conclusion}
                      </p>
                   </section>

                   <section>
                      <h3 className="text-lg font-bold text-neutral-900 border-l-4 border-secondary-main pl-3 mb-4">
                         2. 상세 분석 (Detailed Analysis)
                      </h3>
                      <ul className="space-y-3 pl-4">
                         {turn.result.details.map((detail, idx) => (
                            <li key={idx} className="flex items-start text-neutral-700">
                               <CheckCircle className="w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
                               <span className="leading-relaxed">{detail}</span>
                            </li>
                         ))}
                      </ul>
                   </section>

                   <section>
                      <h3 className="text-lg font-bold text-neutral-900 border-l-4 border-secondary-main pl-3 mb-4">
                         3. 관련 근거 (Legal Basis)
                      </h3>
                      <div className="pl-4 grid gap-3">
                         {turn.result.citations.map((cite, idx) => {
                           const isExpanded = expandedItems.includes(`${turn.id}-citation-${idx}`);
                           return (
                             <div key={idx} className="bg-neutral-50 rounded-lg border border-neutral-100 overflow-hidden">
                                <div 
                                  className="p-3 flex items-center justify-between cursor-pointer hover:bg-neutral-100 transition-colors"
                                  onClick={() => toggleItem(turn.id, 'citation', idx)}
                                >
                                   <div className="flex items-center text-sm font-medium text-neutral-700">
                                      <BookOpen className="w-4 h-4 text-secondary-main mr-2" />
                                      {cite.title}
                                   </div>
                                   {isExpanded ? <ChevronUp className="w-4 h-4 text-neutral-400"/> : <ChevronDown className="w-4 h-4 text-neutral-400"/>}
                                </div>
                                {isExpanded && (
                                   <div className="px-4 pb-4 pt-0 text-sm text-neutral-600 animate-fade-in">
                                      <div className="mb-2 text-xs text-neutral-400 uppercase tracking-wide border-b border-neutral-200 pb-1 mt-2">
                                         Source: {cite.source}
                                      </div>
                                      <p className="leading-relaxed whitespace-pre-line bg-white p-3 rounded border border-neutral-100 text-neutral-700">
                                         "{cite.content}"
                                      </p>
                                   </div>
                                )}
                             </div>
                           );
                         })}
                      </div>
                   </section>

                   <section>
                      <h3 className="text-lg font-bold text-neutral-900 border-l-4 border-secondary-main pl-3 mb-4">
                         4. 자문 위원 개별 의견 (Council Opinions)
                      </h3>
                      <div className="pl-1">
                         {renderOpinions(turn.id, turn.result.opinions)}
                      </div>
                   </section>

                   {/* New Section: Peer Review */}
                   <section>
                      <h3 className="text-lg font-bold text-neutral-900 border-l-4 border-secondary-main pl-3 mb-4 flex items-center">
                         5. 블라인드 피어 리뷰 (Blind Peer Review)
                         <Badge variant="warning" className="ml-2">Cross Validation</Badge>
                      </h3>
                      <div className="pl-1 grid grid-cols-1 md:grid-cols-2 gap-4">
                         {turn.result.peerReviews?.map((review, idx) => {
                           const isExpanded = expandedItems.includes(`${turn.id}-review-${idx}`);
                           return (
                              <div key={idx} className="bg-white rounded-lg border border-neutral-200 p-4 shadow-sm">
                                 <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center space-x-2">
                                       <div className="w-6 h-6 rounded bg-neutral-800 flex items-center justify-center text-white text-[10px] font-bold">
                                          {review.reviewer.substring(0, 1)}
                                       </div>
                                       <span className="text-xs text-neutral-400">reviewed</span>
                                       <div className="w-6 h-6 rounded bg-neutral-200 flex items-center justify-center text-neutral-600 text-[10px] font-bold">
                                          {review.targetModel.substring(0, 1)}
                                       </div>
                                    </div>
                                    <div className="text-sm font-bold text-secondary-main">{review.score}/10</div>
                                 </div>
                                 <div className="relative">
                                    <p className={`text-sm text-neutral-600 ${isExpanded ? '' : 'line-clamp-2'}`}>
                                       "{review.comment}"
                                    </p>
                                    <button 
                                       onClick={() => toggleItem(turn.id, 'review', idx)}
                                       className="text-xs text-neutral-400 hover:text-primary-main mt-1 flex items-center gap-1"
                                    >
                                       {isExpanded ? '접기' : '더 보기'}
                                    </button>
                                 </div>
                              </div>
                           );
                         })}
                      </div>
                   </section>

                </div>

                {/* Footer Seal */}
                <div className="mt-12 pt-8 border-t border-neutral-100 flex justify-end items-center">
                   <div className="text-right mr-4">
                      <div className="font-heading font-bold text-primary-dark text-lg">AI Legal Council Chairman</div>
                      <div className="text-xs text-neutral-400">Synthesized on {new Date(turn.timestamp).toLocaleDateString()}</div>
                   </div>
                   <div className="w-20 h-20 border-2 border-primary-main/20 rounded-full flex items-center justify-center relative rotate-12 opacity-80 mix-blend-multiply">
                      <div className="absolute inset-0 border border-primary-main/20 rounded-full m-1"></div>
                      <div className="text-center">
                         <div className="text-[8px] font-bold text-primary-main tracking-widest uppercase mb-0.5">Approved By</div>
                         <Scale className="w-8 h-8 text-primary-main mx-auto" />
                         <div className="text-[8px] font-bold text-primary-main tracking-widest uppercase mt-0.5">AI Council</div>
                      </div>
                   </div>
                </div>
             </div>
          </div>
       </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-neutral-50 relative overflow-hidden flex flex-col">
       {/* Adaptive Background */}
       <div className="fixed inset-0 z-0 transition-opacity duration-1000 ease-in-out">
          {step === 'input' && currentRecord === null ? (
             // Initial State Background (Legal Classic)
             <>
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1589829085413-56de8ae18c73?auto=format&fit=crop&q=80&w=2000')] bg-cover bg-center opacity-5 grayscale"></div>
                <div className="absolute inset-0 bg-gradient-to-b from-neutral-50 via-transparent to-neutral-50"></div>
             </>
          ) : (
             // Active State Background (Modern Tech)
             <>
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1506399558188-acca3f85ed94?auto=format&fit=crop&q=80&w=2000')] bg-cover bg-center opacity-5"></div>
                <div className="absolute inset-0 bg-white/90 backdrop-blur-[2px]"></div>
             </>
          )}
          {/* Subtle Pattern Overlay */}
          <div className="absolute inset-0 bg-[radial-gradient(#000000_1px,transparent_1px)] [background-size:24px_24px] opacity-[0.03]"></div>
       </div>

       {/* Draggable Header */}
       {currentRecord && (step !== 'input' || currentRecord.turns.length > 0) && (
          <>
             {/* Spacer to prevent content jump */}
             <div className="h-20" />
             <div 
               style={{ top: `${headerTop}px` }}
               className="fixed left-0 right-0 z-40 px-4 md:px-8 pointer-events-none"
             >
                <div 
                  onMouseDown={handleHeaderMouseDown}
                  className="max-w-4xl mx-auto bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-neutral-200/50 p-2 flex items-center justify-between pointer-events-auto cursor-grab active:cursor-grabbing group transition-shadow hover:shadow-xl"
                >
                   <div className="flex items-center pl-4">
                      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse mr-3"></div>
                      <span className="font-bold text-neutral-800 text-sm">AI Legal Council Session</span>
                      <Badge variant="neutral" className="ml-3 text-xs hidden sm:inline-flex">Beta Simulation</Badge>
                   </div>
                   
                   {/* Drag Handle */}
                   <div className="flex-1 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <GripHorizontal className="w-5 h-5 text-neutral-300" />
                   </div>

                   <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm" onClick={() => resetForm()} className="h-8 text-xs px-3">
                         <ChevronRight className="w-3 h-3 mr-1" /> New
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => {}} className="h-8 text-xs px-3 bg-neutral-100 hover:bg-neutral-200 border-none">
                         <History className="w-3 h-3 text-neutral-500" />
                      </Button>
                   </div>
                </div>
             </div>
          </>
       )}

       <div className="relative z-10 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 flex-grow pb-32">
          
          {/* Initial Input View */}
          {step === 'input' && !currentRecord && (
             <div className="max-w-3xl mx-auto pt-8 md:pt-16 animate-fade-in-up">
                <div className="text-center mb-12">
                   <div className="inline-block p-4 bg-primary-main rounded-2xl shadow-xl mb-6 transform hover:rotate-6 transition-transform">
                      <Scale className="w-12 h-12 text-accent-gold" />
                   </div>
                   <h1 className="text-4xl md:text-5xl font-bold text-primary-dark mb-4 leading-tight">
                      어떤 법률 자문이<br/>필요하신가요?
                   </h1>
                   <p className="text-lg text-neutral-500 max-w-xl mx-auto">
                      4개의 AI 모델이 당신의 대변인이 되어드립니다.<br/>
                      복잡한 법률 용어 대신 일상의 언어로 편하게 물어보세요.
                   </p>
                </div>

                <InputArea 
                  query={query} 
                  setQuery={setQuery} 
                  handleSubmit={handleSubmit} 
                  category={category} 
                  setCategory={setCategory}
                  isListening={isListening}
                  toggleListening={toggleListening}
                  files={files}
                  handleFileChange={handleFileChange}
                  removeFile={removeFile}
                  fileInputRef={fileInputRef}
                />

                {/* History Section */}
                {history.length > 0 && (
                   <div className="mt-16">
                      <div className="flex items-center justify-between mb-4 px-1">
                         <h3 className="font-bold text-neutral-700 flex items-center">
                            <Clock className="w-4 h-4 mr-2" /> 최근 상담 내역
                         </h3>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                         {history.slice(0, 4).map((record) => (
                            <div 
                              key={record.id}
                              onClick={() => loadHistoryItem(record)}
                              className="bg-white p-4 rounded-xl border border-neutral-200 shadow-sm hover:shadow-md hover:border-primary-main cursor-pointer transition-all group relative"
                            >
                               <div className="flex justify-between items-start mb-2">
                                  <Badge variant="neutral">{record.category}</Badge>
                                  <span className="text-xs text-neutral-400">
                                     {new Date(record.startDate).toLocaleDateString()}
                                  </span>
                               </div>
                               <h4 className="font-medium text-neutral-800 line-clamp-1 mb-1">
                                  {record.turns[0]?.query || "자문 내용 없음"}
                               </h4>
                               <p className="text-xs text-neutral-500 line-clamp-2">
                                  {record.turns[0]?.result.summary}
                               </p>
                               <button 
                                 onClick={(e) => handleDeleteHistory(e, record.id)}
                                 className="absolute top-2 right-2 p-1.5 rounded-full bg-neutral-100 text-neutral-400 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all"
                               >
                                  <Trash2 className="w-3 h-3" />
                               </button>
                            </div>
                         ))}
                      </div>
                   </div>
                )}
             </div>
          )}

          {/* Active Session View OR Processing First Query */}
          {(currentRecord || step !== 'input') && (
             <div className="w-full mx-auto">
                {/* Chat History */}
                {currentRecord && (
                  <div className="space-y-4">
                    {currentRecord.turns.map((turn) => renderChatTurn(turn))}
                  </div>
                )}

                {/* Active Visualization (Processing Next Turn) */}
                {step !== 'input' && step !== 'complete' && (
                   <div className="mt-8 mb-32 animate-fade-in" ref={bottomRef}>
                      {renderVisualizer()}
                      {renderLogs()}
                   </div>
                )}
             </div>
          )}
          
          <div ref={bottomRef} />
       </div>

       {/* Council Mode: Show "New Consultation" button instead of chat input */}
       {/* 위원회 모드는 1회성 자문이므로 추가 채팅 불가 - 새 자문 요청만 가능 */}
       {currentRecord && step === 'complete' && (
          <div className="fixed bottom-0 left-0 right-0 z-20 bg-white/95 backdrop-blur-md border-t border-neutral-200 p-4 shadow-[0_-5px_20px_rgba(0,0,0,0.05)]">
            <div className="max-w-4xl mx-auto flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 rounded-full bg-green-500"></div>
                <span className="text-sm text-neutral-600">
                  자문이 완료되었습니다. 추가 질문이 있으시면 <strong>새 자문을 요청</strong>하시거나
                  <a href="#/expert-chat" className="text-primary-main hover:underline ml-1">전문가 채팅</a>을 이용하세요.
                </span>
              </div>
              <button
                onClick={resetForm}
                className="px-6 py-2.5 bg-primary-main text-white rounded-lg font-medium hover:bg-primary-dark transition-colors shadow-md"
              >
                새 자문 요청
              </button>
            </div>
          </div>
       )}
    </div>
  );
};

export default Consultation;