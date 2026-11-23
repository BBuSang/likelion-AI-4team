// 답변 스트리밍 중 진행 단계 (worklog 60장): 끝난 단계는 ✓ + 걸린 시간 + 결과(출처·건수), 지금 단계는 깜빡이는 점.
// 답 글이 오기 시작하면 한 줄 요약으로 줄여 글을 밀어내지 않음.
import { Check } from 'lucide-react';
import type { StageData, StreamStage } from '../api/chat.api';

export interface Step {
  stage: StreamStage;
  start: number; // performance.now()
  end?: number;
  data: StageData; // 시작 때 값
  result?: StageData; // step_done 값
}

const SOURCE_KO: Record<string, string> = { internal: '사내 기록', law: '법령', product: '금융상품', noaction: '비조치의견서' };
const TOOL_KO: Record<string, string> = { search_internal: '사내 기록', search_law: '법령', search_product: '금융상품', search_noaction: '비조치의견서' };
const ROUTE_KO: Record<string, string> = { out_of_scope: '업무 범위 밖', clarify: '무엇을 묻는지 확인 필요' };

const names = (xs?: string[], map = SOURCE_KO) => (xs ?? []).map((x) => map[x] ?? x).join('·');

/** 단계 이름 (명사형: 끝난 단계엔 그대로, 지금 단계엔 '… 중') */
function label(s: Step) {
  switch (s.stage) {
    case 'protect': return '질문 속 개인정보 가리기';
    case 'classify': return '질문 분류';
    case 'followup': return '앞 대화 보고 질문 다시 쓰기';
    case 'plan': return '검색 계획 세우기';
    case 'search': return `${names(s.data.sources) || '자료'} 검색`;
    case 'replan': return '사내 결과 보고 계획 다시 세우기';
    case 'search2': return `${names(s.data.tools, TOOL_KO) || '추가'} 추가 검색`;
    case 'judge': return '사례가 질문에 직접 해당하는지 판단';
    case 'relax': return '키워드로 넓혀 다시 찾기';
    case 'answer': return '답변 생성';
    case 'check': return '답변 개인정보 재검사';
    default: return s.stage;
  }
}

/** 단계 결과 한 줄 (있을 때만) */
function detail(s: Step) {
  const r = s.result;
  if (!r) return '';
  if (s.stage === 'classify') return ROUTE_KO[r.route ?? ''] ?? names(r.sources);
  if (s.stage === 'plan' && r.steps) return `검색 ${r.steps}개`;
  if (s.stage === 'search' && r.counts) {
    return Object.entries(r.counts).map(([k, n]) => `${SOURCE_KO[k] ?? k} ${n}건`).join(' · ');
  }
  if (s.stage === 'followup' && r.rewritten != null) return r.rewritten ? '이어서 물음' : '대상을 못 정함';
  if (s.stage === 'judge' && r.total != null) return `직접 해당 ${r.direct ?? 0}/${r.total}건`;
  return '';
}

const secs = (s: Step) => (s.end != null ? `${((s.end - s.start) / 1000).toFixed(1)}초` : '');

export default function StreamProgress({ steps, writing }: { steps: Step[]; writing: boolean }) {
  const cur = steps[steps.length - 1];
  if (!cur) return null;
  if (writing) {  // 답 글이 오는 중: 한 줄로
    const total = ((performance.now() - steps[0].start) / 1000).toFixed(1);
    return (
      <p className="small muted stream-stage" role="status">
        <span className="stream-dot" aria-hidden />답변 작성 중… <span className="stream-sum">{steps.length - 1}단계 완료 · {total}초</span>
      </p>
    );
  }
  return (
    <ol className="stream-steps" aria-live="polite">
      {steps.map((s, i) => {
        const now = i === steps.length - 1;
        const d = detail(s);
        return (
          <li key={i} className={now ? 'now' : 'done'}>
            {now ? <span className="stream-dot" aria-hidden /> : <Check size={13} className="stream-check" aria-hidden />}
            <span className="stream-label">{label(s)}{now ? ' 중…' : ''}</span>
            {d && <span className="stream-detail">{d}</span>}
            {!now && <span className="stream-time">{secs(s)}</span>}
          </li>
        );
      })}
    </ol>
  );
}
