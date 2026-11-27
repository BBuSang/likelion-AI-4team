// 표 밑에 조용히 붙는 작은 쪽 버튼 (모든 표 공통): 테두리 없는 글자 버튼, 지금 쪽만 옅은 배경, 이전·다음은 작은 화살표
import { ChevronLeft, ChevronRight } from 'lucide-react';

/** 이전·다음 + 쪽 번호. pages 가 1 이하면 안 보임 */
export default function Pager({ page, pages, onChange }: { page: number; pages: number; onChange: (p: number) => void }) {
  if (pages <= 1) return null;
  // 현재 쪽 앞뒤 2개까지만 번호로 (많으면 처음·끝은 … 로)
  const nums = Array.from({ length: pages }, (_, i) => i).filter((i) => i === 0 || i === pages - 1 || Math.abs(i - page) <= 2);
  // 표 밑에 작게: 테두리 없는 글자 버튼, 지금 쪽만 옅은 배경
  return (
    <nav className="pager" aria-label="페이지">
      <button className="pager-btn" disabled={page === 0} onClick={() => onChange(page - 1)} aria-label="이전 페이지">
        <ChevronLeft size={14} aria-hidden />
      </button>
      {nums.map((i, k) => (
        <span key={i} className="pager-item">
          {k > 0 && i - nums[k - 1] > 1 && <span className="pager-gap">…</span>}
          <button className={`pager-btn${i === page ? ' on' : ''}`} aria-current={i === page ? 'page' : undefined}
                  onClick={() => onChange(i)}>{i + 1}</button>
        </span>
      ))}
      <button className="pager-btn" disabled={page >= pages - 1} onClick={() => onChange(page + 1)} aria-label="다음 페이지">
        <ChevronRight size={14} aria-hidden />
      </button>
    </nav>
  );
}
