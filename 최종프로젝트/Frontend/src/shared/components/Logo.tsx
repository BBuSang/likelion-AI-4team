// 사공삼(403) 로고 — 팀이 만든 로고 파일을 그대로 씀
// - public/favicon.svg        : 파란 둥근 사각형 + 자물쇠(403 의 0) → 브라우저 탭·좁은 메뉴 마크
// - public/brand/logo-white.svg : 흰 403 (가운데 0 이 자물쇠, 위아래 흐린 403) → 파란 바탕 (로그인·가입·재설정 카드)
// - public/brand/logo-blue.svg  : 파란 403 → 흰 바탕 (채팅 첫 화면)
// - public/brand/logo-blue-row.svg : 파란 로고 가운데 줄만 (가로) → 왼쪽 메뉴 맨 위

/** 작은 마크 (자물쇠 아이콘) */
export function LogoMark({ size = 28 }: { size?: number }) {
  return <img src="/favicon.svg" width={size} height={size} alt="" aria-hidden style={{ flex: 'none', display: 'block' }} />;
}

/** 가로 로고: 파란 로고의 가운데 줄(4·자물쇠·3)만 — public/brand/logo-blue-row.svg (1080.55:420) */
export function LogoRow({ height = 24, className }: { height?: number; className?: string }) {
  return (
    <img src="/brand/logo-blue-row.svg" alt="사공삼(403)" height={height} width={Math.round((height * 1080.55) / 420)}
         className={className} style={{ display: 'block' }} />
  );
}

/** 큰 로고 (세로 1080:1360). tone = 바탕색에 맞춰 white(파란 바탕) / blue(흰 바탕) */
export function LogoLarge({ tone, width, className }: { tone: 'white' | 'blue'; width: number; className?: string }) {
  return (
    <img src={`/brand/logo-${tone}.svg`} alt="사공삼(403)" width={width} height={Math.round((width * 1360) / 1080.55)}
         className={className} style={{ display: 'block' }} />
  );
}
