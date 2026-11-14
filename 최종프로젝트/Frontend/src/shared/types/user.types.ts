// 브라우저에 저장하는 로그인 사용자 정보. 실명 복원 값 같은 개인정보는 여기 두지 않음
// 데이터 접근 역할 (백엔드 entity/Role.java). 관리자 페이지 권한은 admin 으로 따로
export type Role = 'STAFF' | 'LEAD' | 'AUDITOR';

export type UserStatus = 'PENDING' | 'ACTIVE' | 'SUSPENDED';

export interface StoredUser {
  email: string;
  name: string;
  team: string | null;
  position: string | null;
  role: Role;
  admin: boolean;
}
