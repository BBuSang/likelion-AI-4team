// 브라우저에 저장하는 로그인 사용자 정보. 실명 복원 값 같은 개인정보는 여기 두지 않음
export type Role = 'USER' | 'ADMIN';

export interface StoredUser {
  email: string;
  name: string;
  role: Role;
}
