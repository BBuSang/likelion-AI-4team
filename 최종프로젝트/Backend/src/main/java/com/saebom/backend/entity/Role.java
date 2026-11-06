package com.saebom.backend.entity;

/** 데이터 접근 역할 (worklog 3-4-1). AI 서버 app/core/access.py 의 ROLES 와 같은 글자 */
public enum Role {
    STAFF,   // 직원: 공개 + 같은 팀 담당자 있는 문서 + 본인 담당 문서, 복원은 본인 담당만
    LEAD,    // 팀장: 공개 + 같은 팀 전부 + 본인 담당, 복원도 같은 범위
    AUDITOR  // 감사: 전부, 복원은 사유 필수
}
