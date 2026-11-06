package com.saebom.backend.ai;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.saebom.backend.entity.User;

/**
 * 질문·복원 요청 때 AI 서버에 넘길 사용자 정보 (AI app/core/access.py 의 user dict 와 같은 모양).
 * 요청마다 DB 에서 읽은 계정 값으로 만듦. 계정 값(팀·역할)이 문서에서 추론한 직원 정보보다 우선 (worklog 11-4)
 */
public record AccessUser(@JsonProperty("emp_id") String empId, String team, String role) {

    public static AccessUser of(User u) {
        return new AccessUser(u.getEmpId(), u.getTeam() == null ? null : u.getTeam().getName(), u.getRole().name());
    }
}
