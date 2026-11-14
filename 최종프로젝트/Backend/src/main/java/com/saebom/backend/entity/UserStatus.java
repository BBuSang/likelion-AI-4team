package com.saebom.backend.entity;

/** 계정 상태. ACTIVE 만 로그인·API 사용 가능 */
public enum UserStatus {
    PENDING,   // 가입 후 관리자 승인 전
    ACTIVE,
    SUSPENDED
}
