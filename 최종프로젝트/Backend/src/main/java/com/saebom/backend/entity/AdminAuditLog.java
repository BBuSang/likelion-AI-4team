package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/** 관리자가 계정·팀을 바꾼 기록 (app.admin_audit_log). 역할을 감사로 올리는 것 같은 위험한 변경을 추적 */
@Entity
@Table(name = "admin_audit_log", indexes = @Index(name = "admin_audit_log_target", columnList = "target_user_id"))
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AdminAuditLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 0 = 서버 자신 (첫 관리자 만들기·팀 자동 동기화) */
    @Column(name = "admin_id", nullable = false)
    private Long adminId;

    @Column(name = "target_user_id")
    private Long targetUserId;

    /** approve / update / create_employee / create_team / update_team / sync_teams / bootstrap */
    @Column(nullable = false, length = 20)
    private String action;

    /** 바뀐 값 전후 JSON: {"role": ["STAFF", "LEAD"], ...} */
    @Column(nullable = false, length = 4000)
    private String changes;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
