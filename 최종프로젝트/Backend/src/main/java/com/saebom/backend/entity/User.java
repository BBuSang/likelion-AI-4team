package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

/**
 * 직원 계정 (app.users, worklog 11-4 확정안).
 * - role: 데이터 접근 역할 (STAFF/LEAD/AUDITOR). 관리자 페이지 권한은 admin(is_admin) 으로 따로
 * - empId: 문서에서 추론한 직원(AI 서버 직원 목록)과 연결. 승인 때 관리자가 정함
 * - 가입 때 고른 팀·직급은 본인 주장이라 그대로 믿지 않고, 관리자가 승인하면서 확인
 */
@Entity
@Table(name = "users")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 100)
    private String email;

    @Column(nullable = false)
    private String pw; // BCrypt 해시

    @Column(nullable = false, length = 50)
    private String name;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "team_id")
    private Team team;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "position_id")
    private Position position;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 10)
    @Builder.Default
    private Role role = Role.STAFF;

    @Column(name = "is_admin", nullable = false)
    @Builder.Default
    private boolean admin = false;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 10)
    @Builder.Default
    private UserStatus status = UserStatus.PENDING;

    @Column(name = "emp_id", unique = true, length = 20)
    private String empId;

    @Column(name = "approved_by")
    private Long approvedBy;

    @Column(name = "approved_at")
    private LocalDateTime approvedAt;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
