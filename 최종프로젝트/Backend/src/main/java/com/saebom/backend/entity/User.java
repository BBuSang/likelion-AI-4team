package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

/**
 * 직원 계정 (app.users).
 * 이전 프로젝트 User 에서 가져옴. 바뀐 점: 필드 이름 정리(Id → id, create_at → createdAt),
 * 직원 전화번호는 쓰는 곳이 없어 뺌(필요한 개인정보만 저장). 직책(position)은 다음 단계에서 추가.
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

    @Column(nullable = false, length = 20)
    @Builder.Default
    private String role = "USER"; // USER, ADMIN — 직책·권한으로 바꿀 예정

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
