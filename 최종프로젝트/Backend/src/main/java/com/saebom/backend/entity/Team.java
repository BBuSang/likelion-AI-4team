package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;

/** 팀 선택칸 목록 (app.teams). name 은 문서 메타데이터 team 과 같은 글자여야 권한이 맞음 (worklog 11-4) */
@Entity
@Table(name = "teams")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Team {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 50)
    private String name;

    @Column(nullable = false)
    @Builder.Default
    private boolean active = true;

    /** ai = 문서에서 추론한 팀 (AI 서버), admin = 관리자가 추가 */
    @Column(nullable = false, length = 10)
    @Builder.Default
    private String source = "admin";
}
