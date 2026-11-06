package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;

/** 직급 선택칸 목록 (app.positions). default_role = 승인 때 연결 직원이 없으면 쓰는 역할 기본값 */
@Entity
@Table(name = "positions")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Position {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 30)
    private String name;

    @Column(name = "sort_order", nullable = false)
    private int sortOrder;

    @Enumerated(EnumType.STRING)
    @Column(name = "default_role", nullable = false, length = 10)
    private Role defaultRole;
}
