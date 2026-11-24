package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * 채팅 대화 (app.chat_conversations). 계정마다 따로, 본인과 관리자(조회 기록 남김)만 봄.
 * 제목은 첫 질문의 가명처리된 글 — 실명은 어디에도 저장하지 않음.
 */
@Entity
@Table(name = "chat_conversations", indexes = @Index(name = "chat_conv_user", columnList = "user_id, updated_at"))
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatConversation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id", nullable = false)
    private Long userId;

    /** 첫 질문의 가명처리된 글 앞부분. 첫 답이 오기 전엔 null */
    @Column(length = 200)
    private String title;

    /** 첫 질문 원문 앞부분의 암호문 (ChatCipher) — 대화 주인 목록에만 풀어서 제목으로 */
    @Column(name = "title_enc", columnDefinition = "text")
    private String titleEnc;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    /** 마지막 질문 시각 (목록 정렬) */
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;
}
