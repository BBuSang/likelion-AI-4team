package com.saebom.backend.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * 질문 하나 + 답변 하나 (app.chat_messages).
 * 질문은 AI 서버가 가명처리한 글(maskedQuery) + 원문은 암호문으로만(queryEnc), 답변·근거 문서는 토큰 상태 그대로.
 * 실명 보기 결과는 저장하지 않음 (화면에만).
 */
@Entity
@Table(name = "chat_messages", indexes = @Index(name = "chat_msg_conv", columnList = "conversation_id, id"))
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatMessage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "conversation_id", nullable = false)
    private Long conversationId;

    @Column(name = "masked_query", nullable = false, columnDefinition = "text")
    private String maskedQuery;

    /** 사용자가 친 질문 원문, AES-GCM 암호문 (ChatCipher). 대화 주인에게만 풀어서 보여 줌 */
    @Column(name = "query_enc", columnDefinition = "text")
    private String queryEnc;

    /** 질문 응답 전체 (AskResult JSON: 답변·근거 문서·분류·출력 검사·시간) */
    @Column(nullable = false, columnDefinition = "text")
    private String result;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
