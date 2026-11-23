package com.saebom.backend.repository;

import com.saebom.backend.entity.ChatConversation;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ChatConversationRepository extends JpaRepository<ChatConversation, Long> {
    List<ChatConversation> findByUserIdOrderByUpdatedAtDesc(Long userId, Pageable page);
    Optional<ChatConversation> findByIdAndUserId(Long id, Long userId);

    /** 통계: 전체 사용자의 최근 대화 (질문이 하나라도 있어 제목이 붙은 것만) */
    Page<ChatConversation> findByTitleIsNotNullOrderByUpdatedAtDesc(Pageable page);
}
