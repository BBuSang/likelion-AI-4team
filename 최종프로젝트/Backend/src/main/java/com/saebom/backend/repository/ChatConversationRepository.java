package com.saebom.backend.repository;

import com.saebom.backend.entity.ChatConversation;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ChatConversationRepository extends JpaRepository<ChatConversation, Long> {
    List<ChatConversation> findByUserIdOrderByUpdatedAtDesc(Long userId, Pageable page);
    Optional<ChatConversation> findByIdAndUserId(Long id, Long userId);
}
