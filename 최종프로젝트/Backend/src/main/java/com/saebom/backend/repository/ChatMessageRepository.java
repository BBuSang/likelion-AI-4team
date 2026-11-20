package com.saebom.backend.repository;

import com.saebom.backend.entity.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import java.util.Collection;
import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    List<ChatMessage> findByConversationIdOrderByIdAsc(Long conversationId);

    @Query("select m.conversationId, count(m) from ChatMessage m where m.conversationId in :ids group by m.conversationId")
    List<Object[]> countByConversation(Collection<Long> ids);

    @Modifying
    @Query("delete from ChatMessage m where m.conversationId = :conversationId")
    void deleteByConversation(Long conversationId);
}
