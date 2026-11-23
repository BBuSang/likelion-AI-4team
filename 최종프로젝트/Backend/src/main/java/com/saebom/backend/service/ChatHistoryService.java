package com.saebom.backend.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient.AskResult;
import com.saebom.backend.entity.ChatConversation;
import com.saebom.backend.entity.ChatMessage;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.ChatConversationRepository;
import com.saebom.backend.repository.ChatMessageRepository;
import com.saebom.backend.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import com.saebom.backend.entity.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 채팅 내역 서버 저장 (worklog 61장). 브라우저 탭에만 두던 대화를 계정별로 DB(app 스키마)에 저장.
 * - 저장은 화면이 아니라 서버가: 질문 응답(AskResult)을 받은 그 자리에서 저장 → 화면이 꾸민 답을 끼워 넣을 수 없음
 * - 질문은 가명처리된 글(maskedQuery) + 원문은 암호문으로만(ChatCipher, 64장, 주인에게만 풀어 줌). 답은 토큰 상태, 실명 보기 결과는 저장 안 함
 * - 본인 대화만 보고 지움. 관리자는 사용자 화면에서 볼 수 있고, 대화 내용을 열 때마다 admin_audit_log 에 남김
 */
@Service
@RequiredArgsConstructor
public class ChatHistoryService {

    static final int LIST_LIMIT = 100;
    static final int TITLE_LEN = 60;

    private final ChatConversationRepository convs;
    private final ChatMessageRepository msgs;
    private final UserRepository users;
    private final AdminService admin;
    private final ChatCipher cipher;
    private final ObjectMapper om = new ObjectMapper();

    public record ConversationSummary(Long id, String title, int messages,
                                      LocalDateTime createdAt, LocalDateTime updatedAt) {}

    /** 질문 하나 + 답변. result 는 /api/chat/ask 응답과 같은 모양.
     *  query = 사용자가 친 질문 원문 — 대화 주인이 볼 때만 채움 (관리자 열람·못 풀면 null → 화면은 maskedQuery) */
    public record MessageView(Long id, String query, String maskedQuery, AskResult result, LocalDateTime createdAt) {}

    public record ConversationDetail(Long id, String title, LocalDateTime createdAt, LocalDateTime updatedAt,
                                     List<MessageView> messages) {}

    /** 통계 화면의 최근 대화 한 줄: 계정·역할·팀 + 가명처리된 제목·질문 수 (내용·원문 질문은 없음) */
    public record RecentConversation(Long id, Long userId, String email, String role, String team,
                                     String title, int messages, LocalDateTime updatedAt) {}

    public record RecentPage(List<RecentConversation> items, long total, int page, int size) {}

    // ------------------------------------------------------------------ 본인

    @Transactional
    public ConversationSummary create(Long userId) {
        ChatConversation c = convs.save(ChatConversation.builder().userId(userId).updatedAt(LocalDateTime.now()).build());
        return new ConversationSummary(c.getId(), c.getTitle(), 0, c.getCreatedAt(), c.getUpdatedAt());
    }

    @Transactional(readOnly = true)
    public List<ConversationSummary> list(Long userId) {
        return list(userId, true);
    }

    private List<ConversationSummary> list(Long userId, boolean owner) {
        List<ChatConversation> list = convs.findByUserIdOrderByUpdatedAtDesc(userId, PageRequest.of(0, LIST_LIMIT));
        Map<Long, Integer> counts = new HashMap<>();
        if (!list.isEmpty()) {
            for (Object[] row : msgs.countByConversation(list.stream().map(ChatConversation::getId).toList())) {
                counts.put((Long) row[0], ((Number) row[1]).intValue());
            }
        }
        return list.stream().map(c -> new ConversationSummary(c.getId(), titleFor(c, owner), counts.getOrDefault(c.getId(), 0),
                c.getCreatedAt(), c.getUpdatedAt())).toList();
    }

    @Transactional(readOnly = true)
    public ConversationDetail detail(Long userId, Long convId) {
        return detailOf(owned(userId, convId), true);
    }

    /** 질문 전에 부름: 남의 대화 번호로 질문하면 AI 를 부르기 전에 막음 */
    @Transactional(readOnly = true)
    public void checkOwner(Long userId, Long convId) {
        owned(userId, convId);
    }

    @Transactional
    public void delete(Long userId, Long convId) {
        ChatConversation c = owned(userId, convId);
        msgs.deleteByConversation(c.getId());
        convs.delete(c);
    }

    /** 질문 응답을 대화에 붙임. rawQuery = 사용자가 친 질문 (암호문으로만 저장). 제목이 없으면 첫 질문(가명처리된 글)으로 */
    @Transactional
    public void append(Long userId, Long convId, String rawQuery, AskResult r) {
        if (convId == null || r == null) return;
        ChatConversation c = owned(userId, convId);
        String q = r.maskedQuery() == null ? "" : r.maskedQuery().strip();
        String json;
        try {
            json = om.writeValueAsString(r);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("답변을 저장하지 못했습니다", e);
        }
        msgs.save(ChatMessage.builder().conversationId(c.getId()).maskedQuery(q)
                .queryEnc(rawQuery == null ? null : cipher.encrypt(rawQuery)).result(json).build());
        if (c.getTitle() == null && !q.isEmpty()) {
            c.setTitle(cut(q));
            if (rawQuery != null && !rawQuery.isBlank()) c.setTitleEnc(cipher.encrypt(cut(rawQuery.strip())));
        }
        c.setUpdatedAt(LocalDateTime.now());
    }

    // ------------------------------------------------------------------ 관리자

    @Transactional(readOnly = true)
    public List<ConversationSummary> listFor(Long targetUserId) {
        users.findById(targetUserId).orElseThrow(() -> ApiException.notFound("사용자 없음"));
        return list(targetUserId, false);
    }

    /** 관리자가 남의 대화 내용을 열면 기록 (누가·누구의·어느 대화를) */
    @Transactional
    public ConversationDetail detailFor(Long adminId, Long targetUserId, Long convId) {
        ChatConversation c = owned(targetUserId, convId);
        Map<String, Object[]> what = new java.util.LinkedHashMap<>();
        what.put("conversation", new Object[]{null, c.getId()});
        what.put("title", new Object[]{null, c.getTitle()});  // 가명처리된 첫 질문 — 화면에 "어느 대화"로 보여 줌
        admin.log(adminId, targetUserId, "view_chat", what);
        return detailOf(c, false);  // 관리자에겐 원문 질문을 풀지 않음
    }

    /** 통계: 전체 사용자의 최근 대화 (페이지 단위). 제목은 가명처리된 첫 질문만 — 내용을 여는 게 아니라 기록은 남기지 않음 */
    @Transactional(readOnly = true)
    public RecentPage recent(int page, int size) {
        int p = Math.max(0, page);
        int n = Math.min(Math.max(1, size), 50);
        Page<ChatConversation> res = convs.findByTitleIsNotNullOrderByUpdatedAtDesc(PageRequest.of(p, n));
        List<ChatConversation> list = res.getContent();
        Map<Long, Integer> counts = new HashMap<>();
        Map<Long, User> owners = new HashMap<>();
        if (!list.isEmpty()) {
            for (Object[] row : msgs.countByConversation(list.stream().map(ChatConversation::getId).toList())) {
                counts.put((Long) row[0], ((Number) row[1]).intValue());
            }
            users.findAllById(list.stream().map(ChatConversation::getUserId).distinct().toList())
                    .forEach(u -> owners.put(u.getId(), u));
        }
        List<RecentConversation> items = list.stream().map(c -> {
            User u = owners.get(c.getUserId());
            return new RecentConversation(c.getId(), c.getUserId(),
                    u == null ? null : u.getEmail(),
                    u == null || u.getRole() == null ? null : u.getRole().name(),
                    u == null || u.getTeam() == null ? null : u.getTeam().getName(),
                    c.getTitle(), counts.getOrDefault(c.getId(), 0), c.getUpdatedAt());
        }).toList();
        return new RecentPage(items, res.getTotalElements(), p, n);
    }

    // ------------------------------------------------------------------

    private static String cut(String s) {
        return s.length() > TITLE_LEN ? s.substring(0, TITLE_LEN) + "…" : s;
    }

    /** 대화 주인에겐 원문 제목, 못 풀거나 관리자면 가명처리된 제목 */
    private String titleFor(ChatConversation c, boolean owner) {
        String raw = owner ? cipher.decrypt(c.getTitleEnc()) : null;
        return raw != null ? raw : c.getTitle();
    }

    private ChatConversation owned(Long userId, Long convId) {
        return convs.findByIdAndUserId(convId, userId).orElseThrow(() -> ApiException.notFound("대화를 찾을 수 없습니다."));
    }

    private ConversationDetail detailOf(ChatConversation c, boolean owner) {
        List<MessageView> list = msgs.findByConversationIdOrderByIdAsc(c.getId()).stream()
                .map(m -> new MessageView(m.getId(), owner ? cipher.decrypt(m.getQueryEnc()) : null,
                        m.getMaskedQuery(), parse(m.getResult()), m.getCreatedAt())).toList();
        return new ConversationDetail(c.getId(), titleFor(c, owner), c.getCreatedAt(), c.getUpdatedAt(), list);
    }

    private AskResult parse(String json) {
        try {
            return om.readValue(json, AskResult.class);
        } catch (JsonProcessingException e) {
            return null;  // 옛 형식 등 못 읽는 한 건 때문에 대화 전체가 안 열리지 않게
        }
    }
}
