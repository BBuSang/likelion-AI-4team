package com.saebom.backend.controller;

import com.saebom.backend.ai.AccessUser;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.AskResult;
import com.saebom.backend.ai.AiClient.RestoreResult;
import com.saebom.backend.entity.Role;
import com.saebom.backend.entity.User;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.ChatHistoryService;
import com.saebom.backend.service.ChatHistoryService.ConversationDetail;
import com.saebom.backend.service.ChatHistoryService.ConversationSummary;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

/**
 * 질문·실명 복원 중계. 로그인한 ACTIVE 계정만.
 * 묻는 사람 정보 {emp_id, team, role} 는 화면이 보내는 값이 아니라 요청마다 DB 에서 읽은 계정 값 (AccessUser.of).
 * 권한 판단(검색 전 거르기·복원 범위)은 AI 서버 access.py 가 하고, 스프링은 미리 막을 수 있는 것만 막음.
 */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@lombok.extern.slf4j.Slf4j
public class ChatController {

    private final AiClient ai;
    private final ChatHistoryService history;

    /** conversationId: 이 대화에 질문·답변을 저장 (없으면 저장 안 함). 내 대화가 아니면 AI 를 부르기 전에 404 */
    public record AskRequest(@NotBlank @Size(max = 1000) String query,
                             @Min(1) @Max(10) Integer k,
                             Long conversationId) {}

    /** text = 질문 응답의 answer (토큰 상태), docIds = 그 응답의 docs 의 docId */
    public record RestoreRequest(@NotBlank @Size(max = 20_000) String text,
                                 @NotNull @Size(max = 50) List<String> docIds,
                                 @Size(max = 500) String purpose) {}

    @PostMapping("/ask")
    public AskResult ask(@AuthenticationPrincipal UD me, @Valid @RequestBody AskRequest req) {
        User u = linked(me);
        var turns = history.recentTurns(u.getId(), req.conversationId());  // 남의 대화면 여기서 404
        AskResult r = ai.ask(AccessUser.of(u), req.query().trim(), req.k() == null ? 3 : req.k(), turns);
        history.append(u.getId(), req.conversationId(), req.query().trim(), r);
        return r;
    }

    /**
     * 답변 스트리밍. SSE: stage · delta(출력 검사를 거친 답변 조각) · done(ask 와 같은 결과) · error.
     * 비동기(SseEmitter) 대신 이 요청 스레드에서 바로 써서 흘려보냄 — 비동기 재전달 때 JWT 인증이 다시 안 붙는 문제를 피함.
     * AI 가 흐름을 열기 전에 거절하면 보통 오류 응답(JSON)으로.
     */
    @PostMapping("/ask/stream")
    public void askStream(@AuthenticationPrincipal UD me, @Valid @RequestBody AskRequest req, HttpServletResponse res)
            throws IOException {
        User u = linked(me);
        var turns = history.recentTurns(u.getId(), req.conversationId());  // 남의 대화면 여기서 404
        ai.askStream(AccessUser.of(u), req.query().trim(), req.k() == null ? 3 : req.k(), turns, () -> {
            res.setStatus(HttpServletResponse.SC_OK);
            res.setContentType("text/event-stream");
            res.setCharacterEncoding("UTF-8");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("X-Accel-Buffering", "no");
        }, res.getOutputStream(), r -> {
            try {
                history.append(u.getId(), req.conversationId(), req.query().trim(), r);
            } catch (RuntimeException e) {  // 답은 이미 화면에 감 — 저장 실패로 흐름을 깨지 않음
                log.warn("채팅 내역 저장 실패 (대화 {}): {}", req.conversationId(), e.toString());
            }
        });
    }

    // ------------------------------------------------------------------ 채팅 내역 (본인 것만)

    @GetMapping("/conversations")
    public List<ConversationSummary> conversations(@AuthenticationPrincipal UD me) {
        return history.list(me.getUser().getId());
    }

    @PostMapping("/conversations")
    public ConversationSummary newConversation(@AuthenticationPrincipal UD me) {
        return history.create(me.getUser().getId());
    }

    @GetMapping("/conversations/{id}")
    public ConversationDetail conversation(@AuthenticationPrincipal UD me, @PathVariable Long id) {
        return history.detail(me.getUser().getId(), id);
    }

    @DeleteMapping("/conversations/{id}")
    public void deleteConversation(@AuthenticationPrincipal UD me, @PathVariable Long id) {
        history.delete(me.getUser().getId(), id);
    }

    @PostMapping("/restore")
    public RestoreResult restore(@AuthenticationPrincipal UD me, @Valid @RequestBody RestoreRequest req) {
        User u = linked(me);
        String purpose = req.purpose() == null ? null : req.purpose().trim();
        // 팀장·감사는 사유 필수. 사원은 자기 담당 문서만 복원되므로 사유 없이
        if ((u.getRole() == Role.AUDITOR || u.getRole() == Role.LEAD) && (purpose == null || purpose.isEmpty())) {
            ai.logRefused(AccessUser.of(u), u.getEmail(), req.docIds(), "no_purpose");  // 거절도 기록 (이상 접근 탐지, 39장)
            throw ApiException.badRequest("팀장·감사 역할의 실명 보기는 사유를 입력해야 합니다.");
        }
        return ai.restore(AccessUser.of(u), u.getEmail(), req.text(), req.docIds(), purpose);
    }

    /** 감사 말고는 직원 연결(emp_id)이 있어야 함 — 관리자 계정처럼 연결 없는 계정은 사내 문서를 못 봄 */
    private static User linked(UD me) {
        User u = me.getUser();
        if (u.getRole() != Role.AUDITOR && u.getEmpId() == null) {
            throw ApiException.forbidden("직원 정보가 연결되지 않은 계정입니다. 관리자에게 문의하세요.");
        }
        return u;
    }
}
