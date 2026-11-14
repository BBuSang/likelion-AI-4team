package com.saebom.backend.controller;

import com.saebom.backend.ai.AccessUser;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.AskResult;
import com.saebom.backend.ai.AiClient.RestoreResult;
import com.saebom.backend.entity.Role;
import com.saebom.backend.entity.User;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.security.UD;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 질문·실명 복원 중계 (worklog 13장). 로그인한 ACTIVE 계정만.
 * 묻는 사람 정보 {emp_id, team, role} 는 화면이 보내는 값이 아니라 요청마다 DB 에서 읽은 계정 값 (AccessUser.of).
 * 권한 판단(검색 전 거르기·복원 범위)은 AI 서버 access.py 가 하고, 스프링은 미리 막을 수 있는 것만 막음.
 */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatController {

    private final AiClient ai;

    public record AskRequest(@NotBlank @Size(max = 1000) String query,
                             @Min(1) @Max(10) Integer k) {}

    /** text = 질문 응답의 answer (토큰 상태), docIds = 그 응답의 docs 의 docId */
    public record RestoreRequest(@NotBlank @Size(max = 20_000) String text,
                                 @NotNull @Size(max = 50) List<String> docIds,
                                 @Size(max = 500) String purpose) {}

    @PostMapping("/ask")
    public AskResult ask(@AuthenticationPrincipal UD me, @Valid @RequestBody AskRequest req) {
        User u = linked(me);
        return ai.ask(AccessUser.of(u), req.query().trim(), req.k() == null ? 3 : req.k());
    }

    @PostMapping("/restore")
    public RestoreResult restore(@AuthenticationPrincipal UD me, @Valid @RequestBody RestoreRequest req) {
        User u = linked(me);
        String purpose = req.purpose() == null ? null : req.purpose().trim();
        // 팀장·감사는 사유 필수 (worklog 25-9). 사원은 자기 담당 문서만 복원되므로 사유 없이
        if ((u.getRole() == Role.AUDITOR || u.getRole() == Role.LEAD) && (purpose == null || purpose.isEmpty())) {
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
