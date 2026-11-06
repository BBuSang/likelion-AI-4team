package com.saebom.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.dto.AuthDtos.ApiResponse;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.UserService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.annotation.AuthenticationPrincipal;

import java.util.Map;

/** 마이페이지 (본인 것만) + 관리자 통계 */
@RestController
@RequiredArgsConstructor
public class MeController {

    private final UserService userService;
    private final AiClient ai;

    public record PasswordRequest(@NotBlank String currentPassword,
                                  @NotBlank @Size(min = 8, max = 72) String newPassword) {}

    /** 비밀번호 변경: 로그인 + 현재 비밀번호 확인 필수 (이전 프로젝트의 문제 5-3 고침) */
    @PutMapping("/api/me/password")
    public ApiResponse password(@AuthenticationPrincipal UD me, @Valid @RequestBody PasswordRequest req) {
        userService.changePassword(me.getUser().getId(), req.currentPassword(), req.newPassword());
        return new ApiResponse(true, "비밀번호를 바꿨어요.");
    }

    /** 내 실명 보기 기록 (개수만, 실명·토큰 없음) */
    @GetMapping("/api/me/restore-log")
    public JsonNode myRestoreLog(@AuthenticationPrincipal UD me) {
        return ai.getJson("/restore-log", Map.of("actor", me.getEmail(), "limit", 50));
    }

    /** 통계 (관리자만, SecurityConfig /api/admin/**) */
    @GetMapping("/api/admin/stats")
    public JsonNode stats(@RequestParam(defaultValue = "14") int days) {
        return ai.getJson("/stats", Map.of("days", days));
    }

    /** 전체 실명 보기 기록 (관리자만) */
    @GetMapping("/api/admin/restore-log")
    public JsonNode restoreLog(@RequestParam(defaultValue = "100") int limit) {
        return ai.getJson("/restore-log", Map.of("limit", limit));
    }
}
