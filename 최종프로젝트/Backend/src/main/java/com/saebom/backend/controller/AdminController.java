package com.saebom.backend.controller;

import com.saebom.backend.ai.AiClient.Employee;
import com.saebom.backend.dto.AdminDtos.*;
import com.saebom.backend.entity.UserStatus;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.AdminService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/** 관리자 페이지 API. /api/admin/** 는 SecurityConfig 에서 ROLE_ADMIN 만 */
@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final AdminService admin;

    @GetMapping("/users")
    public List<AdminUserResponse> users(@RequestParam(required = false) UserStatus status) {
        return admin.listUsers(status);
    }

    @GetMapping("/users/{id}")
    public AdminUserResponse user(@PathVariable Long id) {
        return admin.getUser(id);
    }

    /** 승인 화면: 가입 정보로 찾은 직원 후보 */
    @GetMapping("/users/{id}/candidates")
    public CandidatesResponse candidates(@PathVariable Long id) {
        return admin.candidates(id);
    }

    @PostMapping("/users/{id}/approve")
    public AdminUserResponse approve(@AuthenticationPrincipal UD me, @PathVariable Long id,
                                     @RequestBody ApproveRequest req) {
        return admin.approve(me.getUser().getId(), id, req);
    }

    @PutMapping("/users/{id}")
    public AdminUserResponse update(@AuthenticationPrincipal UD me, @PathVariable Long id,
                                    @RequestBody UpdateUserRequest req) {
        return admin.update(me.getUser().getId(), id, req);
    }

    /** 직원 목록 (이름 포함) — 후보가 없거나 여럿일 때 관리자가 직접 찾기 */
    @GetMapping("/employees")
    public List<Employee> employees(@RequestParam(required = false) String q) {
        return admin.employees(q);
    }

    @GetMapping("/teams")
    public List<TeamResponse> teams() {
        return admin.listTeams();
    }

    @PostMapping("/teams")
    @ResponseStatus(HttpStatus.CREATED)
    public TeamResponse createTeam(@AuthenticationPrincipal UD me, @Valid @RequestBody TeamRequest req) {
        return admin.createTeam(me.getUser().getId(), req.name());
    }

    @PutMapping("/teams/{id}")
    public TeamResponse updateTeam(@AuthenticationPrincipal UD me, @PathVariable Long id,
                                   @RequestBody TeamUpdateRequest req) {
        return admin.updateTeam(me.getUser().getId(), id, req);
    }

    /** 문서에서 추론한 팀 이름 가져오기 (AI 서버) */
    @PostMapping("/teams/sync")
    public SyncResponse syncTeams(@AuthenticationPrincipal UD me) {
        return admin.syncTeams(me.getUser().getId());
    }

    @GetMapping("/audit-log")
    public List<AuditLogResponse> auditLog(@RequestParam(required = false) Long userId,
                                           @RequestParam(defaultValue = "100") int limit) {
        return admin.auditLog(userId, limit);
    }
}
