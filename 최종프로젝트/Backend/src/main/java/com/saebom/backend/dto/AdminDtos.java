package com.saebom.backend.dto;

import com.saebom.backend.ai.AiClient.Employee;
import com.saebom.backend.entity.AdminAuditLog;
import com.saebom.backend.entity.Role;
import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import java.time.LocalDateTime;
import java.util.List;

/** 관리자 페이지 요청/응답 (관리자만). 직원 이름이 들어가는 응답도 있음 */
public final class AdminDtos {

    private AdminDtos() {}

    public record AdminUserResponse(Long id, String email, String name,
                                    Long teamId, String team, Long positionId, String position,
                                    Role role, boolean admin, UserStatus status, String empId,
                                    Long approvedBy, LocalDateTime approvedAt, LocalDateTime createdAt) {
        public static AdminUserResponse of(User u) {
            return new AdminUserResponse(u.getId(), u.getEmail(), u.getName(),
                    u.getTeam() == null ? null : u.getTeam().getId(),
                    u.getTeam() == null ? null : u.getTeam().getName(),
                    u.getPosition() == null ? null : u.getPosition().getId(),
                    u.getPosition() == null ? null : u.getPosition().getName(),
                    u.getRole(), u.isAdmin(), u.getStatus(), u.getEmpId(),
                    u.getApprovedBy(), u.getApprovedAt(), u.getCreatedAt());
        }
    }

    /**
     * 승인 화면이 처음 열릴 때: 가입 정보로 찾은 직원 후보 + 역할 기본값.
     * linkedTo: 후보가 이미 다른 계정에 연결돼 있으면 그 계정 id (고르면 409)
     */
    public record Candidate(Employee employee, Long linkedTo) {}

    public record CandidatesResponse(List<Candidate> candidates, Role positionDefaultRole) {}

    /**
     * 승인. empId(기존 직원 연결) 또는 createEmployee=true(안 A: 새 직원 번호) 중 하나.
     * teamId·positionId·role 이 있으면 가입 값을 고쳐서 승인, role 이 없으면 기본값
     * (연결 직원의 추론 역할 → 없으면 직급 기본 역할)
     */
    public record ApproveRequest(String empId, boolean createEmployee, Long teamId, Long positionId, Role role) {}

    /** 비어 있는(null) 값은 안 바꿈. unlinkEmployee=true 면 직원 연결 해제 */
    public record UpdateUserRequest(Long teamId, Long positionId, Role role, Boolean admin,
                                    UserStatus status, String empId, Boolean unlinkEmployee) {}

    public record TeamRequest(@NotBlank @Size(max = 50) String name) {}

    public record TeamUpdateRequest(Boolean active) {}

    public record TeamResponse(Long id, String name, boolean active, String source) {}

    public record SyncResponse(int added, int total) {}

    public record AuditLogResponse(Long id, Long adminId, Long targetUserId, String action, String changes,
                                   LocalDateTime createdAt) {
        public static AuditLogResponse of(AdminAuditLog a) {
            return new AuditLogResponse(a.getId(), a.getAdminId(), a.getTargetUserId(), a.getAction(),
                    a.getChanges(), a.getCreatedAt());
        }
    }
}
