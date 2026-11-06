package com.saebom.backend.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.Employee;
import com.saebom.backend.dto.AdminDtos.*;
import com.saebom.backend.entity.*;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.AdminAuditLogRepository;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
import com.saebom.backend.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 관리자 기능 (worklog 11-4): 가입 승인 + 직원 연결, 사용자 정보 보기·바꾸기, 팀 목록, 변경 기록.
 * 모든 변경은 admin_audit_log 에 전후 값으로 남김.
 */
@Service
@RequiredArgsConstructor
public class AdminService {

    /** 서버가 스스로 한 일 (첫 관리자·팀 자동 동기화) 의 admin_id */
    public static final long SYSTEM = 0L;

    private final UserRepository users;
    private final TeamRepository teams;
    private final PositionRepository positions;
    private final AdminAuditLogRepository auditLog;
    private final AiClient ai;
    private final ObjectMapper om;

    // ------------------------------------------------------------------ 사용자

    @Transactional(readOnly = true)
    public List<AdminUserResponse> listUsers(UserStatus status) {
        var list = status == null ? users.findAllByOrderByCreatedAtDesc() : users.findByStatusOrderByCreatedAtAsc(status);
        return list.stream().map(AdminUserResponse::of).toList();
    }

    @Transactional(readOnly = true)
    public AdminUserResponse getUser(Long id) {
        return AdminUserResponse.of(user(id));
    }

    /** 가입 정보(이름·팀)로 AI 서버 직원 후보 찾기. 이미 다른 계정에 연결된 후보는 linkedTo 로 표시 */
    @Transactional(readOnly = true)
    public CandidatesResponse candidates(Long id) {
        User u = user(id);
        String team = u.getTeam() == null ? null : u.getTeam().getName();
        List<Candidate> out = ai.match(u.getName(), team).stream()
                .map(e -> new Candidate(e, users.findByEmpId(e.empId()).map(User::getId)
                        .filter(other -> !other.equals(u.getId())).orElse(null)))
                .toList();
        return new CandidatesResponse(out, u.getPosition() == null ? null : u.getPosition().getDefaultRole());
    }

    public List<Employee> employees(String q) {
        return ai.employees(q);
    }

    @Transactional
    public AdminUserResponse approve(Long adminId, Long id, ApproveRequest req) {
        User u = user(id);
        if (u.getStatus() != UserStatus.PENDING) throw ApiException.conflict("승인 대기 중인 계정이 아닙니다.");
        boolean link = req.empId() != null && !req.empId().isBlank();
        if (link == req.createEmployee()) {
            throw ApiException.badRequest("기존 직원을 고르거나(empId) 새 직원 번호 만들기(createEmployee) 중 하나만 고르세요.");
        }
        Map<String, Object[]> changes = new LinkedHashMap<>();
        applyTeamPosition(u, req.teamId(), req.positionId(), changes);

        Role role;
        if (link) {
            Employee e = linkableEmployee(req.empId().trim(), u.getId());
            role = req.role() != null ? req.role() : roleOf(e.role(), u);
            set(changes, "empId", u.getEmpId(), e.empId());
            u.setEmpId(e.empId());
        } else {
            role = req.role() != null ? req.role() : defaultRole(u);
            // 안 A: 문서에 없는 사람 → AI 서버에 새 직원 번호. 실패하면 이 승인도 롤백 (AI 쪽엔 빈 직원이 남을 수 있음)
            Employee e = ai.createEmployee(u.getName(), u.getTeam() == null ? null : u.getTeam().getName(),
                    u.getPosition() == null ? null : u.getPosition().getName(), role.name());
            set(changes, "empId", u.getEmpId(), e.empId());
            changes.put("createdEmployee", new Object[]{null, e.empId()});
            u.setEmpId(e.empId());
        }
        set(changes, "role", u.getRole(), role);
        set(changes, "status", u.getStatus(), UserStatus.ACTIVE);
        u.setRole(role);
        u.setStatus(UserStatus.ACTIVE);
        u.setApprovedBy(adminId);
        u.setApprovedAt(LocalDateTime.now());
        log(adminId, u.getId(), "approve", changes);
        return AdminUserResponse.of(u);
    }

    @Transactional
    public AdminUserResponse update(Long adminId, Long id, UpdateUserRequest req) {
        User u = user(id);
        Map<String, Object[]> changes = new LinkedHashMap<>();
        applyTeamPosition(u, req.teamId(), req.positionId(), changes);
        if (req.role() != null) {
            set(changes, "role", u.getRole(), req.role());
            u.setRole(req.role());
        }
        if (req.admin() != null && req.admin() != u.isAdmin()) {
            if (!req.admin() && u.getId().equals(adminId)) throw ApiException.badRequest("자기 관리자 권한은 뺄 수 없습니다.");
            set(changes, "admin", u.isAdmin(), req.admin());
            u.setAdmin(req.admin());
        }
        if (req.status() != null && req.status() != u.getStatus()) {
            if (req.status() == UserStatus.PENDING) throw ApiException.badRequest("대기 상태로는 되돌릴 수 없습니다.");
            if (u.getStatus() == UserStatus.PENDING) throw ApiException.badRequest("대기 계정은 승인 화면에서 승인하세요.");
            if (req.status() == UserStatus.SUSPENDED && u.getId().equals(adminId)) {
                throw ApiException.badRequest("자기 계정은 정지할 수 없습니다.");
            }
            set(changes, "status", u.getStatus(), req.status());
            u.setStatus(req.status());
        }
        if (Boolean.TRUE.equals(req.unlinkEmployee())) {
            if (req.empId() != null) throw ApiException.badRequest("empId 와 unlinkEmployee 를 같이 보낼 수 없습니다.");
            set(changes, "empId", u.getEmpId(), null);
            u.setEmpId(null);
        } else if (req.empId() != null && !req.empId().isBlank() && !req.empId().trim().equals(u.getEmpId())) {
            Employee e = linkableEmployee(req.empId().trim(), u.getId());
            set(changes, "empId", u.getEmpId(), e.empId());
            u.setEmpId(e.empId());
        }
        if (!changes.isEmpty()) log(adminId, u.getId(), "update", changes);
        return AdminUserResponse.of(u);
    }

    // ------------------------------------------------------------------ 팀

    @Transactional(readOnly = true)
    public List<TeamResponse> listTeams() {
        return teams.findAllByOrderByNameAsc().stream().map(AdminService::team).toList();
    }

    @Transactional
    public TeamResponse createTeam(Long adminId, String name) {
        String n = name.replaceAll("\\s+", "");
        if (n.isEmpty()) throw ApiException.badRequest("팀 이름을 입력하세요.");
        if (teams.findByName(n).isPresent()) throw ApiException.conflict("이미 있는 팀입니다.");
        Team t = teams.save(Team.builder().name(n).source("admin").build());
        log(adminId, null, "create_team", Map.of("team", new Object[]{null, n}));
        return team(t);
    }

    @Transactional
    public TeamResponse updateTeam(Long adminId, Long id, TeamUpdateRequest req) {
        Team t = teams.findById(id).orElseThrow(() -> ApiException.notFound("팀 없음"));
        if (req.active() != null && req.active() != t.isActive()) {
            log(adminId, null, "update_team", Map.of("team", new Object[]{t.getName(), t.getName()},
                    "active", new Object[]{t.isActive(), req.active()}));
            t.setActive(req.active());
        }
        return team(t);
    }

    /** AI 서버가 문서에서 추론한 팀 이름을 목록에 더함 (있는 건 그대로, 지우지 않음) */
    @Transactional
    public SyncResponse syncTeams(Long adminId) {
        List<String> added = new ArrayList<>();
        for (var t : ai.teams()) {
            String n = t.name() == null ? "" : t.name().replaceAll("\\s+", "");
            if (n.isEmpty() || n.length() > 50 || teams.findByName(n).isPresent()) continue;
            teams.save(Team.builder().name(n).source("ai").build());
            added.add(n);
        }
        if (!added.isEmpty()) log(adminId, null, "sync_teams", Map.of("added", new Object[]{null, added}));
        return new SyncResponse(added.size(), (int) teams.count());
    }

    // ------------------------------------------------------------------ 기록

    @Transactional(readOnly = true)
    public List<AuditLogResponse> auditLog(Long targetUserId, int limit) {
        var list = targetUserId == null
                ? auditLog.findAllByOrderByIdDesc(PageRequest.of(0, Math.max(1, Math.min(limit, 500))))
                : auditLog.findByTargetUserIdOrderByIdDesc(targetUserId);
        return list.stream().map(AuditLogResponse::of).toList();
    }

    public void log(Long adminId, Long targetUserId, String action, Map<String, Object[]> changes) {
        Map<String, List<Object>> asJson = new LinkedHashMap<>();
        changes.forEach((k, v) -> asJson.put(k, Arrays.asList(v)));
        String json;
        try {
            json = om.writeValueAsString(asJson);
        } catch (JsonProcessingException e) {
            json = "{}";
        }
        auditLog.save(AdminAuditLog.builder().adminId(adminId).targetUserId(targetUserId)
                .action(action).changes(json).build());
    }

    // ------------------------------------------------------------------

    private User user(Long id) {
        return users.findById(id).orElseThrow(() -> ApiException.notFound("사용자 없음"));
    }

    private static TeamResponse team(Team t) {
        return new TeamResponse(t.getId(), t.getName(), t.isActive(), t.getSource());
    }

    private void applyTeamPosition(User u, Long teamId, Long positionId, Map<String, Object[]> changes) {
        if (teamId != null && (u.getTeam() == null || !teamId.equals(u.getTeam().getId()))) {
            Team t = teams.findById(teamId).orElseThrow(() -> ApiException.badRequest("팀 없음"));
            set(changes, "team", u.getTeam() == null ? null : u.getTeam().getName(), t.getName());
            u.setTeam(t);
        }
        if (positionId != null && (u.getPosition() == null || !positionId.equals(u.getPosition().getId()))) {
            Position p = positions.findById(positionId).orElseThrow(() -> ApiException.badRequest("직급 없음"));
            set(changes, "position", u.getPosition() == null ? null : u.getPosition().getName(), p.getName());
            u.setPosition(p);
        }
    }

    /** AI 서버에 있는 직원이고, 다른 계정에 연결돼 있지 않아야 함 */
    private Employee linkableEmployee(String empId, Long userId) {
        Employee e = ai.employee(empId).orElseThrow(() -> ApiException.badRequest("직원 목록에 없는 번호입니다: " + empId));
        users.findByEmpId(empId).filter(other -> !other.getId().equals(userId)).ifPresent(other -> {
            throw ApiException.conflict("이미 다른 계정(id " + other.getId() + ")에 연결된 직원입니다.");
        });
        return e;
    }

    private Role roleOf(String aiRole, User u) {
        try {
            return Role.valueOf(aiRole);
        } catch (IllegalArgumentException | NullPointerException e) {
            return defaultRole(u);
        }
    }

    private static Role defaultRole(User u) {
        return u.getPosition() == null ? Role.STAFF : u.getPosition().getDefaultRole();
    }

    private static void set(Map<String, Object[]> changes, String key, Object before, Object after) {
        if (!Objects.equals(before, after)) changes.put(key, new Object[]{before, after});
    }
}
