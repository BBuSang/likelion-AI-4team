package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.Employee;
import com.saebom.backend.entity.Team;
import com.saebom.backend.repository.AdminAuditLogRepository;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 가입(PENDING) → 로그인 거부 → 관리자 승인(직원 연결) → 로그인 → 역할 변경·정지가 기존 토큰에 바로 반영.
 * H2 메모리 DB, AI 서버는 가짜.
 */
@SpringBootTest
@AutoConfigureMockMvc
class AuthFlowTest {

    static final String ADMIN = "admin@saebom.test";
    static final String ADMIN_PW = "admin-password-123";

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;
    @Autowired TeamRepository teams;
    @Autowired PositionRepository positions;
    @Autowired AdminAuditLogRepository auditLog;
    @MockitoBean AiClient ai;

    Long teamId, staffPos, leadPos;

    @BeforeEach
    void setUp() {
        teamId = teams.findByName("고객센터").orElseGet(() -> teams.save(Team.builder().name("고객센터").build())).getId();
        teams.findByName("인사팀").orElseGet(() -> teams.save(Team.builder().name("인사팀").build()));
        staffPos = positions.findByName("대리").orElseThrow().getId();
        leadPos = positions.findByName("팀장").orElseThrow().getId();
    }

    // ------------------------------------------------------------------ 도우미

    private String json(Object o) throws Exception {
        return om.writeValueAsString(o);
    }

    private ResultActions post_(String url, Object body, String token) throws Exception {
        var r = post(url).contentType(MediaType.APPLICATION_JSON).content(json(body));
        return mvc.perform(token == null ? r : r.header("Authorization", "Bearer " + token));
    }

    private ResultActions put_(String url, Object body, String token) throws Exception {
        return mvc.perform(put(url).contentType(MediaType.APPLICATION_JSON).content(json(body))
                .header("Authorization", "Bearer " + token));
    }

    private ResultActions get_(String url, String token) throws Exception {
        return mvc.perform(get(url).header("Authorization", "Bearer " + token));
    }

    private void signup(String email, String name, Long team, Long pos) throws Exception {
        post_("/api/auth/signup", Map.of("email", email, "password", "password123", "name", name,
                "teamId", team, "positionId", pos), null).andExpect(status().isCreated());
    }

    private String login(String email, String pw) throws Exception {
        String body = post_("/api/auth/login", Map.of("email", email, "password", pw), null)
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        return om.readTree(body).get("token").asText();
    }

    private long userId(String adminToken, String email) throws Exception {
        String body = get_("/api/admin/users", adminToken).andReturn().getResponse().getContentAsString();
        for (var n : om.readTree(body)) if (n.get("email").asText().equals(email)) return n.get("id").asLong();
        throw new AssertionError("no user " + email);
    }

    private static Employee emp(String id, String name, String team, String role, Boolean same) {
        return new Employee(id, name, team, "대리", role, 3, same);
    }

    // ------------------------------------------------------------------

    @Test
    void optionsArePublic() throws Exception {
        mvc.perform(get("/api/auth/options"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.teams[?(@.name == '고객센터')]").exists())
                .andExpect(jsonPath("$.positions[0].name").value("사원"))
                .andExpect(jsonPath("$.positions.length()").value(20));
    }

    @Test
    void signupPendingThenApproveWithEmployee() throws Exception {
        signup("Kim@Saebom.test", "김 민수", teamId, staffPos);
        // 같은 이메일(대소문자만 다름) → 409
        post_("/api/auth/signup", Map.of("email", "kim@saebom.test", "password", "password123", "name", "김민수",
                "teamId", teamId, "positionId", staffPos), null).andExpect(status().isConflict());

        // 승인 전: 비밀번호 틀리면 401(상태 안 알려 줌), 맞으면 403 대기
        post_("/api/auth/login", Map.of("email", "kim@saebom.test", "password", "wrong-pass"), null)
                .andExpect(status().isUnauthorized());
        post_("/api/auth/login", Map.of("email", "kim@saebom.test", "password", "password123"), null)
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.message").value("관리자 승인 대기 중입니다."));

        String admin = login(ADMIN, ADMIN_PW);
        long id = userId(admin, "kim@saebom.test");

        // 후보: 이름은 띄어쓰기 없이 AI 에 보냄
        when(ai.match("김민수", "고객센터")).thenReturn(List.of(
                emp("E010", "김민수", "고객센터", "LEAD", true), emp("E020", "김민수", "인사팀", "STAFF", false)));
        get_("/api/admin/users/" + id + "/candidates", admin)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.candidates[0].employee.empId").value("E010"))
                .andExpect(jsonPath("$.candidates[0].linkedTo").doesNotExist())
                .andExpect(jsonPath("$.positionDefaultRole").value("STAFF"));

        // empId 와 createEmployee 둘 다 없으면 400, 없는 직원 번호 400
        post_("/api/admin/users/" + id + "/approve", Map.of(), admin).andExpect(status().isBadRequest());
        when(ai.employee("E999")).thenReturn(Optional.empty());
        post_("/api/admin/users/" + id + "/approve", Map.of("empId", "E999"), admin).andExpect(status().isBadRequest());

        // 승인: 역할 기본값 = 연결 직원의 추론 역할 (LEAD)
        when(ai.employee("E010")).thenReturn(Optional.of(emp("E010", "김민수", "고객센터", "LEAD", null)));
        post_("/api/admin/users/" + id + "/approve", Map.of("empId", "E010"), admin)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ACTIVE"))
                .andExpect(jsonPath("$.empId").value("E010"))
                .andExpect(jsonPath("$.role").value("LEAD"));
        // 두 번 승인 → 409
        post_("/api/admin/users/" + id + "/approve", Map.of("empId", "E010"), admin).andExpect(status().isConflict());

        String token = login("kim@saebom.test", "password123");
        get_("/api/auth/me", token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("김민수"))
                .andExpect(jsonPath("$.team").value("고객센터"))
                .andExpect(jsonPath("$.role").value("LEAD"))
                .andExpect(jsonPath("$.pw").doesNotExist())
                .andExpect(jsonPath("$.empId").doesNotExist());
        // 일반 사용자는 관리자 API 403
        get_("/api/admin/users", token).andExpect(status().isForbidden());

        // 같은 직원을 다른 계정에 연결 → 409
        signup("lee@saebom.test", "김민수", teamId, staffPos);
        long id2 = userId(admin, "lee@saebom.test");
        post_("/api/admin/users/" + id2 + "/approve", Map.of("empId", "E010"), admin).andExpect(status().isConflict());
        when(ai.match("김민수", "고객센터")).thenReturn(List.of(emp("E010", "김민수", "고객센터", "LEAD", true)));
        get_("/api/admin/users/" + id2 + "/candidates", admin)
                .andExpect(jsonPath("$.candidates[0].linkedTo").value((int) id));

        // 관리자가 역할을 바꾸면 이미 받은 토큰에도 바로 반영 (요청마다 DB)
        put_("/api/admin/users/" + id, Map.of("role", "STAFF"), admin).andExpect(status().isOk());
        get_("/api/auth/me", token).andExpect(jsonPath("$.role").value("STAFF"));

        // 정지 → 기존 토큰 401, 로그인 403
        put_("/api/admin/users/" + id, Map.of("status", "SUSPENDED"), admin).andExpect(status().isOk());
        get_("/api/auth/me", token).andExpect(status().isUnauthorized());
        post_("/api/auth/refresh", Map.of(), token).andExpect(status().isUnauthorized());
        post_("/api/auth/login", Map.of("email", "kim@saebom.test", "password", "password123"), null)
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.message").value("사용이 정지된 계정입니다."));

        // 변경 기록: 승인·역할·정지
        String log = get_("/api/admin/audit-log?userId=" + id, admin).andReturn().getResponse().getContentAsString();
        var actions = om.readTree(log).findValuesAsText("action");
        assertThat(actions).containsExactly("update", "update", "approve");
        assertThat(om.readTree(log).get(2).get("changes").asText()).contains("\"empId\":[null,\"E010\"]");
    }

    @Test
    void approveCreatesNewEmployeeWhenNotInDocs() throws Exception {
        signup("park@saebom.test", "박신입", teamId, leadPos);
        String admin = login(ADMIN, ADMIN_PW);
        long id = userId(admin, "park@saebom.test");
        when(ai.createEmployee(eq("박신입"), eq("고객센터"), eq("팀장"), eq("LEAD")))
                .thenReturn(emp("U0001", "박신입", "고객센터", "LEAD", null));
        // 직급 기본 역할 (팀장 → LEAD)
        post_("/api/admin/users/" + id + "/approve", Map.of("createEmployee", true), admin)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.empId").value("U0001"))
                .andExpect(jsonPath("$.role").value("LEAD"));
        // 둘 다 보내면 400 (다른 계정으로)
        signup("choi@saebom.test", "최둘", teamId, staffPos);
        long id2 = userId(admin, "choi@saebom.test");
        post_("/api/admin/users/" + id2 + "/approve", Map.of("createEmployee", true, "empId", "E1"), admin)
                .andExpect(status().isBadRequest());
    }

    @Test
    void adminGuards() throws Exception {
        String admin = login(ADMIN, ADMIN_PW);
        long me = userId(admin, ADMIN);
        put_("/api/admin/users/" + me, Map.of("admin", false), admin).andExpect(status().isBadRequest());
        put_("/api/admin/users/" + me, Map.of("status", "SUSPENDED"), admin).andExpect(status().isBadRequest());

        signup("wait@saebom.test", "대기자", teamId, staffPos);
        long id = userId(admin, "wait@saebom.test");
        // 대기 계정을 update 로 ACTIVE 로 만들 수 없음 (직원 연결 없이 활성화 방지)
        put_("/api/admin/users/" + id, Map.of("status", "ACTIVE"), admin).andExpect(status().isBadRequest());
    }

    @Test
    void teamsCreateAndSync() throws Exception {
        String admin = login(ADMIN, ADMIN_PW);
        post_("/api/admin/teams", Map.of("name", "리스크 관리팀"), admin)
                .andExpect(status().isCreated()).andExpect(jsonPath("$.name").value("리스크관리팀"));
        post_("/api/admin/teams", Map.of("name", "리스크관리팀"), admin).andExpect(status().isConflict());

        when(ai.teams()).thenReturn(List.of(new AiClient.TeamInfo("여신심사팀", 12), new AiClient.TeamInfo("고객센터", 30)));
        post_("/api/admin/teams/sync", Map.of(), admin)
                .andExpect(status().isOk()).andExpect(jsonPath("$.added").value(1));

        // 쓰지 않는 팀은 가입 선택칸에서 빠짐 + 그 팀으로 가입 불가
        long id = teams.findByName("여신심사팀").orElseThrow().getId();
        put_("/api/admin/teams/" + id, Map.of("active", false), admin).andExpect(status().isOk());
        var body = new HashMap<String, Object>(Map.of("email", "x@saebom.test", "password", "password123",
                "name", "엑스", "teamId", id, "positionId", staffPos));
        post_("/api/auth/signup", body, null).andExpect(status().isBadRequest());
        assertThat(auditLog.findAll()).anyMatch(a -> a.getAction().equals("sync_teams"));
    }

    @Test
    void rejectsWithoutTokenAndBadSignup() throws Exception {
        mvc.perform(get("/api/auth/me")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/auth/me").header("Authorization", "Bearer not-a-token")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/admin/users")).andExpect(status().isUnauthorized());
        post_("/api/auth/signup", Map.of("email", "a@b.test", "password", "short", "name", "이름",
                "teamId", teamId, "positionId", staffPos), null).andExpect(status().isBadRequest());
        post_("/api/auth/signup", Map.of("email", "a@b.test", "password", "password123", "name", "이름"), null)
                .andExpect(status().isBadRequest()); // 팀·직급 없음
        post_("/api/auth/signup", Map.of("email", "a@b.test", "password", "password123", "name", "이름",
                "teamId", 9999, "positionId", staffPos), null).andExpect(status().isBadRequest());
    }

    @Test
    void aiDownGivesBadGateway() throws Exception {
        String admin = login(ADMIN, ADMIN_PW);
        when(ai.employees(any())).thenThrow(com.saebom.backend.exception.ApiException.badGateway("AI 서버에 연결할 수 없습니다"));
        get_("/api/admin/employees?q=김", admin).andExpect(status().isBadGateway());
    }
}
