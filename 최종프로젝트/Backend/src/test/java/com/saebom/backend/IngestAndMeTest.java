package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.entity.*;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
import com.saebom.backend.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 적재 중계(관리자만, 올린 사람 = 로그인 계정) + 비밀번호 변경 */
@SpringBootTest
@AutoConfigureMockMvc
class IngestAndMeTest {

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;
    @Autowired UserRepository users;
    @Autowired TeamRepository teams;
    @Autowired PositionRepository positions;
    @Autowired PasswordEncoder enc;
    @Autowired com.saebom.backend.repository.AdminAuditLogRepository auditLog;
    @MockitoBean AiClient ai;

    private String token(String email, String pw) throws Exception {
        String body = send(post("/api/auth/login"), Map.of("email", email, "password", pw), null)
                .andReturn().getResponse().getContentAsString();
        return om.readTree(body).get("token").asText();
    }

    private ResultActions send(org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder b,
                               Object body, String token) throws Exception {
        b.contentType(MediaType.APPLICATION_JSON).content(om.writeValueAsString(body));
        return mvc.perform(token == null ? b : b.header("Authorization", "Bearer " + token));
    }

    private void staff(String email) {
        if (users.findByEmail(email).isPresent()) return;
        Team t = teams.findByName("적재팀").orElseGet(() -> teams.save(Team.builder().name("적재팀").build()));
        users.save(User.builder().email(email).pw(enc.encode("password123")).name("직원").team(t)
                .position(positions.findByName("대리").orElseThrow()).empId("E7" + Math.abs(email.hashCode() % 10000)).status(UserStatus.ACTIVE).build());
    }

    @Test
    void ingestIsAdminOnlyAndUsesAccountEmail() throws Exception {
        staff("worker@ingest.test");
        String worker = token("worker@ingest.test", "password123");
        send(post("/api/ingest/jobs"), Map.of("source", "text", "content", "글"), worker).andExpect(status().isForbidden());

        String admin = token("admin@saebom.test", "admin-password-123");
        when(ai.postJson(eq("/ingest/jobs"), any())).thenReturn(om.readTree("{\"job_id\":1,\"total\":1}"));
        send(post("/api/ingest/jobs"), Map.of("source", "text", "content", "본문", "uploader", "someone@else"), admin)
                .andExpect(status().isAccepted()).andExpect(jsonPath("$.job_id").value(1));
        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> cap = ArgumentCaptor.forClass(Map.class);
        verify(ai).postJson(eq("/ingest/jobs"), cap.capture());
        assertThat(cap.getValue()).containsEntry("uploader", "admin@saebom.test").containsEntry("source", "text");

        send(post("/api/ingest/jobs"), Map.of("source", "csv", "content", "x"), admin).andExpect(status().isBadRequest());

        when(ai.getJson(eq("/ingest/docs/5"), any())).thenReturn(om.readTree("{\"id\":5}"));
        mvc.perform(get("/api/ingest/docs/5").header("Authorization", "Bearer " + admin)).andExpect(status().isOk());
        verify(ai).getJson("/ingest/docs/5", Map.of("actor", "admin@saebom.test"));
    }

    @Test
    void deleteDocIsAdminOnlyAndLogged() throws Exception {
        staff("del@ingest.test");
        String worker = token("del@ingest.test", "password123");
        send(post("/api/admin/docs/H0001/delete"), Map.of("reason", "x"), worker).andExpect(status().isForbidden());
        String admin = token("admin@saebom.test", "admin-password-123");
        send(post("/api/admin/docs/H0001/delete"), Map.of("reason", " "), admin).andExpect(status().isBadRequest());
        send(post("/api/admin/docs/H0001.x/delete"), Map.of("reason", "x"), admin).andExpect(status().isBadRequest());
        when(ai.postJson(eq("/docs/H0001/delete"), any())).thenReturn(om.readTree("{\"doc_id\":\"H0001\",\"chunks\":2}"));
        send(post("/api/admin/docs/H0001/delete"), Map.of("reason", "잘못 적재"), admin)
                .andExpect(status().isOk()).andExpect(jsonPath("$.chunks").value(2));
        verify(ai).postJson("/docs/H0001/delete", Map.of("actor", "admin@saebom.test", "reason", "잘못 적재"));
        assertThat(auditLog.findAll()).anyMatch(a -> a.getAction().equals("delete_doc") && a.getChanges().contains("H0001"));
    }

    @Test
    void updateDocMetaChecksTeamAndLogs() throws Exception {
        String admin = token("admin@saebom.test", "admin-password-123");
        teams.findByName("적재팀").orElseGet(() -> teams.save(Team.builder().name("적재팀").build()));
        send(post("/api/admin/docs/H0002/meta"), Map.of("visibility", "team", "team", "없는팀"), admin).andExpect(status().isBadRequest());
        send(post("/api/admin/docs/H0002/meta"), Map.of("visibility", "secret"), admin).andExpect(status().isBadRequest());
        when(ai.postJson(eq("/docs/H0002/meta"), any())).thenReturn(om.readTree(
                "{\"doc_id\":\"H0002\",\"before\":{\"visibility\":\"restricted\",\"team\":null,\"owner_emp_id\":null},"
                + "\"after\":{\"visibility\":\"team\",\"team\":\"적재팀\",\"owner_emp_id\":null}}"));
        send(post("/api/admin/docs/H0002/meta"), Map.of("visibility", "team", "team", "적재팀"), admin).andExpect(status().isOk());
        assertThat(auditLog.findAll()).anyMatch(a -> a.getAction().equals("update_doc")
                && a.getChanges().contains("\"visibility\":[\"restricted\",\"team\"]"));
    }

    @Test
    void changePassword() throws Exception {
        staff("pw@ingest.test");
        String t = token("pw@ingest.test", "password123");
        send(put("/api/me/password"), Map.of("currentPassword", "wrong-one", "newPassword", "newpassword1"), t)
                .andExpect(status().isBadRequest());
        send(put("/api/me/password"), Map.of("currentPassword", "password123", "newPassword", "short"), t)
                .andExpect(status().isBadRequest());
        send(put("/api/me/password"), Map.of("currentPassword", "password123", "newPassword", "newpassword1"), t)
                .andExpect(status().isOk());
        send(post("/api/auth/login"), Map.of("email", "pw@ingest.test", "password", "password123"), null)
                .andExpect(status().isUnauthorized());
        token("pw@ingest.test", "newpassword1");
    }
}
