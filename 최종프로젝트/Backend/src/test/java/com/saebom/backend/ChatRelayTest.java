package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AccessUser;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.AskResult;
import com.saebom.backend.ai.AiClient.RestoreResult;
import com.saebom.backend.entity.*;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
import com.saebom.backend.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 질문·복원 중계: AI 로 넘기는 사용자 정보가 화면 값이 아니라 DB 계정 값인지, 미리 막는 것들 */
@SpringBootTest
@AutoConfigureMockMvc
class ChatRelayTest {

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;
    @Autowired UserRepository users;
    @Autowired TeamRepository teams;
    @Autowired PositionRepository positions;
    @Autowired PasswordEncoder enc;
    @MockitoBean AiClient ai;

    private String account(String email, Role role, String empId, UserStatus st) throws Exception {
        if (users.findByEmail(email).isEmpty()) {
            Team t = teams.findByName("채팅팀").orElseGet(() -> teams.save(Team.builder().name("채팅팀").build()));
            users.save(User.builder().email(email).pw(enc.encode("password123")).name("사람").team(t)
                    .position(positions.findByName("대리").orElseThrow()).role(role).empId(empId)
                    .status(UserStatus.ACTIVE).build());
        }
        String body = post_("/api/auth/login", Map.of("email", email, "password", "password123"), null)
                .andReturn().getResponse().getContentAsString();
        String token = om.readTree(body).get("token").asText();
        var u = users.findByEmail(email).orElseThrow();
        u.setStatus(st);
        users.save(u);
        return token;
    }

    private ResultActions post_(String url, Object body, String token) throws Exception {
        var r = post(url).contentType(MediaType.APPLICATION_JSON).content(om.writeValueAsString(body));
        return mvc.perform(token == null ? r : r.header("Authorization", "Bearer " + token));
    }

    @Test
    void askSendsDbAccountNotClientValues() throws Exception {
        String token = account("staff@chat.test", Role.STAFF, "E001", UserStatus.ACTIVE);
        when(ai.ask(any(), anyString(), anyInt())).thenReturn(new AskResult("q", "<PERSON_ab> 고객 상담 [H1]", false,
                List.of(new AiClient.SourceDoc("H1", "상담", 0.9, "…")), null, "m", Map.of("search", 0.1)));
        // 화면이 role·emp_id 를 끼워 보내도 무시됨
        post_("/api/chat/ask", Map.of("query", "  대출 상담  ", "role", "AUDITOR", "emp_id", "E999"), token)
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.answer").value("<PERSON_ab> 고객 상담 [H1]"))
                .andExpect(jsonPath("$.docs[0].docId").value("H1"));
        verify(ai).ask(eq(new AccessUser("E001", "채팅팀", "STAFF")), eq("대출 상담"), eq(3));
    }

    @Test
    void blocksUnlinkedAndInactive() throws Exception {
        String unlinked = account("nolink@chat.test", Role.STAFF, null, UserStatus.ACTIVE);
        post_("/api/chat/ask", Map.of("query", "x"), unlinked).andExpect(status().isForbidden());
        String suspended = account("susp@chat.test", Role.LEAD, "E002", UserStatus.SUSPENDED);
        post_("/api/chat/ask", Map.of("query", "x"), suspended).andExpect(status().isUnauthorized());
        post_("/api/chat/ask", Map.of("query", "x"), null).andExpect(status().isUnauthorized());
        post_("/api/chat/ask", Map.of("query", ""), unlinked).andExpect(status().isBadRequest());
        verify(ai, never()).ask(any(), anyString(), anyInt());
    }

    @Test
    void restore() throws Exception {
        String lead = account("lead@chat.test", Role.LEAD, "E003", UserStatus.ACTIVE);
        when(ai.restore(any(), anyString(), anyString(), anyList(), any()))
                .thenReturn(new RestoreResult("홍길동 고객", 1, List.of(), List.of(), List.of("H1")));
        post_("/api/chat/restore", Map.of("text", "<PERSON_ab> 고객", "docIds", List.of("H1")), lead)
                .andExpect(status().isOk()).andExpect(jsonPath("$.restored").value(1));
        verify(ai).restore(eq(new AccessUser("E003", "채팅팀", "LEAD")), eq("lead@chat.test"), eq("<PERSON_ab> 고객"),
                eq(List.of("H1")), isNull());

        // 감사: 사유 없으면 AI 부르기 전에 400, 연결 없어도 됨
        String aud = account("aud@chat.test", Role.AUDITOR, null, UserStatus.ACTIVE);
        post_("/api/chat/restore", Map.of("text", "t", "docIds", List.of("H1"), "purpose", " "), aud)
                .andExpect(status().isBadRequest());
        post_("/api/chat/restore", Map.of("text", "t", "docIds", List.of("H1"), "purpose", "민원 감사"), aud)
                .andExpect(status().isOk());

        // AI 가 권한으로 거절하면 403 + 그 메시지
        when(ai.ask(any(), anyString(), anyInt())).thenThrow(ApiException.forbidden("역할을 모르는 사용자"));
        post_("/api/chat/ask", Map.of("query", "x"), lead)
                .andExpect(status().isForbidden()).andExpect(jsonPath("$.message").value("역할을 모르는 사용자"));
    }
}
