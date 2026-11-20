package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.ai.AiClient.AskResult;
import com.saebom.backend.entity.*;
import com.saebom.backend.repository.AdminAuditLogRepository;
import com.saebom.backend.repository.ChatMessageRepository;
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
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 채팅 내역 서버 저장 (worklog 61장): 서버가 가명처리된 질문·답만 저장, 본인만, 관리자 조회는 기록 */
@SpringBootTest
@AutoConfigureMockMvc
class ChatHistoryTest {

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;
    @Autowired UserRepository users;
    @Autowired TeamRepository teams;
    @Autowired PositionRepository positions;
    @Autowired PasswordEncoder enc;
    @Autowired ChatMessageRepository messages;
    @Autowired com.saebom.backend.repository.ChatConversationRepository convRepo;
    @Autowired AdminAuditLogRepository auditLog;
    @MockitoBean AiClient ai;

    private String login(String email, String empId) throws Exception {
        if (users.findByEmail(email).isEmpty()) {
            Team t = teams.findByName("내역팀").orElseGet(() -> teams.save(Team.builder().name("내역팀").build()));
            users.save(User.builder().email(email).pw(enc.encode("password123")).name("사람").team(t)
                    .position(positions.findByName("대리").orElseThrow()).role(Role.STAFF).empId(empId)
                    .status(UserStatus.ACTIVE).build());
        }
        return token(email, "password123");
    }

    private String token(String email, String pw) throws Exception {
        String body = send(post("/api/auth/login"), Map.of("email", email, "password", pw), null)
                .andReturn().getResponse().getContentAsString();
        return om.readTree(body).get("token").asText();
    }

    private ResultActions send(MockHttpServletRequestBuilder b, Object body, String token) throws Exception {
        if (body != null) b.contentType(MediaType.APPLICATION_JSON).content(om.writeValueAsString(body));
        return mvc.perform(token == null ? b : b.header("Authorization", "Bearer " + token));
    }

    private long newConv(String token) throws Exception {
        String body = send(post("/api/chat/conversations"), null, token).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        return om.readTree(body).get("id").asLong();
    }

    private static AskResult answer(String masked) {
        return new AskResult(masked, "<PERSON_ab12cd34> 고객 반송 처리 완료 [H1]", false,
                List.of(new AiClient.SourceDoc("H1", "반송", 0.9, "<PERSON_ab12cd34> 우편물 반송")), null, "m",
                Map.of("search", 0.1), null, List.of());
    }

    @Test
    void savesMaskedQueryAndAnswerNotRawQuestion() throws Exception {
        String me = login("h1@history.test", "E961");
        long conv = newConv(me);
        when(ai.ask(any(), anyString(), anyInt())).thenReturn(answer("<PERSON_ab12cd34> 고객 우편물 반송"));

        send(post("/api/chat/ask"), Map.of("query", "황시온 고객 우편물 반송", "conversationId", conv), me)
                .andExpect(status().isOk());

        // 목록: 본인에겐 친 질문 그대로(암호문을 풀어서), 1건
        send(get("/api/chat/conversations"), null, me)
                .andExpect(jsonPath("$[0].id").value(conv))
                .andExpect(jsonPath("$[0].title").value("황시온 고객 우편물 반송"))
                .andExpect(jsonPath("$[0].messages").value(1));
        // 내용: 원문 질문 + 가명처리된 질문 + 답변·근거 문서까지 /ask 응답과 같은 모양
        send(get("/api/chat/conversations/" + conv), null, me)
                .andExpect(jsonPath("$.messages[0].query").value("황시온 고객 우편물 반송"))
                .andExpect(jsonPath("$.messages[0].maskedQuery").value("<PERSON_ab12cd34> 고객 우편물 반송"))
                .andExpect(jsonPath("$.messages[0].result.answer").value("<PERSON_ab12cd34> 고객 반송 처리 완료 [H1]"))
                .andExpect(jsonPath("$.messages[0].result.docs[0].docId").value("H1"));
        // DB 에는 실명이 평문으로 없음 (원문은 암호문으로만)
        assertThat(messages.findAll()).allSatisfy(m -> {
            assertThat(m.getMaskedQuery()).doesNotContain("황시온");
            assertThat(m.getResult()).doesNotContain("황시온");
            assertThat(m.getQueryEnc()).isNotBlank().doesNotContain("황시온");
        });
        assertThat(convRepo.findAll()).allSatisfy(c -> {
            assertThat(c.getTitle()).doesNotContain("황시온");
            if (c.getTitleEnc() != null) assertThat(c.getTitleEnc()).doesNotContain("황시온");
        });
    }

    @Test
    void streamSavesWhenDone() throws Exception {
        String me = login("h2@history.test", "E962");
        long conv = newConv(me);
        doAnswer(inv -> {
            ((Runnable) inv.getArgument(3)).run();
            @SuppressWarnings("unchecked") Consumer<AskResult> onDone = inv.getArgument(5);
            onDone.accept(answer("<PERSON_ab12cd34> 반송"));
            return null;
        }).when(ai).askStream(any(), anyString(), anyInt(), any(), any(), any());

        send(post("/api/chat/ask/stream"), Map.of("query", "황시온 반송", "conversationId", conv), me).andExpect(status().isOk());
        send(get("/api/chat/conversations/" + conv), null, me)
                .andExpect(jsonPath("$.messages.length()").value(1))
                .andExpect(jsonPath("$.messages[0].query").value("황시온 반송"))
                .andExpect(jsonPath("$.title").value("황시온 반송"));  // 본인에겐 친 질문 그대로
    }

    @Test
    void onlyOwnerCanReadAskOrDelete() throws Exception {
        String owner = login("h3@history.test", "E963");
        String other = login("h4@history.test", "E964");
        long conv = newConv(owner);

        send(get("/api/chat/conversations/" + conv), null, other).andExpect(status().isNotFound());
        send(delete("/api/chat/conversations/" + conv), null, other).andExpect(status().isNotFound());
        // 남의 대화에 질문을 붙이려 하면 AI 를 부르기 전에 막음
        send(post("/api/chat/ask"), Map.of("query", "x", "conversationId", conv), other).andExpect(status().isNotFound());
        verify(ai, never()).ask(any(), anyString(), anyInt());
        send(get("/api/chat/conversations"), null, other).andExpect(jsonPath("$.length()").value(0));

        send(delete("/api/chat/conversations/" + conv), null, owner).andExpect(status().isOk());
        send(get("/api/chat/conversations/" + conv), null, owner).andExpect(status().isNotFound());
    }

    @Test
    void adminCanReadAndEachViewIsLogged() throws Exception {
        String me = login("h5@history.test", "E965");
        long conv = newConv(me);
        when(ai.ask(any(), anyString(), anyInt())).thenReturn(answer("<PERSON_ab12cd34> 반송"));
        send(post("/api/chat/ask"), Map.of("query", "황시온 반송", "conversationId", conv), me).andExpect(status().isOk());
        long userId = users.findByEmail("h5@history.test").orElseThrow().getId();

        // 일반 사용자는 관리자 경로 불가
        send(get("/api/admin/users/" + userId + "/conversations"), null, me).andExpect(status().isForbidden());

        String admin = token("admin@saebom.test", "admin-password-123");
        // 관리자에겐 원문 질문을 풀어 주지 않음: 제목·질문 모두 가명처리된 글
        send(get("/api/admin/users/" + userId + "/conversations"), null, admin)
                .andExpect(jsonPath("$[0].id").value(conv))
                .andExpect(jsonPath("$[0].title").value("<PERSON_ab12cd34> 반송"));
        long before = auditLog.findByTargetUserIdOrderByIdDesc(userId).size();
        send(get("/api/admin/users/" + userId + "/conversations/" + conv), null, admin)
                .andExpect(jsonPath("$.title").value("<PERSON_ab12cd34> 반송"))
                .andExpect(jsonPath("$.messages[0].query").doesNotExist())
                .andExpect(jsonPath("$.messages[0].maskedQuery").value("<PERSON_ab12cd34> 반송"))
                .andExpect(jsonPath("$.messages[0].result.docs[0].docId").value("H1"));
        var logs = auditLog.findByTargetUserIdOrderByIdDesc(userId);
        assertThat(logs).hasSize((int) before + 1);
        assertThat(logs.get(0).getAction()).isEqualTo("view_chat");
        assertThat(logs.get(0).getChanges()).contains(String.valueOf(conv));
    }
}
