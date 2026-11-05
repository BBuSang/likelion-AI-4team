package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 회원가입 → 로그인 → /me 흐름 확인 (H2 메모리 DB) */
@SpringBootTest
@AutoConfigureMockMvc
class AuthFlowTest {

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;

    private String json(Object o) throws Exception {
        return om.writeValueAsString(o);
    }

    private void signup(String email, String pw, String name) throws Exception {
        mvc.perform(post("/api/auth/signup").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", email, "password", pw, "name", name))))
                .andExpect(status().isCreated());
    }

    @Test
    void signupLoginMe() throws Exception {
        signup("Kim@Saebom.test", "password123", "김사원");

        // 같은 이메일(대소문자만 다름) 다시 가입 → 409
        mvc.perform(post("/api/auth/signup").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "kim@saebom.test", "password", "password123", "name", "김사원"))))
                .andExpect(status().isConflict());

        mvc.perform(post("/api/auth/check-email").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "kim@saebom.test"))))
                .andExpect(status().isConflict());

        // 틀린 비밀번호 → 401
        mvc.perform(post("/api/auth/login").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "kim@saebom.test", "password", "wrong-pass"))))
                .andExpect(status().isUnauthorized());

        // 로그인 → 토큰
        String body = mvc.perform(post("/api/auth/login").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "kim@saebom.test", "password", "password123"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.role").value("USER"))
                .andReturn().getResponse().getContentAsString();
        String token = om.readTree(body).get("token").asText();

        // /me: 토큰 있으면 정보, 비밀번호 해시는 없음
        mvc.perform(get("/api/auth/me").header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.email").value("kim@saebom.test"))
                .andExpect(jsonPath("$.pw").doesNotExist());

        // 토큰 갱신
        mvc.perform(post("/api/auth/refresh").header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token").isString());
    }

    @Test
    void rejectsWithoutToken() throws Exception {
        mvc.perform(get("/api/auth/me")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/auth/me").header("Authorization", "Bearer not-a-token"))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/api/auth/refresh")).andExpect(status().isUnauthorized());
    }

    @Test
    void rejectsBadSignup() throws Exception {
        // 짧은 비밀번호, 이메일 형식 아님 → 400
        mvc.perform(post("/api/auth/signup").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "a@b.test", "password", "short", "name", "이름"))))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/api/auth/signup").contentType(MediaType.APPLICATION_JSON)
                        .content(json(java.util.Map.of("email", "not-email", "password", "password123", "name", "이름"))))
                .andExpect(status().isBadRequest());
    }
}
