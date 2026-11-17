package com.saebom.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.entity.Team;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
import jakarta.mail.Session;
import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mail.MailSendException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.Map;
import java.util.Properties;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 가입 이메일 인증 (worklog 41장). 메일은 가짜 발송기로 받아 본문에서 코드를 꺼냄 */
@SpringBootTest(properties = "app.mail.verify-signup=true")
@AutoConfigureMockMvc
class EmailVerificationTest {

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper om;
    @Autowired TeamRepository teams;
    @Autowired PositionRepository positions;
    @Autowired com.saebom.backend.repository.UserRepository users;
    @Autowired org.springframework.security.crypto.password.PasswordEncoder encoder;
    @MockitoBean AiClient ai;
    @MockitoBean JavaMailSender mail;

    Long teamId, posId;

    @BeforeEach
    void setUp() {
        teamId = teams.findByName("고객센터").orElseGet(() -> teams.save(Team.builder().name("고객센터").build())).getId();
        posId = positions.findByName("대리").orElseThrow().getId();
        reset(mail);
        when(mail.createMimeMessage()).thenAnswer(i -> new MimeMessage(Session.getInstance(new Properties())));
    }

    private ResultActions post_(String url, Object body) throws Exception {
        return mvc.perform(post(url).contentType(MediaType.APPLICATION_JSON).content(om.writeValueAsString(body)));
    }

    private Map<String, Object> signupBody(String email) {
        return Map.of("email", email, "password", "password123", "name", "인증자", "teamId", teamId, "positionId", posId);
    }

    /** 마지막으로 보낸 메일 제목에서 6자리 코드 */
    private String lastCode() throws Exception {
        var cap = ArgumentCaptor.forClass(MimeMessage.class);
        verify(mail, atLeastOnce()).send(cap.capture());
        var m = Pattern.compile("(\\d{6})").matcher(cap.getValue().getSubject());
        assertThat(m.find()).isTrue();
        return m.group(1);
    }

    @Test
    void optionsSaysVerificationOn() throws Exception {
        mvc.perform(get("/api/auth/options")).andExpect(status().isOk()).andExpect(jsonPath("$.emailVerification").value(true));
    }

    @Test
    void signupNeedsVerifiedEmail() throws Exception {
        String email = "verify1@saebom.test";
        post_("/api/auth/signup", signupBody(email)).andExpect(status().isBadRequest());

        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isOk());
        String code = lastCode();
        assertThat(code).hasSize(6);

        // 60초 안에 다시 요청하면 429
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isTooManyRequests());

        String wrong = code.equals("000000") ? "111111" : "000000";
        post_("/api/auth/email/verify", Map.of("email", email, "code", wrong))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("남은 횟수 4번")));
        post_("/api/auth/email/verify", Map.of("email", email, "code", code)).andExpect(status().isOk());

        // 대문자로 써도 같은 이메일
        post_("/api/auth/signup", signupBody("Verify1@Saebom.test")).andExpect(status().isCreated());
    }

    @Test
    void codeLockedAfterFiveWrong() throws Exception {
        String email = "verify2@saebom.test";
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isOk());
        String code = lastCode();
        String wrong = code.equals("000000") ? "111111" : "000000";
        for (int i = 0; i < 5; i++) post_("/api/auth/email/verify", Map.of("email", email, "code", wrong)).andExpect(status().isBadRequest());
        post_("/api/auth/email/verify", Map.of("email", email, "code", code))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("다시 받아")));
        post_("/api/auth/signup", signupBody(email)).andExpect(status().isBadRequest());
    }

    @Test
    void senderNameIsKorean() throws Exception {
        post_("/api/auth/email/send", Map.of("email", "verify5@saebom.test")).andExpect(status().isOk());
        var cap = ArgumentCaptor.forClass(MimeMessage.class);
        verify(mail).send(cap.capture());
        var from = (jakarta.mail.internet.InternetAddress) cap.getValue().getFrom()[0];
        assertThat(from.getPersonal()).isEqualTo("사공삼(403)");
        // .properties·.env 에서 ISO-8859-1 로 깨져 들어온 한글도 되살림
        String broken = new String("새봄팀".getBytes(java.nio.charset.StandardCharsets.UTF_8), java.nio.charset.StandardCharsets.ISO_8859_1);
        assertThat(com.saebom.backend.service.EmailVerificationService.senderName(broken)).isEqualTo("새봄팀");
        assertThat(com.saebom.backend.service.EmailVerificationService.senderName("Saebom")).isEqualTo("Saebom");
        assertThat(com.saebom.backend.service.EmailVerificationService.senderName("Café")).isEqualTo("Café");
    }

    @Test
    void htmlBodyHasSelectableCode() throws Exception {
        String html = com.saebom.backend.service.EmailVerificationService.codeHtml("가입 인증 코드", "가입 화면", "206313", 10);
        assertThat(html).contains(">206313</span>").contains("user-select:all").contains("10분").doesNotContain("<script");
        // 눈으로 볼 미리보기 (빌드 폴더라 깃에 안 올라감)
        java.nio.file.Files.writeString(java.nio.file.Path.of("target", "email-preview.html"), html);
    }

    @Test
    void passwordReset() throws Exception {
        // 가입 → 인증 없이 바로 쓰려고 test 설정 그대로 (가입 인증만 켜진 상태라 먼저 가입 인증)
        String email = "reset1@saebom.test";
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isOk());
        post_("/api/auth/email/verify", Map.of("email", email, "code", lastCode())).andExpect(status().isOk());
        post_("/api/auth/signup", signupBody(email)).andExpect(status().isCreated());
        reset(mail);
        when(mail.createMimeMessage()).thenAnswer(i -> new MimeMessage(Session.getInstance(new Properties())));

        // 인증 없이 재설정은 거절
        post_("/api/auth/password/reset", Map.of("email", email, "password", "newpassword1")).andExpect(status().isBadRequest());

        post_("/api/auth/password/send", Map.of("email", email)).andExpect(status().isOk());
        var cap = ArgumentCaptor.forClass(MimeMessage.class);
        verify(mail).send(cap.capture());
        assertThat(cap.getValue().getSubject()).contains("비밀번호 재설정");
        String code = lastCode();

        // 재설정 코드는 가입 인증에 못 씀 (목적이 hash 에 들어감)
        post_("/api/auth/email/verify", Map.of("email", email, "code", code)).andExpect(status().isBadRequest());

        post_("/api/auth/password/verify", Map.of("email", email, "code", code)).andExpect(status().isOk());
        post_("/api/auth/password/reset", Map.of("email", email, "password", "short")).andExpect(status().isBadRequest());
        post_("/api/auth/password/reset", Map.of("email", email, "password", "newpassword1")).andExpect(status().isOk());
        // 한 번만
        post_("/api/auth/password/reset", Map.of("email", email, "password", "newpassword2")).andExpect(status().isBadRequest());

        var user = users.findByEmail(email).orElseThrow();
        assertThat(encoder.matches("newpassword1", user.getPw())).isTrue();
    }

    @Test
    void passwordResetUnknownEmailLooksSame() throws Exception {
        post_("/api/auth/password/send", Map.of("email", "nobody@saebom.test"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.message").value("가입된 이메일이면 인증 코드를 보냈어요."));
        verify(mail, never()).send(any(MimeMessage.class));
        post_("/api/auth/password/verify", Map.of("email", "nobody@saebom.test", "code", "123456")).andExpect(status().isBadRequest());
    }

    @Test
    void existingEmailRejected() throws Exception {
        post_("/api/auth/email/send", Map.of("email", "admin@saebom.test")).andExpect(status().isConflict());
        verify(mail, never()).send(any(MimeMessage.class));
    }

    @Test
    void sendFailureDoesNotBlockRetry() throws Exception {
        String email = "verify3@saebom.test";
        doThrow(new MailSendException("smtp down")).when(mail).send(any(MimeMessage.class));
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isBadGateway());
        doNothing().when(mail).send(any(MimeMessage.class));
        // 실패한 발송은 기록이 되돌려져서 60초 제한에 걸리지 않음
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isOk());
    }

    @Test
    void verificationIsSingleUse() throws Exception {
        String email = "verify4@saebom.test";
        post_("/api/auth/email/send", Map.of("email", email)).andExpect(status().isOk());
        post_("/api/auth/email/verify", Map.of("email", email, "code", lastCode())).andExpect(status().isOk());
        post_("/api/auth/signup", signupBody(email)).andExpect(status().isCreated());
        // 같은 이메일 재가입은 어차피 409 지만, 인증도 다시 쓸 수 없음
        post_("/api/auth/email/verify", Map.of("email", email, "code", "123456")).andExpect(status().isBadRequest());
    }
}
