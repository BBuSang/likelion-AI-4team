package com.saebom.backend.service;

import com.saebom.backend.entity.EmailVerification;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.EmailVerificationRepository;
import com.saebom.backend.repository.UserRepository;
import jakarta.mail.internet.InternetAddress;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.mail.MailException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HexFormat;

/**
 * 이메일 인증 코드 — 가입(SIGNUP)·비밀번호 재설정(RESET) 공통.
 * 코드 6자리, 10분 유효, 5번 틀리면 무효. 같은 이메일·목적은 60초에 한 번·하루 10번, 같은 IP 는 1시간 30번까지 발송.
 * 확인 후 30분 안에 가입·재설정을 마쳐야 함 (한 번 쓰면 끝).
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EmailVerificationService {

    public static final String SIGNUP = "SIGNUP";
    public static final String RESET = "RESET";
    static final Duration CODE_TTL = Duration.ofMinutes(10);
    static final Duration RESEND_GAP = Duration.ofSeconds(60);
    static final Duration VERIFIED_TTL = Duration.ofMinutes(30);
    static final int MAX_ATTEMPTS = 5;
    static final int MAX_PER_EMAIL_DAY = 10;
    static final int MAX_PER_IP_HOUR = 30;

    private final EmailVerificationRepository repo;
    private final UserRepository userRepository;
    private final JavaMailSender mailSender;
    private final SecureRandom random = new SecureRandom();

    @Value("${jwt.secret}")
    private String secret;
    @Value("${spring.mail.username:}")
    private String from;
    @Value("${app.mail.from-name:}")
    private String fromName;
    @Value("${app.mail.verify-signup:true}")
    private boolean verifySignup;

    public boolean signupVerificationRequired() {
        return verifySignup;
    }

    /** 메일 설정이 있어야 비밀번호 재설정을 화면에 보여 줌 */
    public boolean mailConfigured() {
        return from != null && !from.isBlank();
    }

    // ------------------------------------------------------------------ 가입

    /** 가입 코드 발송. 이미 가입된 이메일은 409 (가입 화면 check-email 과 같은 정보라 숨기지 않음) */
    @Transactional
    public void sendSignupCode(String rawEmail, String ip) {
        String email = UserService.normalizeEmail(rawEmail);
        requireMail();
        if (userRepository.existsByEmail(email)) throw ApiException.conflict("이미 존재하는 이메일입니다.");
        issue(email, SIGNUP, ip);
    }

    @Transactional(noRollbackFor = ApiException.class)
    public void verifySignupCode(String rawEmail, String code) {
        verify(UserService.normalizeEmail(rawEmail), SIGNUP, code);
    }

    /** 가입 신청 때 호출 (가입 저장과 같은 트랜잭션) */
    @Transactional
    public void consumeSignup(String rawEmail) {
        if (!verifySignup) return;
        consume(UserService.normalizeEmail(rawEmail), SIGNUP);
    }

    // ------------------------------------------------------------------ 비밀번호 재설정

    /**
     * 재설정 코드 발송. 계정이 없어도 같은 응답 (가입 여부를 알려 주지 않음) — 없으면 메일·기록 없이 끝.
     * 단 60초·하루 횟수 제한은 계정이 있을 때만 걸려서, 없는 주소는 항상 성공처럼 보임
     */
    @Transactional
    public void sendResetCode(String rawEmail, String ip) {
        String email = UserService.normalizeEmail(rawEmail);
        requireMail();
        if (!userRepository.existsByEmail(email)) return;
        issue(email, RESET, ip);
    }

    @Transactional(noRollbackFor = ApiException.class)
    public void verifyResetCode(String rawEmail, String code) {
        verify(UserService.normalizeEmail(rawEmail), RESET, code);
    }

    /** 새 비밀번호 저장 때 호출 (저장과 같은 트랜잭션) */
    @Transactional
    public void consumeReset(String rawEmail) {
        consume(UserService.normalizeEmail(rawEmail), RESET);
    }

    // ------------------------------------------------------------------ 공통

    private void requireMail() {
        if (!mailConfigured()) {
            throw new ApiException(HttpStatus.SERVICE_UNAVAILABLE, "메일 발송이 설정되지 않았어요. 관리자에게 문의해 주세요.");
        }
    }

    private void issue(String email, String purpose, String ip) {
        LocalDateTime now = LocalDateTime.now();
        repo.findFirstByEmailAndPurposeOrderByIdDesc(email, purpose).ifPresent(last -> {
            long wait = RESEND_GAP.getSeconds() - Duration.between(last.getCreatedAt(), now).getSeconds();
            if (wait > 0) throw tooMany("인증 코드는 " + wait + "초 뒤에 다시 받을 수 있어요.");
        });
        if (repo.countByEmailAndPurposeAndCreatedAtAfter(email, purpose, now.minusDays(1)) >= MAX_PER_EMAIL_DAY) {
            throw tooMany("오늘 이 이메일로 보낼 수 있는 인증 코드를 모두 썼어요. 내일 다시 시도해 주세요.");
        }
        if (ip != null && repo.countByRequestIpAndCreatedAtAfter(ip, now.minusHours(1)) >= MAX_PER_IP_HOUR) {
            throw tooMany("요청이 너무 많아요. 잠시 후 다시 시도해 주세요.");
        }
        String code = String.format("%06d", random.nextInt(1_000_000));
        repo.save(EmailVerification.builder()
                .email(email).purpose(purpose).codeHash(hash(email, purpose, code))
                .expiresAt(now.plus(CODE_TTL)).requestIp(ip).build());
        send(email, purpose, code); // 실패하면 예외 → 저장도 되돌림 (발송 못 한 코드로 60초 막히지 않게)
    }

    /** 코드 확인. 맞으면 verifiedAt 기록. 틀린 횟수는 예외가 나도 저장 (호출하는 쪽 noRollbackFor) */
    private void verify(String email, String purpose, String code) {
        EmailVerification v = repo.findFirstByEmailAndPurposeOrderByIdDesc(email, purpose)
                .orElseThrow(() -> ApiException.badRequest(RESET.equals(purpose)
                        ? "인증 코드가 맞지 않아요. 이메일을 확인하고 코드를 다시 받아 주세요."
                        : "먼저 인증 코드를 받아 주세요."));
        LocalDateTime now = LocalDateTime.now();
        if (v.getVerifiedAt() != null && v.getUsedAt() == null) return; // 이미 확인됨 (두 번 눌러도 성공)
        if (v.getUsedAt() != null) throw ApiException.badRequest("이미 사용한 인증이에요. 다시 받아 주세요.");
        if (v.getExpiresAt().isBefore(now)) throw ApiException.badRequest("인증 코드가 만료됐어요. 다시 받아 주세요.");
        if (v.getAttempts() >= MAX_ATTEMPTS) throw ApiException.badRequest("여러 번 틀려서 이 코드는 쓸 수 없어요. 다시 받아 주세요.");
        String c = code == null ? "" : code.replaceAll("\\s+", "");
        if (!MessageDigest.isEqual(hash(email, purpose, c).getBytes(StandardCharsets.US_ASCII),
                                   v.getCodeHash().getBytes(StandardCharsets.US_ASCII))) {
            v.setAttempts(v.getAttempts() + 1);
            int left = MAX_ATTEMPTS - v.getAttempts();
            throw ApiException.badRequest(left > 0 ? "인증 코드가 맞지 않아요. (남은 횟수 " + left + "번)"
                                                   : "여러 번 틀려서 이 코드는 쓸 수 없어요. 다시 받아 주세요.");
        }
        v.setVerifiedAt(now);
    }

    /** 30분 안에 확인된, 아직 안 쓴 인증이 있어야 함. 쓰면 used 처리 */
    private void consume(String email, String purpose) {
        EmailVerification v = repo.findFirstByEmailAndPurposeOrderByIdDesc(email, purpose)
                .filter(x -> x.getVerifiedAt() != null && x.getUsedAt() == null)
                .filter(x -> x.getVerifiedAt().plus(VERIFIED_TTL).isAfter(LocalDateTime.now()))
                .orElseThrow(() -> ApiException.badRequest("이메일 인증을 먼저 마쳐 주세요."));
        v.setUsedAt(LocalDateTime.now());
    }

    private void send(String to, String purpose, String code) {
        boolean reset = RESET.equals(purpose);
        String title = reset ? "비밀번호 재설정 코드" : "가입 인증 코드";
        String where = reset ? "비밀번호 재설정 화면" : "가입 화면";
        try {
            var msg = mailSender.createMimeMessage();
            var h = new MimeMessageHelper(msg, true, "UTF-8");
            h.setFrom(new InternetAddress(from, senderName(fromName), "UTF-8"));
            h.setTo(to);
            h.setSubject("[사공삼] " + title + " " + code);
            // 글자만 보이는 메일 앱용 + HTML
            h.setText("사공삼(403) " + title + ": " + code + "\n\n"
                    + CODE_TTL.toMinutes() + "분 안에 " + where + "에 입력해 주세요.\n"
                    + "본인이 요청하지 않았다면 이 메일은 무시하셔도 됩니다.",
                    codeHtml(title, where, code, CODE_TTL.toMinutes()));
            mailSender.send(msg);
        } catch (MailException | jakarta.mail.MessagingException | java.io.UnsupportedEncodingException e) {
            log.warn("인증 메일 발송 실패: {}", e.getClass().getSimpleName()); // 주소·코드는 남기지 않음
            throw ApiException.badGateway("인증 메일을 보내지 못했어요. 이메일 주소를 확인하거나 잠시 후 다시 시도해 주세요.");
        }
    }

    /** 목적도 넣어서, 가입 코드를 재설정에 (또는 반대로) 쓸 수 없게 */
    private String hash(String email, String purpose, String code) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            return HexFormat.of().formatHex(mac.doFinal((purpose + ":" + email + ":" + code).getBytes(StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * 인증 메일 HTML. 메일 앱은 스크립트를 막아서 복사 버튼 대신 코드 칸을 user-select:all 로
     * (한 번 클릭·탭하면 6자리 전체가 선택됨). 스타일은 메일 앱 호환을 위해 전부 인라인, 표로 배치.
     * 색은 화면 테마와 같음 (--brand #0046ff, --brand-soft #e8efff), 바깥 배경은 흰색.
     * 들어오는 값은 전부 코드 안의 고정 문구와 숫자 6자리라 이스케이프 필요 없음.
     */
    public static String codeHtml(String title, String where, String code, long minutes) {
        String font = "-apple-system,BlinkMacSystemFont,'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif";
        return """
            <!doctype html>
            <html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
            <title>사공삼(403) %4$s</title></head>
            <body style="margin:0;padding:0;background:#ffffff;">
            <div style="display:none;max-height:0;overflow:hidden;">인증 코드 %1$s · %2$d분 안에 입력해 주세요</div>
            <table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="background:#ffffff;padding:32px 12px;font-family:%3$s;">
              <tr><td align="center">
                <table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="max-width:460px;background:#ffffff;border-radius:16px;border:1px solid #dbe4f5;">
                  <tr><td style="padding:28px 32px 0;">
                    <table role="presentation" cellpadding="0" cellspacing="0"><tr>
                      <!-- 로고: 열쇠 구멍 (Frontend Logo.tsx 와 같은 모양. 메일 앱이 SVG 를 막아서 div 로 그림) -->
                      <td width="32" height="32" valign="top" style="width:32px;height:32px;background:#0046ff;border-radius:9px;">
                        <div style="width:10px;height:10px;margin:7px auto 0;background:#ffffff;border-radius:50%%;font-size:0;line-height:0;">&nbsp;</div>
                        <div style="width:5px;height:8px;margin:-1px auto 0;background:#ffffff;border-radius:0 0 1px 1px;font-size:0;line-height:0;">&nbsp;</div>
                      </td>
                      <td style="padding-left:10px;line-height:1.3;">
                        <div style="font-size:16px;font-weight:800;color:#00236e;">사공삼 <span style="font-size:11px;font-weight:700;letter-spacing:.8px;color:#7b8bb5;">403</span></div>
                        <div style="font-size:12px;color:#6b7280;">권한 있는 사람만 보는 문서 검색</div>
                      </td>
                    </tr></table>
                  </td></tr>
                  <tr><td style="padding:24px 32px 0;">
                    <h1 style="margin:0;font-size:20px;line-height:1.4;color:#1f2937;">%4$s</h1>
                    <p style="margin:8px 0 0;font-size:14px;line-height:1.6;color:#6b7280;">아래 6자리 숫자를 %5$s에 입력해 주세요.</p>
                  </td></tr>
                  <tr><td style="padding:20px 32px 0;">
                    <table role="presentation" width="100%%" cellpadding="0" cellspacing="0"><tr>
                      <td align="center" style="background:#e8efff;border-radius:12px;padding:18px 8px;">
                        <span style="font-family:'SFMono-Regular',Consolas,'Courier New',monospace;font-size:34px;font-weight:700;letter-spacing:8px;color:#0046ff;-webkit-user-select:all;user-select:all;">%1$s</span>
                      </td>
                    </tr></table>
                    <p style="margin:8px 0 0;font-size:12px;color:#9ca3af;text-align:center;">숫자를 한 번 누르면 전체가 선택돼요 · %2$d분 동안 쓸 수 있어요</p>
                  </td></tr>
                  <tr><td style="padding:24px 32px 28px;">
                    <p style="margin:0;padding-top:16px;border-top:1px solid #eef1f6;font-size:12px;line-height:1.6;color:#9ca3af;">
                      본인이 요청하지 않았다면 이 메일은 무시하셔도 됩니다. 코드는 누구에게도 알려 주지 마세요.
                    </p>
                  </td></tr>
                </table>
                <p style="margin:16px 0 0;font-size:11px;color:#9ca3af;">사공삼(403) · 이 메일은 발신 전용이에요</p>
              </td></tr>
            </table>
            </body></html>
            """.formatted(code, minutes, font, title, where);
    }

    /**
     * 보낸 사람 이름. 비면 "사공삼(403)" (프로젝트 이름).
     * .properties·.env 는 ISO-8859-1 로 읽혀서 한글(UTF-8 바이트)이 'ìƒˆë´„' 처럼 깨져 옴 → UTF-8 로 다시 읽어 봄
     */
    public static String senderName(String configured) {
        if (configured == null || configured.isBlank()) return "사공삼(403)";
        String s = configured.strip();
        if (s.chars().allMatch(c -> c <= 0xFF) && s.chars().anyMatch(c -> c >= 0x80)) {
            try {
                return java.nio.charset.StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                        .decode(java.nio.ByteBuffer.wrap(s.getBytes(StandardCharsets.ISO_8859_1))).toString();
            } catch (java.nio.charset.CharacterCodingException e) {
                return s; // 원래 라틴 문자였음
            }
        }
        return s;
    }

    private static ApiException tooMany(String m) {
        return new ApiException(HttpStatus.TOO_MANY_REQUESTS, m);
    }
}
