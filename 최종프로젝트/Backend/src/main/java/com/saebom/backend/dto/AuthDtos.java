package com.saebom.backend.dto;

import com.saebom.backend.entity.Role;
import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.util.List;

/**
 * 로그인·회원가입 요청/응답 모양.
 * 이전 프로젝트는 @RequestParam(주소 뒤 ?pw=...)이라 비밀번호가 서버 접속 기록에 남을 수 있어 JSON 본문으로 바꿈.
 */
public final class AuthDtos {

    private AuthDtos() {}

    /** 팀·직급은 선택칸에서 고른 id (GET /api/auth/options) */
    public record SignupRequest(
            @NotBlank @Email @Size(max = 100) String email,
            @NotBlank @Size(min = 8, max = 72) String password, // BCrypt 는 72바이트까지만 씀
            @NotBlank @Size(max = 50) String name,
            @NotNull Long teamId,
            @NotNull Long positionId) {}

    public record LoginRequest(
            @NotBlank String email,
            @NotBlank String password,
            boolean rememberMe) {}

    public record EmailCheckRequest(@NotBlank @Email String email) {}

    public record LoginResponse(String token, String type, UserResponse user) {}

    public record TokenResponse(String token) {}

    /** 본인 정보. 비밀번호 해시·직원 번호는 내보내지 않음 */
    public record UserResponse(Long id, String email, String name, String team, String position,
                               Role role, boolean admin, UserStatus status) {
        public static UserResponse of(User u) {
            return new UserResponse(u.getId(), u.getEmail(), u.getName(),
                    u.getTeam() == null ? null : u.getTeam().getName(),
                    u.getPosition() == null ? null : u.getPosition().getName(),
                    u.getRole(), u.isAdmin(), u.getStatus());
        }
    }

    /** 가입 화면 선택칸 */
    public record Option(Long id, String name) {}

    /** emailVerification: 가입 전에 이메일 인증이 필요한지 (서버 설정 app.mail.verify-signup) */
    public record OptionsResponse(List<Option> teams, List<Option> positions, boolean emailVerification) {}

    public record EmailCodeRequest(@NotBlank @Email @Size(max = 100) String email) {}

    public record EmailVerifyRequest(@NotBlank @Email @Size(max = 100) String email,
                                     @NotBlank @Size(max = 10) String code) {}

    /** 비밀번호 재설정: 이메일 인증을 마친 뒤 새 비밀번호 */
    public record PasswordResetRequest(@NotBlank @Email @Size(max = 100) String email,
                                       @NotBlank @Size(min = 8, max = 72) String password) {}

    public record ApiResponse(boolean success, String message) {}
}
