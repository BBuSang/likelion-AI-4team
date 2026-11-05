package com.saebom.backend.dto;

import com.saebom.backend.entity.User;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * 로그인·회원가입 요청/응답 모양.
 * 이전 프로젝트는 @RequestParam(주소 뒤 ?pw=...)이라 비밀번호가 서버 접속 기록에 남을 수 있어 JSON 본문으로 바꿈.
 */
public final class AuthDtos {

    private AuthDtos() {}

    public record SignupRequest(
            @NotBlank @Email @Size(max = 100) String email,
            @NotBlank @Size(min = 8, max = 72) String password, // BCrypt 는 72바이트까지만 씀
            @NotBlank @Size(max = 50) String name) {}

    public record LoginRequest(
            @NotBlank String email,
            @NotBlank String password,
            boolean rememberMe) {}

    public record EmailCheckRequest(@NotBlank @Email String email) {}

    public record LoginResponse(String token, String type, String email, String name, String role) {}

    public record TokenResponse(String token) {}

    /** /me 응답: 비밀번호 해시는 내보내지 않음 (이전 프로젝트는 User 엔티티를 그대로 돌려줘 해시가 나갔음) */
    public record UserResponse(Long id, String email, String name, String role) {
        public static UserResponse of(User u) {
            return new UserResponse(u.getId(), u.getEmail(), u.getName(), u.getRole());
        }
    }

    public record ApiResponse(boolean success, String message) {}
}
