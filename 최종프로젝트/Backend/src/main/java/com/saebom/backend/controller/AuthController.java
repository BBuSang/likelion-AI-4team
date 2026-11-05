package com.saebom.backend.controller;

import com.saebom.backend.dto.AuthDtos.*;
import com.saebom.backend.entity.User;
import com.saebom.backend.security.JwtUtil;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.UserService;
import com.saebom.backend.service.UserService.DuplicateEmailException;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

/**
 * 이전 프로젝트 AuthController 에서 로그인·회원가입 관련만 가져옴:
 * check-email, signup, login, refresh, me.
 * 아직 안 가져온 것: 메일 인증(mailSend/mailCheck), 프로필 수정, 비밀번호 변경.
 */
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final UserService userService;
    private final JwtUtil jwtUtil;

    // 이메일 중복 검사
    @PostMapping("/check-email")
    public ResponseEntity<ApiResponse> checkEmail(@Valid @RequestBody EmailCheckRequest req) {
        if (userService.emailExists(req.email())) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(new ApiResponse(false, "이미 존재하는 이메일입니다."));
        }
        return ResponseEntity.ok(new ApiResponse(true, "사용 가능한 이메일입니다."));
    }

    // 회원가입
    @PostMapping("/signup")
    public ResponseEntity<ApiResponse> signup(@Valid @RequestBody SignupRequest req) {
        try {
            userService.registerUser(req.email(), req.password(), req.name());
            return ResponseEntity.status(HttpStatus.CREATED).body(new ApiResponse(true, "회원가입 성공"));
        } catch (DuplicateEmailException e) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(new ApiResponse(false, e.getMessage()));
        }
    }

    // 로그인
    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest req) {
        try {
            var auth = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(UserService.normalizeEmail(req.email()), req.password()));
            UD ud = (UD) auth.getPrincipal();
            User user = ud.getUser();
            String jwt = jwtUtil.generateToken(ud, req.rememberMe());
            return ResponseEntity.ok(new LoginResponse(jwt, "Bearer", user.getEmail(), user.getName(), user.getRole()));
        } catch (AuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ApiResponse(false, "이메일 또는 비밀번호가 올바르지 않습니다."));
        }
    }

    // 토큰 갱신 (로그인 필요)
    @PostMapping("/refresh")
    public ResponseEntity<TokenResponse> refresh(@AuthenticationPrincipal UD ud,
                                                 @RequestHeader("Authorization") String authHeader) {
        String oldToken = authHeader.substring("Bearer ".length());
        return ResponseEntity.ok(new TokenResponse(jwtUtil.refreshToken(oldToken, ud)));
    }

    // 현재 사용자 정보 (로그인 필요)
    @GetMapping("/me")
    public ResponseEntity<UserResponse> me(@AuthenticationPrincipal UD ud) {
        return ResponseEntity.ok(UserResponse.of(userService.findByEmail(ud.getEmail())));
    }
}
