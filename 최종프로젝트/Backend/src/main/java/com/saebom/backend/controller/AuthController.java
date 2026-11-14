package com.saebom.backend.controller;

import com.saebom.backend.dto.AuthDtos.*;
import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import com.saebom.backend.security.JwtUtil;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.UserService;
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
 * check-email, options(선택칸), signup, login, refresh, me.
 * 가입 → PENDING (로그인 불가) → 관리자 승인 → ACTIVE. SUSPENDED 도 로그인 불가.
 */
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final UserService userService;
    private final JwtUtil jwtUtil;

    @PostMapping("/check-email")
    public ResponseEntity<ApiResponse> checkEmail(@Valid @RequestBody EmailCheckRequest req) {
        if (userService.emailExists(req.email())) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(new ApiResponse(false, "이미 존재하는 이메일입니다."));
        }
        return ResponseEntity.ok(new ApiResponse(true, "사용 가능한 이메일입니다."));
    }

    /** 가입 화면 팀·직급 선택칸 (로그인 없이) */
    @GetMapping("/options")
    public OptionsResponse options() {
        return userService.options();
    }

    @PostMapping("/signup")
    public ResponseEntity<ApiResponse> signup(@Valid @RequestBody SignupRequest req) {
        userService.registerUser(req.email(), req.password(), req.name(), req.teamId(), req.positionId());
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(new ApiResponse(true, "가입 신청이 접수되었습니다. 관리자 승인 후 로그인할 수 있습니다."));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest req) {
        UD ud;
        try {
            var auth = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(UserService.normalizeEmail(req.email()), req.password()));
            ud = (UD) auth.getPrincipal();
        } catch (AuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ApiResponse(false, "이메일 또는 비밀번호가 올바르지 않습니다."));
        }
        // 비밀번호가 맞은 뒤에만 상태를 알려 줌
        User user = ud.getUser();
        if (user.getStatus() != UserStatus.ACTIVE) {
            String msg = user.getStatus() == UserStatus.PENDING ? "관리자 승인 대기 중입니다." : "사용이 정지된 계정입니다.";
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(new ApiResponse(false, msg));
        }
        String jwt = jwtUtil.generateToken(ud, req.rememberMe());
        return ResponseEntity.ok(new LoginResponse(jwt, "Bearer", UserResponse.of(user)));
    }

    @PostMapping("/refresh")
    public ResponseEntity<TokenResponse> refresh(@AuthenticationPrincipal UD ud,
                                                 @RequestHeader("Authorization") String authHeader) {
        String oldToken = authHeader.substring("Bearer ".length());
        return ResponseEntity.ok(new TokenResponse(jwtUtil.refreshToken(oldToken, ud)));
    }

    @GetMapping("/me")
    public ResponseEntity<UserResponse> me(@AuthenticationPrincipal UD ud) {
        return ResponseEntity.ok(UserResponse.of(ud.getUser()));
    }
}
