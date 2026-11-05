package com.saebom.backend.service;

import com.saebom.backend.entity.User;
import com.saebom.backend.repository.UserRepository;
import com.saebom.backend.security.UD;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Locale;

/** 이전 프로젝트 UserService 의 로그인·회원가입 부분. 프로필 수정·비밀번호 변경은 나중에 가져옴. */
@Service
@RequiredArgsConstructor
public class UserService implements UserDetailsService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public static String normalizeEmail(String email) {
        return email == null ? null : email.trim().toLowerCase(Locale.ROOT);
    }

    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        return userRepository.findByEmail(normalizeEmail(email))
                .map(UD::new)
                .orElseThrow(() -> new UsernameNotFoundException("사용자 없음"));
    }

    public boolean emailExists(String email) {
        return userRepository.existsByEmail(normalizeEmail(email));
    }

    @Transactional
    public User registerUser(String email, String password, String name) {
        String normalized = normalizeEmail(email);
        if (userRepository.existsByEmail(normalized)) {
            throw new DuplicateEmailException();
        }
        User user = User.builder()
                .email(normalized)
                .pw(passwordEncoder.encode(password))
                .name(name.trim())
                .role("USER") // 가입자는 항상 USER. 직책·권한은 관리자가 부여 (다음 단계)
                .build();
        return userRepository.save(user);
    }

    public User findByEmail(String email) {
        return userRepository.findByEmail(normalizeEmail(email))
                .orElseThrow(() -> new UsernameNotFoundException("사용자 없음"));
    }

    public static class DuplicateEmailException extends RuntimeException {
        public DuplicateEmailException() {
            super("이미 존재하는 이메일입니다.");
        }
    }
}
