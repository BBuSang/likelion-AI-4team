package com.saebom.backend.service;

import com.saebom.backend.dto.AuthDtos.Option;
import com.saebom.backend.dto.AuthDtos.OptionsResponse;
import com.saebom.backend.entity.Position;
import com.saebom.backend.entity.Team;
import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.TeamRepository;
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

/** 로그인·회원가입. 가입하면 PENDING → 관리자가 승인해야 로그인 가능 (worklog 11-4) */
@Service
@RequiredArgsConstructor
public class UserService implements UserDetailsService {

    private final UserRepository userRepository;
    private final TeamRepository teamRepository;
    private final PositionRepository positionRepository;
    private final PasswordEncoder passwordEncoder;
    private final EmailVerificationService emailVerification;

    public static String normalizeEmail(String email) {
        return email == null ? null : email.trim().toLowerCase(Locale.ROOT);
    }

    /** 이름은 띄어쓰기 없이 (AI 서버 직원 이름과 같은 모양) */
    public static String normalizeName(String name) {
        return name == null ? null : name.replaceAll("\\s+", "");
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

    /** 가입 화면 선택칸: 쓰는 팀 + 직급 */
    @Transactional(readOnly = true)
    public OptionsResponse options() {
        return new OptionsResponse(
                teamRepository.findByActiveTrueOrderByNameAsc().stream().map(t -> new Option(t.getId(), t.getName())).toList(),
                positionRepository.findAllByOrderBySortOrderAsc().stream().map(p -> new Option(p.getId(), p.getName())).toList(),
                emailVerification.signupVerificationRequired());
    }

    @Transactional
    public User registerUser(String email, String password, String name, Long teamId, Long positionId) {
        String normalized = normalizeEmail(email);
        if (userRepository.existsByEmail(normalized)) {
            throw ApiException.conflict("이미 존재하는 이메일입니다.");
        }
        Team team = teamRepository.findById(teamId).filter(Team::isActive)
                .orElseThrow(() -> ApiException.badRequest("팀을 목록에서 골라 주세요."));
        Position position = positionRepository.findById(positionId)
                .orElseThrow(() -> ApiException.badRequest("직급을 목록에서 골라 주세요."));
        String cleanName = normalizeName(name);
        if (cleanName.isEmpty()) throw ApiException.badRequest("이름을 입력해 주세요.");
        // 이메일 인증 확인·소진 (같은 트랜잭션: 저장이 실패하면 인증도 그대로 남음, worklog 41장)
        emailVerification.consumeSignup(normalized);
        User user = User.builder()
                .email(normalized)
                .pw(passwordEncoder.encode(password))
                .name(cleanName)
                .team(team)
                .position(position)
                .role(position.getDefaultRole()) // 승인 전엔 로그인이 안 돼서 쓰이지 않음. 승인 때 관리자가 확정
                .status(UserStatus.PENDING)
                .build();
        return userRepository.save(user);
    }

    @Transactional
    public void changePassword(Long userId, String current, String next) {
        User u = userRepository.findById(userId).orElseThrow(() -> ApiException.notFound("사용자 없음"));
        if (!passwordEncoder.matches(current, u.getPw())) throw ApiException.badRequest("현재 비밀번호가 맞지 않아요.");
        if (current.equals(next)) throw ApiException.badRequest("지금과 다른 비밀번호를 입력해 주세요.");
        u.setPw(passwordEncoder.encode(next));
    }

    /** 이메일 인증(RESET)을 마친 뒤 새 비밀번호. 인증 소진과 저장은 같은 트랜잭션 (worklog 42장) */
    @Transactional
    public void resetPassword(String email, String next) {
        String normalized = normalizeEmail(email);
        emailVerification.consumeReset(normalized);
        User u = userRepository.findByEmail(normalized)
                .orElseThrow(() -> ApiException.badRequest("이메일 인증을 먼저 마쳐 주세요."));
        u.setPw(passwordEncoder.encode(next));
    }

    public User findByEmail(String email) {
        return userRepository.findByEmail(normalizeEmail(email))
                .orElseThrow(() -> new UsernameNotFoundException("사용자 없음"));
    }
}
