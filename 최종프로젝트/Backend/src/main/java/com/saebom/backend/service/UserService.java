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
                positionRepository.findAllByOrderBySortOrderAsc().stream().map(p -> new Option(p.getId(), p.getName())).toList());
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

    public User findByEmail(String email) {
        return userRepository.findByEmail(normalizeEmail(email))
                .orElseThrow(() -> new UsernameNotFoundException("사용자 없음"));
    }
}
