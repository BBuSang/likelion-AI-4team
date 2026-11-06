package com.saebom.backend.config;

import com.saebom.backend.entity.Position;
import com.saebom.backend.entity.Role;
import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import com.saebom.backend.repository.PositionRepository;
import com.saebom.backend.repository.UserRepository;
import com.saebom.backend.service.AdminService;
import com.saebom.backend.service.UserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 서버 시작 때 한 번:
 * 1. 직급 목록 채우기 (없는 것만). 기본 역할은 AI 서버 app/org/infer.py 의 LEAD_TITLES·AUDIT_TITLES 와 같게
 * 2. 관리자가 한 명도 없으면 환경변수 ADMIN_EMAIL·ADMIN_PASSWORD 로 첫 관리자 만들기
 * 3. AI 서버에서 팀 목록 가져오기 (AI 서버가 꺼져 있으면 건너뜀 → 관리자 화면에서 다시)
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DataInitializer implements ApplicationRunner {

    /** 이름, 기본 역할 (순서 = 선택칸 순서) */
    static final List<Map.Entry<String, Role>> POSITIONS = List.of(
            Map.entry("사원", Role.STAFF), Map.entry("행원", Role.STAFF), Map.entry("주임", Role.STAFF),
            Map.entry("계장", Role.STAFF), Map.entry("대리", Role.STAFF), Map.entry("선임", Role.STAFF),
            Map.entry("과장", Role.STAFF), Map.entry("책임", Role.STAFF), Map.entry("수석", Role.STAFF),
            Map.entry("매니저", Role.STAFF), Map.entry("상담원", Role.STAFF), Map.entry("심사역", Role.STAFF),
            Map.entry("차장", Role.LEAD), Map.entry("부장", Role.LEAD), Map.entry("팀장", Role.LEAD),
            Map.entry("실장", Role.LEAD), Map.entry("센터장", Role.LEAD),
            Map.entry("본부장", Role.AUDITOR), Map.entry("이사", Role.AUDITOR), Map.entry("대표", Role.AUDITOR));

    private final PositionRepository positions;
    private final UserRepository users;
    private final PasswordEncoder passwordEncoder;
    private final AdminService adminService;

    @Value("${app.admin.email:}")
    private String adminEmail;
    @Value("${app.admin.password:}")
    private String adminPassword;
    @Value("${app.admin.name:}")
    private String adminName;
    @Value("${app.teams.sync-on-start:true}")
    private boolean syncTeamsOnStart;

    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        seedPositions();
        bootstrapAdmin();
        if (syncTeamsOnStart) {
            try {
                var r = adminService.syncTeams(AdminService.SYSTEM);
                log.info("팀 목록: AI 서버에서 {}개 추가 (전체 {})", r.added(), r.total());
            } catch (Exception e) {
                log.warn("팀 목록 가져오기 건너뜀 (AI 서버 확인): {}", e.getMessage());
            }
        }
    }

    private void seedPositions() {
        for (int i = 0; i < POSITIONS.size(); i++) {
            var p = POSITIONS.get(i);
            if (positions.findByName(p.getKey()).isEmpty()) {
                positions.save(Position.builder().name(p.getKey()).sortOrder(i + 1).defaultRole(p.getValue()).build());
            }
        }
    }

    private void bootstrapAdmin() {
        if (users.existsByAdminTrue()) return;
        if (adminEmail.isBlank() || adminPassword.isBlank()) {
            log.warn("관리자 계정이 없습니다. ADMIN_EMAIL·ADMIN_PASSWORD 를 설정하고 다시 시작하세요.");
            return;
        }
        if (adminPassword.length() < 12) {
            log.warn("ADMIN_PASSWORD 가 12자보다 짧아 첫 관리자를 만들지 않았습니다.");
            return;
        }
        String email = UserService.normalizeEmail(adminEmail);
        if (users.existsByEmail(email)) {
            // 이미 있는 일반 계정을 환경변수로 관리자로 올리지는 않음 (실수로 권한 올리기 방지)
            log.warn("ADMIN_EMAIL 이 이미 일반 계정으로 있어 첫 관리자를 만들지 않았습니다.");
            return;
        }
        User admin = users.save(User.builder()
                .email(email)
                .pw(passwordEncoder.encode(adminPassword))
                .name(adminName.isBlank() ? "관리자" : adminName)
                .role(Role.STAFF) // 관리자는 계정 관리용. 데이터 접근 역할은 따로 (직원 연결 없음)
                .admin(true)
                .status(UserStatus.ACTIVE)
                .approvedAt(LocalDateTime.now())
                .build());
        adminService.log(AdminService.SYSTEM, admin.getId(), "bootstrap", Map.of("admin", new Object[]{false, true}));
        log.info("첫 관리자 계정을 만들었습니다: {}", email);
    }
}
