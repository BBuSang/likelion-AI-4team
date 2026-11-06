package com.saebom.backend.security;

import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Spring Security 가 쓰는 로그인 사용자 정보 (이메일 = username).
 * 요청마다 DB 에서 새로 읽은 User 로 만들어서, 관리자가 역할·상태를 바꾸면 다음 요청부터 바로 반영 (worklog 11-4).
 * 권한: ROLE_USER(모두) + ROLE_ADMIN(관리자) + DATA_STAFF / DATA_LEAD / DATA_AUDITOR (데이터 접근 역할)
 */
@Getter
public class UD implements UserDetails {

    private final User user;

    public UD(User user) {
        this.user = user;
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        List<GrantedAuthority> out = new ArrayList<>();
        out.add(new SimpleGrantedAuthority("ROLE_USER"));
        if (user.isAdmin()) out.add(new SimpleGrantedAuthority("ROLE_ADMIN"));
        out.add(new SimpleGrantedAuthority("DATA_" + user.getRole().name()));
        return out;
    }

    /** ACTIVE 만. 로그인은 비밀번호 확인 뒤 컨트롤러에서 따로 막음 (비밀번호 모르는 사람에게 상태를 안 알려 주려고) */
    public boolean isActive() {
        return user.getStatus() == UserStatus.ACTIVE;
    }

    @Override
    public String getPassword() {
        return user.getPw();
    }

    @Override
    public String getUsername() {
        return user.getEmail();
    }

    public String getEmail() {
        return user.getEmail();
    }

    public String getName() {
        return user.getName();
    }
}
