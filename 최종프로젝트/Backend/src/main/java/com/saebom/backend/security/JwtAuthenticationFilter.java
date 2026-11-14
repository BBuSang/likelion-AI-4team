package com.saebom.backend.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/** Authorization: Bearer 토큰을 읽어 로그인 상태로 만듦. 토큰이 틀리거나 계정이 ACTIVE 가 아니면 로그인 안 된 상태로 통과 (→ 401). */
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final UserDetailsService userDetailsService;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {

        String header = request.getHeader("Authorization");
        if (header != null && header.startsWith("Bearer ")
                && SecurityContextHolder.getContext().getAuthentication() == null) {
            String jwt = header.substring(7);
            try {
                String username = jwtUtil.extractUsername(jwt);
                // 요청마다 DB 에서 읽음 → 정지·역할 변경이 이미 받은 토큰에도 바로 적용
                UserDetails user = userDetailsService.loadUserByUsername(username);
                boolean active = user instanceof UD ud && ud.isActive();
                if (active && jwtUtil.validateToken(jwt, user)) {
                    var auth = new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities());
                    auth.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(auth);
                }
            } catch (Exception e) {
                logger.warn("JWT 인증 실패: " + e.getClass().getSimpleName());
            }
        }
        chain.doFilter(request, response);
    }
}
