package com.saebom.backend.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.Map;
import java.util.function.Function;

/** JWT 발급·검증 (이전 프로젝트 그대로). 비밀키는 환경 변수 JWT_SECRET (32바이트 이상). */
@Component
public class JwtUtil {

    private static final long HOUR = 60 * 60 * 1000L;

    private final SecretKey signingKey;

    public JwtUtil(@Value("${jwt.secret}") String secret) {
        this.signingKey = Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
    }

    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }

    public Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }

    public <T> T extractClaim(String token, Function<Claims, T> resolver) {
        return resolver.apply(extractAllClaims(token));
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parser().verifyWith(signingKey).build().parseSignedClaims(token).getPayload();
    }

    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }

    /** rememberMe: 24시간, 아니면 1시간 */
    public String generateToken(UserDetails userDetails, boolean rememberMe) {
        long ttl = rememberMe ? 24 * HOUR : HOUR;
        long now = System.currentTimeMillis();
        return Jwts.builder()
                .claims(Map.of("rememberMe", rememberMe))
                .subject(userDetails.getUsername())
                .issuedAt(new Date(now))
                .expiration(new Date(now + ttl))
                .signWith(signingKey, Jwts.SIG.HS256)
                .compact();
    }

    /** 기존 토큰의 rememberMe 설정을 유지하며 새 토큰 발급 */
    public String refreshToken(String token, UserDetails userDetails) {
        Boolean rememberMe = extractAllClaims(token).get("rememberMe", Boolean.class);
        return generateToken(userDetails, Boolean.TRUE.equals(rememberMe));
    }

    public boolean validateToken(String token, UserDetails userDetails) {
        return extractUsername(token).equals(userDetails.getUsername()) && !isTokenExpired(token);
    }
}
