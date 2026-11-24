package com.saebom.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;

/**
 * 채팅 질문 원문 암호화. 사용자가 친 질문엔 실명이 들어 있을 수 있어 DB 에는 AES-256-GCM 으로만 저장하고,
 * 대화 주인에게 보여 줄 때만 풂 (관리자 열람·DB 덤프로는 안 보임).
 * 키 = SHA-256("saebom-chat-query|" + CHAT_QUERY_KEY). CHAT_QUERY_KEY 가 없으면 JWT 비밀키에서 만듦 —
 * 어느 쪽이든 키가 바뀌면 예전 질문은 못 풀어서 가명처리된 질문으로 보임 (그 밖엔 동작 같음).
 * 저장 형식: base64(12바이트 IV + 암호문·태그)
 */
@Component
public class ChatCipher {

    private static final int IV_LEN = 12;
    private static final int TAG_BITS = 128;

    private final SecretKeySpec key;
    private final SecureRandom random = new SecureRandom();

    public ChatCipher(@Value("${app.chat.query-key:}") String queryKey, @Value("${jwt.secret}") String jwtSecret) {
        String base = queryKey == null || queryKey.isBlank() ? jwtSecret : queryKey;
        try {
            byte[] k = MessageDigest.getInstance("SHA-256")
                    .digest(("saebom-chat-query|" + base).getBytes(StandardCharsets.UTF_8));
            this.key = new SecretKeySpec(k, "AES");
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException(e);
        }
    }

    public String encrypt(String plain) {
        if (plain == null) return null;
        try {
            byte[] iv = new byte[IV_LEN];
            random.nextBytes(iv);
            Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
            c.init(Cipher.ENCRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, iv));
            byte[] ct = c.doFinal(plain.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(ByteBuffer.allocate(IV_LEN + ct.length).put(iv).put(ct).array());
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("질문을 암호화하지 못했습니다", e);
        }
    }

    /** 못 풀면(키가 바뀜·손상) null → 화면은 가명처리된 질문으로 */
    public String decrypt(String stored) {
        if (stored == null || stored.isEmpty()) return null;
        try {
            byte[] all = Base64.getDecoder().decode(stored);
            if (all.length <= IV_LEN) return null;
            Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
            c.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, all, 0, IV_LEN));
            return new String(c.doFinal(all, IV_LEN, all.length - IV_LEN), StandardCharsets.UTF_8);
        } catch (GeneralSecurityException | IllegalArgumentException e) {
            return null;
        }
    }
}
