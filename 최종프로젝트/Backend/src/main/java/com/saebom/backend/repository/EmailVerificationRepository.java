package com.saebom.backend.repository;

import com.saebom.backend.entity.EmailVerification;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;
import java.util.Optional;

public interface EmailVerificationRepository extends JpaRepository<EmailVerification, Long> {
    Optional<EmailVerification> findFirstByEmailAndPurposeOrderByIdDesc(String email, String purpose);
    long countByEmailAndPurposeAndCreatedAtAfter(String email, String purpose, LocalDateTime after);
    long countByRequestIpAndCreatedAtAfter(String requestIp, LocalDateTime after);
}
