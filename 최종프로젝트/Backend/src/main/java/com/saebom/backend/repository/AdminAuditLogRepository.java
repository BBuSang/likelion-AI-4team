package com.saebom.backend.repository;

import com.saebom.backend.entity.AdminAuditLog;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface AdminAuditLogRepository extends JpaRepository<AdminAuditLog, Long> {
    List<AdminAuditLog> findAllByOrderByIdDesc(Pageable page);
    List<AdminAuditLog> findByTargetUserIdOrderByIdDesc(Long targetUserId);
}
