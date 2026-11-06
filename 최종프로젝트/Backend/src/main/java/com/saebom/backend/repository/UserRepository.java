package com.saebom.backend.repository;

import com.saebom.backend.entity.User;
import com.saebom.backend.entity.UserStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
    boolean existsByEmail(String email);
    boolean existsByAdminTrue();
    Optional<User> findByEmpId(String empId);
    List<User> findByStatusOrderByCreatedAtAsc(UserStatus status);
    List<User> findAllByOrderByCreatedAtDesc();
}
