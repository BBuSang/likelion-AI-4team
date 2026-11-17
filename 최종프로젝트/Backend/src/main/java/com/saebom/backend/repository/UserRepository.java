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

    /** 새 직원 번호(U…)를 겹치지 않게 만들려고 AI 서버에 넘기는 목록 (worklog 44장) */
    @org.springframework.data.jpa.repository.Query("select u.empId from User u where u.empId like 'U%'")
    List<String> findNewStyleEmpIds();
    List<User> findByStatusOrderByCreatedAtAsc(UserStatus status);
    List<User> findAllByOrderByCreatedAtDesc();
}
