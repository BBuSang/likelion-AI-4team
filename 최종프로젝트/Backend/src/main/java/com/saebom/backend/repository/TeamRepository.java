package com.saebom.backend.repository;

import com.saebom.backend.entity.Team;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface TeamRepository extends JpaRepository<Team, Long> {
    Optional<Team> findByName(String name);
    List<Team> findByActiveTrueOrderByNameAsc();
    List<Team> findAllByOrderByNameAsc();
}
