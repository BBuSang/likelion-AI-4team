package com.saebom.backend.repository;

import com.saebom.backend.entity.Position;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface PositionRepository extends JpaRepository<Position, Long> {
    List<Position> findAllByOrderBySortOrderAsc();
    Optional<Position> findByName(String name);
}
