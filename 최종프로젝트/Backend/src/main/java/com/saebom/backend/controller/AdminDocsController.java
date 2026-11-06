package com.saebom.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.repository.TeamRepository;
import com.saebom.backend.security.UD;
import com.saebom.backend.service.AdminService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 저장된 문서 보기·지우기 (worklog 15장). /api/admin/** 라 관리자만.
 * 글은 가명처리된 토큰 상태만 (원문 없음). 지우면 AI 쪽 deletion_log + 스프링 admin_audit_log 둘 다 남김.
 */
@RestController
@RequestMapping("/api/admin/docs")
@RequiredArgsConstructor
public class AdminDocsController {

    private final AiClient ai;
    private final AdminService admin;
    private final TeamRepository teams;

    public record DeleteRequest(@NotBlank @Size(max = 500) String reason) {}

    /** team 은 팀 목록(app.teams)에 있는 이름, ownerEmpId 는 직원 번호 (비우면 담당자 없음) */
    public record MetaRequest(@NotBlank @Pattern(regexp = "public|team|restricted") String visibility,
                              @Size(max = 50) String team, @Size(max = 20) String ownerEmpId) {}

    @GetMapping
    public JsonNode list(@RequestParam(required = false) String q, @RequestParam(required = false) String visibility,
                         @RequestParam(required = false) String source,
                         @RequestParam(defaultValue = "50") int limit, @RequestParam(defaultValue = "0") int offset) {
        Map<String, Object> query = new HashMap<>();
        query.put("q", q);
        query.put("visibility", visibility);
        query.put("source", source);
        query.put("limit", limit);
        query.put("offset", offset);
        return ai.getJson("/docs", query);
    }

    @GetMapping("/deletion-log")
    public JsonNode deletionLog() {
        return ai.getJson("/docs/deletion-log", Map.of("limit", 100));
    }

    /** 문서 ID 는 영문·숫자·-_ 만 (AI 주소에 그대로 붙이므로) */
    private static String checkId(String docId) {
        if (!docId.matches("[A-Za-z0-9_-]{1,80}")) throw ApiException.badRequest("문서 ID 형식이 아닙니다.");
        return docId;
    }

    @GetMapping("/{docId}")
    public JsonNode doc(@PathVariable String docId) {
        return ai.getJson("/docs/" + checkId(docId), Map.of());
    }

    @PostMapping("/{docId}/meta")
    public JsonNode updateMeta(@AuthenticationPrincipal UD me, @PathVariable String docId, @Valid @RequestBody MetaRequest req) {
        String team = req.team() == null || req.team().isBlank() ? null : req.team().trim();
        if (team != null && teams.findByName(team).isEmpty()) throw ApiException.badRequest("팀 목록에 없는 팀입니다: " + team);
        String owner = req.ownerEmpId() == null || req.ownerEmpId().isBlank() ? null : req.ownerEmpId().trim();
        Map<String, Object> body = new HashMap<>();
        body.put("visibility", req.visibility());
        body.put("team", team);
        body.put("owner_emp_id", owner);
        JsonNode out = ai.postJson("/docs/" + checkId(docId) + "/meta", body);
        Map<String, Object[]> changes = new LinkedHashMap<>();
        changes.put("doc", new Object[]{docId, docId});
        for (String k : new String[]{"visibility", "team", "owner_emp_id"}) {
            JsonNode b = out.path("before").path(k), a = out.path("after").path(k);
            if (!b.equals(a)) changes.put(k, new Object[]{b.isNull() ? null : b.asText(), a.isNull() ? null : a.asText()});
        }
        admin.log(me.getUser().getId(), null, "update_doc", changes);
        return out;
    }

    @PostMapping("/{docId}/delete")
    public JsonNode delete(@AuthenticationPrincipal UD me, @PathVariable String docId, @Valid @RequestBody DeleteRequest req) {
        JsonNode out = ai.postJson("/docs/" + checkId(docId) + "/delete", Map.of("actor", me.getEmail(), "reason", req.reason().trim()));
        Map<String, Object[]> changes = new LinkedHashMap<>();
        changes.put("doc", new Object[]{docId, null});
        changes.put("chunks", new Object[]{out.path("chunks").asInt(), 0});
        admin.log(me.getUser().getId(), null, "delete_doc", changes);
        return out;
    }
}
