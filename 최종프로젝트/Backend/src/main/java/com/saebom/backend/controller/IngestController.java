package com.saebom.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.security.UD;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 적재·검수 중계 (worklog 9장, 14장). /api/ingest/** 는 SecurityConfig 에서 관리자만.
 * 누가 했는지(uploader·actor)는 화면 값이 아니라 로그인 계정 이메일로 채움.
 * 검수 화면은 원문(개인정보)을 보여 주므로 AI 쪽이 조회할 때마다 review_log 에 남김.
 */
@RestController
@RequestMapping("/api/ingest")
@RequiredArgsConstructor
public class IngestController {

    private final AiClient ai;

    public record UploadRequest(@NotBlank @Pattern(regexp = "jsonl|text") String source,
                                @NotBlank @Size(max = 20_000_000) String content,
                                @Size(max = 255) String filename,
                                @Size(max = 50) String docType) {}

    public record Span(@NotBlank String label, int start, int end, String text) {}

    public record PreviewRequest(@Valid List<Span> spans, Map<String, String> choices) {}

    public record ReviewRequest(@NotBlank @Pattern(regexp = "approve|reject") String action,
                                @Valid List<Span> spans, @Size(max = 500) String reason,
                                Map<String, String> choices) {}

    @PostMapping("/jobs")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public JsonNode upload(@AuthenticationPrincipal UD me, @Valid @RequestBody UploadRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("uploader", me.getEmail());
        body.put("source", req.source());
        body.put("content", req.content());
        body.put("filename", req.filename());
        body.put("doc_type", req.docType());
        return ai.postJson("/ingest/jobs", body);
    }

    @GetMapping("/jobs")
    public JsonNode jobs(@RequestParam(defaultValue = "50") int limit) {
        return ai.getJson("/ingest/jobs", Map.of("limit", limit));
    }

    @GetMapping("/jobs/{id}")
    public JsonNode job(@PathVariable long id) {
        return ai.getJson("/ingest/jobs/" + id, Map.of());
    }

    @GetMapping("/jobs/{id}/docs")
    public JsonNode docs(@PathVariable long id) {
        return ai.getJson("/ingest/jobs/" + id + "/docs", Map.of());
    }

    @GetMapping("/docs/{id}")
    public JsonNode doc(@AuthenticationPrincipal UD me, @PathVariable long id) {
        return ai.getJson("/ingest/docs/" + id, Map.of("actor", me.getEmail()));
    }

    @PostMapping("/docs/{id}/preview-links")
    public JsonNode preview(@PathVariable long id, @Valid @RequestBody PreviewRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("spans", req.spans() == null ? List.of() : req.spans());
        body.put("choices", req.choices());
        return ai.postJson("/ingest/docs/" + id + "/preview-links", body);
    }

    @PostMapping("/docs/{id}/review")
    public JsonNode review(@AuthenticationPrincipal UD me, @PathVariable long id, @Valid @RequestBody ReviewRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("actor", me.getEmail());
        body.put("action", req.action());
        body.put("spans", req.spans());
        body.put("reason", req.reason());
        body.put("choices", req.choices());
        return ai.postJson("/ingest/docs/" + id + "/review", body);
    }
}
