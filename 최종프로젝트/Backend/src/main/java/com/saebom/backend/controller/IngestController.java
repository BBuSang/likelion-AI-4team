package com.saebom.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.exception.ApiException;
import com.saebom.backend.security.UD;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 적재·검수 중계 (14장). /api/ingest/** 는 SecurityConfig 에서 관리자만.
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

    public record EditTextRequest(@NotBlank @Size(max = 20_000) String text, @Valid List<Span> spans) {}

    public record ReviewRequest(@NotBlank @Pattern(regexp = "approve|reject") String action,
                                @Valid List<Span> spans, @Size(max = 500) String reason,
                                Map<String, String> choices, @Size(max = 50) String docType) {}

    @PostMapping("/jobs")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public JsonNode upload(@AuthenticationPrincipal UD me, @Valid @RequestBody UploadRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("uploader", me.getEmail());
        body.put("source", req.source());
        body.put("content", req.content());
        body.put("filename", req.filename());
        body.put("doc_type", req.docType() == null || req.docType().isBlank() ? null : req.docType().strip()); // 비우면 AI 가 자동 판단
        return ai.postJson("/ingest/jobs", body);
    }

    /** 받는 녹음 형식 (실제로 풀리는지는 AI 가 다시 확인) */
    private static final List<String> AUDIO_EXT = List.of(".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm");

    /** 녹음 (wav·mp3·m4a 등, 40MB·30분까지). kind = call 상담 · meeting 회의(목소리로 화자 나눔), speakers = 회의 참석자 수(모르면 비움).
     *  받아쓰기·탐지는 AI 가 백그라운드로. 녹음 파일은 어디에도 저장하지 않음 */
    @PostMapping(value = "/audio-jobs", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @ResponseStatus(HttpStatus.ACCEPTED)
    public JsonNode uploadAudio(@AuthenticationPrincipal UD me, @RequestParam("file") MultipartFile file,
                                @RequestParam(value = "docType", required = false) String docType,
                                @RequestParam(value = "kind", defaultValue = "call") String kind,
                                @RequestParam(value = "speakers", required = false) Integer speakers) throws IOException {
        String name = file.getOriginalFilename();
        if (file.isEmpty()) throw ApiException.badRequest("빈 파일입니다.");
        if (name == null || AUDIO_EXT.stream().noneMatch(name.toLowerCase()::endsWith))
            throw ApiException.badRequest("녹음 파일(wav·mp3·m4a·aac·flac·ogg·opus·webm)만 올릴 수 있어요.");
        if (name.length() > 255 || (docType != null && docType.length() > 50)) throw ApiException.badRequest("파일 이름이 너무 깁니다.");
        if (!kind.equals("call") && !kind.equals("meeting")) throw ApiException.badRequest("녹음 종류는 상담 또는 회의예요.");
        if (speakers != null && (speakers < 1 || speakers > 10)) throw ApiException.badRequest("참석자 수는 1~10명이에요.");
        Map<String, Object> q = new HashMap<>();
        q.put("uploader", me.getEmail());
        q.put("filename", name);
        q.put("doc_type", docType == null || docType.isBlank() ? null : docType);
        q.put("kind", kind);
        q.put("speakers", kind.equals("meeting") ? speakers : null);
        return ai.postBytes("/ingest/audio-jobs", q, file.getBytes());
    }

    /** 올리기 화면 드롭다운: 저장된 문서 종류 (많은 순) */
    @GetMapping("/doc-types")
    public JsonNode docTypes() {
        return ai.getJson("/ingest/doc-types", Map.of());
    }

    @GetMapping("/jobs")
    public JsonNode jobs(@RequestParam(defaultValue = "50") int limit) {
        return ai.getJson("/ingest/jobs", Map.of("limit", limit));
    }

    @GetMapping("/jobs/{id}")
    public JsonNode job(@PathVariable long id) {
        return ai.getJson("/ingest/jobs/" + id, Map.of());
    }

    /** 잘못 올린 업로드 취소 (검수 대기 문서는 버리고 이미 저장된 문서는 검색에서 지움). 처리 중이면 AI 가 409 */
    @PostMapping("/jobs/{id}/cancel")
    public JsonNode cancel(@AuthenticationPrincipal UD me, @PathVariable long id) {
        return ai.postJson("/ingest/jobs/" + id + "/cancel", Map.of("actor", me.getEmail()));
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

    /** 검수 때 본문 고치기 (받아쓰기 오류·화자 이름 등). 안 바뀐 곳의 구간은 옮기고 바뀐 곳은 AI 가 다시 탐지 */
    @PostMapping("/docs/{id}/text")
    public JsonNode editText(@AuthenticationPrincipal UD me, @PathVariable long id, @Valid @RequestBody EditTextRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("actor", me.getEmail());
        body.put("text", req.text());
        body.put("spans", req.spans());
        return ai.postJson("/ingest/docs/" + id + "/text", body);
    }

    @PostMapping("/docs/{id}/review")
    public JsonNode review(@AuthenticationPrincipal UD me, @PathVariable long id, @Valid @RequestBody ReviewRequest req) {
        Map<String, Object> body = new HashMap<>();
        body.put("actor", me.getEmail());
        body.put("action", req.action());
        body.put("spans", req.spans());
        body.put("reason", req.reason());
        body.put("choices", req.choices());
        body.put("doc_type", req.docType() == null || req.docType().isBlank() ? null : req.docType().strip());
        return ai.postJson("/ingest/docs/" + id + "/review", body);
    }
}
