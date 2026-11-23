package com.saebom.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.saebom.backend.ai.AiClient;
import com.saebom.backend.exception.ApiException;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 공개 자료 보기 (관리자 문서 탭의 법령·금융상품·금융당국 회신). /api/admin/** 라 관리자만.
 * 읽기만 — AI 서버 /internal/public-docs 를 그대로 넘겨 줌.
 */
@RestController
@RequestMapping("/api/admin/public-docs")
@RequiredArgsConstructor
public class AdminPublicDocsController {

    private final AiClient ai;

    /** 출처는 정해진 셋만 (AI 주소에 그대로 붙이므로) */
    private static String checkSource(String source) {
        if (!source.matches("product|law|noaction")) throw ApiException.badRequest("자료 출처가 아닙니다.");
        return source;
    }

    /** 문서 ID: 상품 문서 D001, 법령 law:284415, 회신 사례 번호 — 영문·숫자·-_: 만 */
    private static String checkId(String id) {
        if (!id.matches("[A-Za-z0-9_:-]{1,80}")) throw ApiException.badRequest("문서 ID 형식이 아닙니다.");
        return id;
    }

    @GetMapping("/{source}")
    public JsonNode list(@PathVariable String source, @RequestParam(required = false) String q,
                         @RequestParam(defaultValue = "30") int limit, @RequestParam(defaultValue = "0") int offset) {
        Map<String, Object> query = new HashMap<>();
        query.put("q", q);
        query.put("limit", limit);
        query.put("offset", offset);
        return ai.getJson("/public-docs/" + checkSource(source), query);
    }

    @GetMapping("/{source}/{id}")
    public JsonNode detail(@PathVariable String source, @PathVariable String id) {
        return ai.getJson("/public-docs/" + checkSource(source) + "/" + checkId(id), Map.of());
    }
}
