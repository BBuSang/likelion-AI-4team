package com.saebom.backend.ai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.saebom.backend.exception.ApiException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * AI 서버 (FastAPI /internal/*) 호출. 서비스 키는 X-Service-Key 헤더로 (worklog 4-1-1).
 * 스프링 DB 계정(app_user)은 rag·vault 를 못 읽어서, 직원 목록은 여기로만 봄.
 */
@Component
public class AiClient {

    private final RestClient http;
    private final RestClient chatHttp; // 질문은 검색 + LLM 이라 오래 걸림
    private final ObjectMapper om = new ObjectMapper();

    public AiClient(@Value("${ai.base-url}") String baseUrl, @Value("${ai.service-key}") String serviceKey,
                    @Value("${ai.chat-timeout-ms:120000}") int chatTimeoutMs) {
        this.http = client(baseUrl, serviceKey, 30_000);
        this.chatHttp = client(baseUrl, serviceKey, chatTimeoutMs);
    }

    private static RestClient client(String baseUrl, String serviceKey, int readTimeoutMs) {
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(3_000);
        factory.setReadTimeout(readTimeoutMs);
        return RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(factory)
                .defaultHeader("X-Service-Key", serviceKey)
                .build();
    }

    // ------------------------------------------------------------------ 응답 모양 (AI 서버 app/org/directory.py)

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record TeamInfo(String name, int n) {}

    /** name 은 match·employees·employee·create 에서만 옴 (관리자용). AI 는 snake_case, 프론트로는 camelCase */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Employee(@JsonAlias("emp_id") String empId, String name, String team, String position,
                           String role, @JsonAlias("n_docs") int nDocs,
                           @JsonAlias("same_team") Boolean sameTeam) {}

    // ------------------------------------------------------------------ 호출

    public List<TeamInfo> teams() {
        return call(() -> http.get().uri("/internal/org/teams").retrieve()
                .body(new ParameterizedTypeReference<List<TeamInfo>>() {}));
    }

    public List<Employee> match(String name, String team) {
        Map<String, Object> body = team == null ? Map.of("name", name) : Map.of("name", name, "team", team);
        return call(() -> http.post().uri("/internal/org/match").contentType(MediaType.APPLICATION_JSON)
                .body(body).retrieve().body(new ParameterizedTypeReference<List<Employee>>() {}));
    }

    public List<Employee> employees(String q) {
        return call(() -> http.get().uri(b -> {
            b.path("/internal/org/employees");
            if (q != null && !q.isBlank()) b.queryParam("q", q);
            return b.build();
        }).retrieve().body(new ParameterizedTypeReference<List<Employee>>() {}));
    }

    public Optional<Employee> employee(String empId) {
        try {
            return Optional.ofNullable(call(() -> http.get().uri("/internal/org/employees/{id}", empId)
                    .retrieve().body(Employee.class)));
        } catch (NotFound e) {
            return Optional.empty();
        }
    }

    public Employee createEmployee(String name, String team, String position, String role) {
        var body = new java.util.HashMap<String, Object>();
        body.put("name", name);
        body.put("team", team);
        body.put("position", position);
        body.put("role", role);
        return call(() -> http.post().uri("/internal/org/employees").contentType(MediaType.APPLICATION_JSON)
                .body(body).retrieve().body(Employee.class));
    }

    // ------------------------------------------------------------------ 질문·복원 (AI 서버 app/api/chat.py)

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record SourceDoc(@JsonAlias("doc_id") String docId, @JsonAlias("doc_type") String docType,
                            double score, String snippet) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record OutputCheck(@JsonAlias("unverified_tokens") List<String> unverifiedTokens,
                              @JsonAlias("raw_pii") List<Object> rawPii) {}

    /** answer: 토큰 상태 (실명 아님). LLM 키가 없으면 null */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AskResult(@JsonAlias("masked_query") String maskedQuery, String answer,
                            @JsonAlias("ask_back") boolean askBack, List<SourceDoc> docs, OutputCheck check,
                            String model, Map<String, Double> timing, AgentInfo agent,
                            // 답변 속 사람 중 같은 이름이 여럿인 사람 토큰 → 화면 '동명이인 가능' 경고 (worklog 37장)
                            @JsonAlias("ambiguous_people") List<String> ambiguousPeople) {}

    /** 에이전트 분류 결과 (worklog 25장). route: internal·public·mixed·out_of_scope·clarify, pending: 필요하지만 아직 없는 RAG */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AgentInfo(String route, List<String> sources, String reason, boolean fallback,
                            List<String> pending, String model, @JsonAlias("cost_usd") double costUsd) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RestoreResult(String text, int restored, List<String> denied, List<String> unknown,
                                List<String> docs) {}

    public AskResult ask(AccessUser user, String query, int k) {
        return call(() -> chatHttp.post().uri("/internal/chat/ask").contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("user", user, "query", query, "k", k)).retrieve().body(AskResult.class));
    }

    public RestoreResult restore(AccessUser user, String actor, String text, List<String> docIds, String purpose) {
        var body = new java.util.HashMap<String, Object>();
        body.put("user", user);
        body.put("actor", actor);
        body.put("text", text);
        body.put("doc_ids", docIds);
        body.put("purpose", purpose);
        return call(() -> http.post().uri("/internal/chat/restore").contentType(MediaType.APPLICATION_JSON)
                .body(body).retrieve().body(RestoreResult.class));
    }

    // ------------------------------------------------------------------ 그대로 넘기는 호출 (적재·통계)

    /** GET /internal{path}. 404 는 ApiException 404 */
    public JsonNode getJson(String path, Map<String, ?> query) {
        try {
            return call(() -> http.get().uri(b -> {
                b.path("/internal" + path);
                query.forEach((k, v) -> { if (v != null) b.queryParam(k, v); });
                return b.build();
            }).retrieve().body(JsonNode.class));
        } catch (NotFound e) {
            throw ApiException.notFound("찾을 수 없습니다.");
        }
    }

    /** POST /internal{path} (body 는 JSON). 적재 저장은 임베딩까지 해서 오래 걸릴 수 있어 질문과 같은 대기 시간 */
    public JsonNode postJson(String path, Object body) {
        try {
            return call(() -> chatHttp.post().uri("/internal" + path).contentType(MediaType.APPLICATION_JSON)
                    .body(body).retrieve().body(JsonNode.class));
        } catch (NotFound e) {
            throw ApiException.notFound("찾을 수 없습니다.");
        }
    }

    // ------------------------------------------------------------------

    private static class NotFound extends RuntimeException {}

    private String detail(HttpClientErrorException e, String fallback) {
        try {
            var d = om.readTree(e.getResponseBodyAsString()).get("detail");
            if (d != null && d.isTextual()) return d.asText();
        } catch (Exception ignored) {
            // 아래 기본 문구
        }
        return fallback;
    }

    private <T> T call(java.util.function.Supplier<T> f) {
        try {
            return f.get();
        } catch (HttpClientErrorException e) {
            if (e.getStatusCode() == HttpStatus.NOT_FOUND) throw new NotFound();
            // 403 = 권한 판단 (AI app/core/access.py 의 메시지: 우리가 쓴 한국어 문장이라 그대로 보여 줌)
            if (e.getStatusCode() == HttpStatus.FORBIDDEN) throw ApiException.forbidden(detail(e, "권한이 없습니다."));
            // 400·409 = 입력·상태 문제. AI 쪽 메시지도 우리가 쓴 문장(값 없음)이라 그대로
            if (e.getStatusCode() == HttpStatus.BAD_REQUEST) throw ApiException.badRequest(detail(e, "요청이 올바르지 않습니다."));
            if (e.getStatusCode() == HttpStatus.CONFLICT) throw ApiException.conflict(detail(e, "이미 처리되었습니다."));
            if (e.getStatusCode().value() == 422) throw ApiException.badRequest("요청 형식이 맞지 않습니다.");
            throw new ApiException(HttpStatus.BAD_GATEWAY, "AI 서버가 요청을 거절했습니다 (" + e.getStatusCode().value() + ")");
        } catch (RestClientException e) {
            throw ApiException.badGateway("AI 서버에 연결할 수 없습니다");
        }
    }
}
