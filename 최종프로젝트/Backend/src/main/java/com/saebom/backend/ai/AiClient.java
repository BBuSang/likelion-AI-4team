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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * AI 서버 (FastAPI /internal/*) 호출. 서비스 키는 X-Service-Key 헤더로.
 * 스프링 DB 계정(app_user)은 rag·vault 를 못 읽어서, 직원 목록은 여기로만 봄.
 */
@Component
public class AiClient {

    private final RestClient http;
    private final RestClient chatHttp; // 질문은 검색 + LLM 이라 오래 걸림
    private final ObjectMapper om = new ObjectMapper();
    // 답변 스트리밍용: RestClient 는 응답을 다 받은 뒤에 넘겨서 SSE 를 그대로 흘려보낼 수 없음
    // HTTP/1.1 고정: 기본값(HTTP/2)은 http:// 에 'h2c 업그레이드' 머리를 붙여 보내는데, uvicorn 이 본문 있는 POST 의
    // 업그레이드 요청을 처리하지 못해 본문이 빠짐 → AI 가 422 ("요청 형식이 맞지 않습니다")
    private final HttpClient streamHttp = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(3)).build();
    private final String baseUrl;
    private final String serviceKey;
    private final int chatTimeoutMs;

    public AiClient(@Value("${ai.base-url}") String baseUrl, @Value("${ai.service-key}") String serviceKey,
                    @Value("${ai.chat-timeout-ms:120000}") int chatTimeoutMs) {
        this.http = client(baseUrl, serviceKey, 30_000);
        this.chatHttp = client(baseUrl, serviceKey, chatTimeoutMs);
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.serviceKey = serviceKey;
        this.chatTimeoutMs = chatTimeoutMs;
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

    /** taken: 계정에 이미 연결된 직원 번호 — AI 가 새 번호를 이것들과 겹치지 않게 고름 */
    public Employee createEmployee(String name, String team, String position, String role, List<String> taken) {
        var body = new java.util.HashMap<String, Object>();
        body.put("name", name);
        body.put("team", team);
        body.put("position", position);
        body.put("role", role);
        body.put("taken", taken == null ? List.of() : taken);
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
                            // 답변 속 사람 중 같은 이름이 여럿인 사람 토큰 → 화면 '동명이인 가능' 경고
                            @JsonAlias("ambiguous_people") List<String> ambiguousPeople,
                            // 질문에 쓴 이름 중 데이터에 없는 사람의 토큰 (이름 원문은 없음) → 화면이 질문 원문의 그 말로 보여 줌
                            @JsonAlias("query_only_tokens") List<String> queryOnlyTokens) {}

    /** 에이전트 분류 결과. route: internal·public·mixed·out_of_scope·clarify, pending: 필요하지만 아직 없는 RAG */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AgentInfo(String route, List<String> sources, String reason, boolean fallback,
                            List<String> pending, String model, @JsonAlias("cost_usd") double costUsd,
                            // 앞 대화를 이어 다시 쓴 질문 (토큰 상태). 이어지는 질문이 아니면 null
                            String followup) {}

    /** 같은 대화의 앞 턴 — AI 가 이어지는 질문일 때만 씀. query = 원문(AI 서버 안에서만), 나머지는 토큰 상태 */
    public record Turn(String query, String maskedQuery, String answer) {
        Map<String, Object> body() {
            Map<String, Object> m = new java.util.LinkedHashMap<>();
            m.put("query", query);
            m.put("masked_query", maskedQuery);
            m.put("answer", answer);
            return m;
        }
    }

    private static Map<String, Object> askBody(AccessUser user, String query, int k, List<Turn> history) {
        Map<String, Object> m = new java.util.LinkedHashMap<>();
        m.put("user", user);
        m.put("query", query);
        m.put("k", k);
        m.put("history", history == null ? List.of() : history.stream().map(Turn::body).toList());
        return m;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RestoreResult(String text, int restored, List<String> denied, List<String> outside, List<String> unknown,
                                List<String> docs, List<AccessWarning> warnings) {}

    /** 실명 보기 직후 본인에게 보여 줄 이상 접근 경고 */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AccessWarning(String rule, @JsonAlias("rule_ko") String ruleKo, String severity, String detail) {}

    public AskResult ask(AccessUser user, String query, int k) {
        return ask(user, query, k, List.of());
    }

    public AskResult ask(AccessUser user, String query, int k, List<Turn> history) {
        return call(() -> chatHttp.post().uri("/internal/chat/ask").contentType(MediaType.APPLICATION_JSON)
                .body(askBody(user, query, k, history)).retrieve().body(AskResult.class));
    }

    /**
     * 답변 스트리밍: AI 서버 /internal/chat/ask/stream 의 SSE(단계 · 출력 검사를 거친 답변 조각 · 마지막 결과)를
     * 받는 대로 out 에 그대로 넘김. AI 가 흐름을 열기 전에 거절하면(403·502 등) ask 와 같은 오류로 던지고,
     * onStart(응답 머리 설정)는 흐름이 열린 뒤에만 부름 — 그 전에 실패하면 화면은 보통 오류 응답(JSON)을 받음.
     */
    public void askStream(AccessUser user, String query, int k, Runnable onStart, OutputStream out) throws IOException {
        askStream(user, query, k, List.of(), onStart, out, r -> { });
    }

    /** onDone: 마지막 결과(done)를 화면에 보낸 뒤 부름 — 채팅 내역 저장 */
    public void askStream(AccessUser user, String query, int k, List<Turn> history, Runnable onStart, OutputStream out,
                          java.util.function.Consumer<AskResult> onDone) throws IOException {
        byte[] body = om.writeValueAsBytes(askBody(user, query, k, history));
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/internal/chat/ask/stream"))
                .timeout(Duration.ofMillis(chatTimeoutMs))  // 응답 머리까지 (AI 는 흐름을 바로 엶)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("X-Service-Key", serviceKey)
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();
        HttpResponse<InputStream> res;
        try {
            res = streamHttp.send(req, HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ApiException.badGateway("AI 서버에 연결할 수 없습니다");
        } catch (IOException e) {
            throw ApiException.badGateway("AI 서버에 연결할 수 없습니다");
        }
        try (InputStream in = res.body()) {
            if (res.statusCode() != 200) {
                throw error(res.statusCode(), detailOf(new String(in.readAllBytes(), StandardCharsets.UTF_8)));
            }
            onStart.run();
            relay(in, out, onDone);
        }
    }

    /**
     * SSE 를 줄 단위로 넘김. 이벤트 하나가 끝날 때마다(빈 줄) 바로 내보냄 (모아 두지 않음).
     * done 의 result 는 AskResult 로 읽어 다시 써서 /ask 응답과 같은 모양(camelCase)으로 — 나머지 이벤트는 그대로.
     */
    void relay(InputStream in, OutputStream out) throws IOException {
        relay(in, out, r -> { });
    }

    void relay(InputStream in, OutputStream out, java.util.function.Consumer<AskResult> onDone) throws IOException {
        var reader = new java.io.BufferedReader(new java.io.InputStreamReader(in, StandardCharsets.UTF_8));
        String event = null;
        AskResult done = null;
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("event:")) {
                event = line.substring(6).trim();
            } else if (line.startsWith("data:") && "done".equals(event)) {
                JsonNode result = om.readTree(line.substring(5)).get("result");
                done = om.treeToValue(result, AskResult.class);
                line = "data: " + om.writeValueAsString(Map.of("result", done));
            }
            out.write((line + "\n").getBytes(StandardCharsets.UTF_8));
            if (line.isEmpty()) {
                out.flush();
                event = null;
            }
        }
        out.flush();
        if (done != null) onDone.accept(done);  // 화면에 다 보낸 뒤 (저장이 늦어도 답은 먼저 감)
    }

    /** AI 오류 응답 → 다른 호출(call)과 같은 오류 */
    private ApiException error(int status, String detail) {
        if (status == 403) return ApiException.forbidden(detail != null ? detail : "권한이 없습니다.");
        if (status == 400) return ApiException.badRequest(detail != null ? detail : "요청이 올바르지 않습니다.");
        if (status == 422) return ApiException.badRequest("요청 형식이 맞지 않습니다.");
        if (status >= 500) {
            return ApiException.badGateway((detail != null ? detail : "AI 서버가 요청을 처리하지 못했습니다") + " 잠시 후 다시 시도해 주세요.");
        }
        return new ApiException(HttpStatus.BAD_GATEWAY, "AI 서버가 요청을 거절했습니다 (" + status + ")");
    }

    private String detailOf(String body) {
        try {
            var d = om.readTree(body).get("detail");
            if (d != null && d.isTextual()) return d.asText();
        } catch (Exception ignored) {
            // 기본 문구
        }
        return null;
    }

    /** 백엔드에서 거절한 실명 보기 시도(사유 없음 등)를 AI 쪽 복원 기록에 남김 — 이상 접근 탐지용.
     *  기록 실패는 무시 (거절 응답은 그대로) */
    public void logRefused(AccessUser user, String actor, List<String> docIds, String reason) {
        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("user", user);
            body.put("actor", actor);
            body.put("doc_ids", docIds == null ? List.of() : docIds);
            body.put("reason", reason);
            http.post().uri("/internal/audit/refused").contentType(MediaType.APPLICATION_JSON)
                    .body(body).retrieve().toBodilessEntity();
        } catch (Exception ignored) {
            // 기록만 실패
        }
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

    /** POST /internal{path}?query (body = 바이트 그대로, 상담 녹음 wav). AI 는 받자마자 job 만 만들고 202 */
    public JsonNode postBytes(String path, Map<String, ?> query, byte[] body) {
        try {
            return call(() -> http.post().uri(b -> {
                b.path("/internal" + path);
                query.forEach((k, v) -> { if (v != null) b.queryParam(k, v); });
                return b.build();
            }).contentType(MediaType.APPLICATION_OCTET_STREAM).body(body).retrieve().body(JsonNode.class));
        } catch (NotFound e) {
            throw ApiException.notFound("찾을 수 없습니다.");
        }
    }

    // ------------------------------------------------------------------

    private static class NotFound extends RuntimeException {}

    private String detail(org.springframework.web.client.HttpStatusCodeException e, String fallback) {
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
        } catch (org.springframework.web.client.HttpServerErrorException e) {
            // AI 서버까지는 닿았지만 처리 실패 (예: 502 "검색 또는 답변 생성에 실패했습니다") — 연결 문제로 보이지 않게 그 이유를 보여 줌
            throw ApiException.badGateway(detail(e, "AI 서버가 요청을 처리하지 못했습니다") + " 잠시 후 다시 시도해 주세요.");
        } catch (RestClientException e) {
            throw ApiException.badGateway("AI 서버에 연결할 수 없습니다");
        }
    }
}
