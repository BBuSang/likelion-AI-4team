package com.saebom.backend.ai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonAlias;
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

    public AiClient(@Value("${ai.base-url}") String baseUrl, @Value("${ai.service-key}") String serviceKey) {
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(3_000);
        factory.setReadTimeout(30_000);
        this.http = RestClient.builder()
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

    // ------------------------------------------------------------------

    private static class NotFound extends RuntimeException {}

    private <T> T call(java.util.function.Supplier<T> f) {
        try {
            return f.get();
        } catch (HttpClientErrorException e) {
            if (e.getStatusCode() == HttpStatus.NOT_FOUND) throw new NotFound();
            // 400 같은 입력 문제는 그대로 알려 주되, AI 서버 내부 내용은 안 보냄
            throw new ApiException(HttpStatus.BAD_GATEWAY, "AI 서버가 요청을 거절했습니다 (" + e.getStatusCode().value() + ")");
        } catch (RestClientException e) {
            throw ApiException.badGateway("AI 서버에 연결할 수 없습니다");
        }
    }
}
