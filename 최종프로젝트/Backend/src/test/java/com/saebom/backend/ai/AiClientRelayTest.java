package com.saebom.backend.ai;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;

/** 답변 스트리밍 중계: 단계·조각은 그대로, done 결과는 /ask 와 같은 camelCase 로 */
class AiClientRelayTest {

    @Test
    void relayKeepsEventsAndConvertsDoneToCamelCase() throws Exception {
        String sse = """
                event: stage
                data: {"stage": "classify"}

                event: delta
                data: {"text": "<PERSON_ab12cd34> 고객 건입니다. "}

                event: done
                data: {"result": {"masked_query": "q", "answer": "답 [H0001]", "ask_back": false, "docs": [{"doc_id": "H0001", "doc_type": "상담", "score": 0.9, "snippet": "글"}], "not_found": false, "check": {"unverified_tokens": [], "raw_pii": []}, "model": "m", "timing": {"search": 0.1}, "agent": null, "ambiguous_people": []}}

                """;
        var out = new ByteArrayOutputStream();
        new AiClient("http://localhost:1", "k", 1000)
                .relay(new ByteArrayInputStream(sse.getBytes(StandardCharsets.UTF_8)), out);
        String got = out.toString(StandardCharsets.UTF_8);

        assertThat(got).contains("event: stage\ndata: {\"stage\": \"classify\"}\n\n");
        assertThat(got).contains("data: {\"text\": \"<PERSON_ab12cd34> 고객 건입니다. \"}\n\n");
        String doneLine = got.lines().filter(l -> l.startsWith("data: {\"result\"")).findFirst().orElseThrow();
        var result = new ObjectMapper().readTree(doneLine.substring(5)).get("result");
        assertThat(result.get("maskedQuery").asText()).isEqualTo("q");
        assertThat(result.get("askBack").asBoolean()).isFalse();
        assertThat(result.get("docs").get(0).get("docId").asText()).isEqualTo("H0001");
        assertThat(result.has("ask_back")).isFalse();
    }
}
