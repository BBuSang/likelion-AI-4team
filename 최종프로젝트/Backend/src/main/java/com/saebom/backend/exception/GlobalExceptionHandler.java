package com.saebom.backend.exception;

import com.saebom.backend.dto.AuthDtos.ApiResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiResponse> api(ApiException e) {
        return ResponseEntity.status(e.getStatus()).body(new ApiResponse(false, e.getMessage()));
    }

    /** 파일이 spring.servlet.multipart 한도(상담 녹음 40MB)를 넘음 — 기본 처리는 500 으로 보여서 */
    @ExceptionHandler(org.springframework.web.multipart.MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiResponse> tooBig(org.springframework.web.multipart.MaxUploadSizeExceededException e) {
        return ResponseEntity.status(org.springframework.http.HttpStatus.PAYLOAD_TOO_LARGE)
                .body(new ApiResponse(false, "파일이 너무 커요. 녹음은 40MB(약 20분)까지 올릴 수 있어요."));
    }

    /** DB 제약 위반: SQL·제약 이름이 화면에 그대로 나가던 것 막음. 자세한 건 서버 로그에만 */
    @ExceptionHandler(org.springframework.dao.DataIntegrityViolationException.class)
    public ResponseEntity<ApiResponse> integrity(org.springframework.dao.DataIntegrityViolationException e) {
        org.slf4j.LoggerFactory.getLogger(GlobalExceptionHandler.class).warn("DB 제약 위반: {}", e.getMostSpecificCause().getMessage());
        return ResponseEntity.status(org.springframework.http.HttpStatus.CONFLICT)
                .body(new ApiResponse(false, "다른 계정과 겹치는 값이 있어 저장하지 못했어요. 새로고침한 뒤 다시 시도해 주세요."));
    }
}
