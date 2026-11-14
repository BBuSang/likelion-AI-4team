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
}
