package com.saebom.backend.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

/** 상태 코드와 화면에 보여 줄 메시지를 같이 던짐 → GlobalExceptionHandler 가 {success:false, message} 로 */
@Getter
public class ApiException extends RuntimeException {

    private final HttpStatus status;

    public ApiException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public static ApiException badRequest(String m) { return new ApiException(HttpStatus.BAD_REQUEST, m); }
    public static ApiException notFound(String m) { return new ApiException(HttpStatus.NOT_FOUND, m); }
    public static ApiException conflict(String m) { return new ApiException(HttpStatus.CONFLICT, m); }
    public static ApiException forbidden(String m) { return new ApiException(HttpStatus.FORBIDDEN, m); }
    public static ApiException badGateway(String m) { return new ApiException(HttpStatus.BAD_GATEWAY, m); }
}
