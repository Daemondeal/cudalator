#![allow(dead_code)]
use crate::cir::Token;

pub enum FrontendErrorKind {
    Other,
    Todo,
    Unsupported,
}

pub struct FrontendError {
    pub token: Token,
    pub message: String,
    pub kind: FrontendErrorKind,
}

impl FrontendError {
    pub fn new(token: Token, message: String, kind: FrontendErrorKind) -> Self {
        Self {
            token,
            message,
            kind,
        }
    }

    pub fn todo(token: Token, message: String) -> Self {
        Self::new(token, message, FrontendErrorKind::Todo)
    }

    pub fn unsupported(token: Token, message: String) -> Self {
        Self::new(token, message, FrontendErrorKind::Unsupported)
    }

    pub fn other(token: Token, message: String) -> Self {
        Self::new(token, message, FrontendErrorKind::Other)
    }
} 
