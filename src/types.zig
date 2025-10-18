pub const NodeId = u32;

pub const Error = error{
    UnknownVariable,
    MissingInput,
    SymbolicGradientUnsupported,
    UnexpectedCharacter,
    UnexpectedToken,
    UnexpectedEndOfInput,
    ExpectedCommaOrRParen,
    ExpectedRightParen,
    UnknownFunction,
    ArityMismatch,
    UnknownCustomOp,
    OutOfMemory,
};
