const std = @import("std");
const types = @import("types.zig");
const graph_mod = @import("graph.zig");
const registry_mod = @import("registry.zig");

pub const Error = types.Error;
pub const Graph = graph_mod.Graph;
pub const NodeId = types.NodeId;
pub const NodeKind = graph_mod.NodeKind;
pub const Registry = registry_mod.Registry;

pub const Parser = struct {
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: []const u8,
    pos: usize,
    graph: Graph,

    pub const ParseResult = struct {
        graph: Graph,
        output: NodeId,
    };

    pub fn init(allocator: std.mem.Allocator, registry: *Registry, source: []const u8) Parser {
        return .{
            .allocator = allocator,
            .registry = registry,
            .source = source,
            .pos = 0,
            .graph = Graph.init(allocator),
        };
    }

    pub fn deinit(self: *Parser) void {
        self.graph.deinit();
    }

    pub fn parse(self: *Parser) Error!ParseResult {
        const expr = try self.parseExpression();
        self.skipWhitespace();
        if (self.pos != self.source.len) return Error.UnexpectedToken;
        const result = ParseResult{ .graph = self.graph, .output = expr };
        self.graph = Graph.init(self.allocator);
        return result;
    }

    fn parseExpression(self: *Parser) Error!NodeId {
        return try self.parseAddSub();
    }

    fn parseAddSub(self: *Parser) Error!NodeId {
        var node: NodeId = try self.parseMulDiv();
        while (true) {
            self.skipWhitespace();
            if (self.peekChar()) |c| {
                if (c == '+') {
                    self.pos += 1;
                    const rhs = try self.parseMulDiv();
                    node = try self.graph.addBinary(.add, node, rhs);
                    continue;
                } else if (c == '-') {
                    self.pos += 1;
                    const rhs = try self.parseMulDiv();
                    node = try self.graph.addBinary(.sub, node, rhs);
                    continue;
                }
            }
            break;
        }
        return node;
    }

    fn parseMulDiv(self: *Parser) Error!NodeId {
        var node: NodeId = try self.parsePower();
        while (true) {
            self.skipWhitespace();
            if (self.peekChar()) |c| {
                if (c == '*') {
                    self.pos += 1;
                    const rhs = try self.parsePower();
                    node = try self.graph.addBinary(.mul, node, rhs);
                    continue;
                } else if (c == '/') {
                    self.pos += 1;
                    const rhs = try self.parsePower();
                    node = try self.graph.addBinary(.div, node, rhs);
                    continue;
                }
            }
            break;
        }
        return node;
    }

    fn parsePower(self: *Parser) Error!NodeId {
        var node: NodeId = try self.parseUnary();
        self.skipWhitespace();
        if (self.peekChar()) |c| {
            if (c == '^') {
                self.pos += 1;
                const rhs = try self.parseUnary();
                node = try self.graph.addBinary(.pow, node, rhs);
            }
        }
        return node;
    }

    fn parseUnary(self: *Parser) Error!NodeId {
        self.skipWhitespace();
        if (self.peekChar()) |c| {
            if (c == '+') {
                self.pos += 1;
                return self.parseUnary();
            } else if (c == '-') {
                self.pos += 1;
                const operand = try self.parseUnary();
                const zero = try self.graph.addConstant(0.0);
                return self.graph.addBinary(.sub, zero, operand);
            }
        }
        return self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) Error!NodeId {
        self.skipWhitespace();
        if (self.pos >= self.source.len) return Error.UnexpectedEndOfInput;
        const c = self.source[self.pos];
        if (std.ascii.isDigit(c) or c == '.') {
            return self.parseNumber();
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            return self.parseIdentifierOrCall();
        }
        if (c == '(') {
            self.pos += 1;
            const expr = try self.parseExpression();
            self.skipWhitespace();
            if (self.pos >= self.source.len or self.source[self.pos] != ')') return Error.ExpectedRightParen;
            self.pos += 1;
            return expr;
        }
        return Error.UnexpectedCharacter;
    }

    fn parseNumber(self: *Parser) Error!NodeId {
        const start = self.pos;
        while (self.pos < self.source.len and (std.ascii.isDigit(self.source[self.pos]) or self.source[self.pos] == '.')) {
            self.pos += 1;
        }
        const slice = self.source[start..self.pos];
        const value = std.fmt.parseFloat(f32, slice) catch {
            return Error.UnexpectedCharacter;
        };
        return self.graph.addConstant(value);
    }

    fn parseIdentifierOrCall(self: *Parser) Error!NodeId {
        const start = self.pos;
        while (self.pos < self.source.len and (std.ascii.isAlphanumeric(self.source[self.pos]) or self.source[self.pos] == '_')) {
            self.pos += 1;
        }
        const ident = self.source[start..self.pos];
        self.skipWhitespace();
        if (self.pos < self.source.len and self.source[self.pos] == '(') {
            self.pos += 1;
            var args: std.ArrayList(NodeId) = .empty;
            defer args.deinit(self.allocator);
            self.skipWhitespace();
            if (!(self.pos < self.source.len and self.source[self.pos] == ')')) {
                while (true) {
                    const arg = try self.parseExpression();
                    try args.append(self.allocator, arg);
                    self.skipWhitespace();
                    if (self.pos >= self.source.len) return Error.ExpectedRightParen;
                    if (self.source[self.pos] == ',') {
                        self.pos += 1;
                        continue;
                    } else if (self.source[self.pos] == ')') {
                        break;
                    } else {
                        return Error.ExpectedCommaOrRParen;
                    }
                }
            }
            if (self.pos >= self.source.len or self.source[self.pos] != ')') return Error.ExpectedRightParen;
            self.pos += 1;
            return self.createCall(ident, args.items);
        }
        return self.graph.input_lookup.get(ident) orelse try self.graph.addInput(ident);
    }

    fn createCall(self: *Parser, name: []const u8, args: []const NodeId) Error!NodeId {
        if (std.mem.eql(u8, name, "log")) {
            if (args.len != 1) return Error.ArityMismatch;
            return self.graph.addUnary(.log, args[0]);
        }
        if (std.mem.eql(u8, name, "exp")) {
            if (args.len != 1) return Error.ArityMismatch;
            return self.graph.addUnary(.exp, args[0]);
        }
        if (std.mem.eql(u8, name, "sin")) {
            if (args.len != 1) return Error.ArityMismatch;
            return self.graph.addUnary(.sin, args[0]);
        }
        if (std.mem.eql(u8, name, "cos")) {
            if (args.len != 1) return Error.ArityMismatch;
            return self.graph.addUnary(.cos, args[0]);
        }
        const op = self.registry.get(name) orelse return Error.UnknownFunction;
        return self.graph.addCustom(op, args);
    }

    fn skipWhitespace(self: *Parser) void {
        while (self.pos < self.source.len and std.ascii.isWhitespace(self.source[self.pos])) {
            self.pos += 1;
        }
    }

    fn peekChar(self: *Parser) ?u8 {
        if (self.pos >= self.source.len) return null;
        return self.source[self.pos];
    }
};
