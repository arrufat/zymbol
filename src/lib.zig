const std = @import("std");

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

pub const Operation = struct {
    name: []const u8,
    arity: u8,
    forward: *const fn (args: []const f32) f32,
    backward: *const fn (grad_output: f32, args: []const f32, arg_index: usize) f32,
    symbolic: ?*const fn (ctx: *SymbolicContext, info: SymbolicOpInfo) Error!void = null,
    printer: ?*const fn (allocator: std.mem.Allocator, args: []const []const u8) Error![]u8 = null,
};

pub const SymbolicOpInfo = struct {
    node_id: NodeId,
    inputs: []const NodeId,
    upstream: NodeId,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    ops: std.StringHashMapUnmanaged(Operation),

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .ops = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        var it = self.ops.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.ops.deinit(self.allocator);
    }

    pub fn register(self: *Registry, op: Operation) !void {
        const name_copy = try self.allocator.dupe(u8, op.name);
        errdefer self.allocator.free(name_copy);

        try self.ops.put(self.allocator, name_copy, op);
    }

    pub fn get(self: *Registry, name: []const u8) ?*const Operation {
        return self.ops.getPtr(name);
    }

    pub fn registerBuiltins(self: *Registry) !void {
        try self.register(relu_op);
        try self.register(relu_grad_op);
        try self.register(sigmoid_op);
        try self.register(max_op);
        try self.register(log_guard_op);
    }
};

const NodeKind = enum {
    input,
    constant,
    add,
    sub,
    mul,
    div,
    pow,
    log,
    exp,
    sin,
    cos,
    custom,
};

const BinaryInputs = struct { lhs: NodeId, rhs: NodeId };

const Node = struct {
    kind: NodeKind,
    payload: Payload,

    const Payload = union(enum) {
        none,
        input: []const u8,
        constant: f32,
        unary: NodeId,
        binary: BinaryInputs,
        custom: struct { op: *const Operation, inputs: []NodeId },
    };
};

const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    input_lookup: std.StringHashMap(NodeId),
    interned_names: std.ArrayList([]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{
            .allocator = allocator,
            .nodes = .empty,
            .input_lookup = std.StringHashMap(NodeId).init(allocator),
            .interned_names = .empty,
        };
    }

    pub fn deinit(self: *Graph) void {
        const allocator = self.allocator;
        var idx: usize = 0;
        while (idx < self.nodes.items.len) : (idx += 1) {
            const n = self.nodes.items[idx];
            switch (n.payload) {
                .custom => |custom| allocator.free(custom.inputs),
                else => {},
            }
        }

        for (self.interned_names.items) |name| allocator.free(name);
        self.interned_names.deinit(allocator);
        self.nodes.deinit(allocator);
        self.input_lookup.deinit();
    }

    fn appendNode(self: *Graph, new_node: Node) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        try self.nodes.append(self.allocator, new_node);
        return id;
    }

    pub fn addInput(self: *Graph, name: []const u8) !NodeId {
        const copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(copy);

        try self.interned_names.append(self.allocator, copy);
        const id = try self.appendNode(.{
            .kind = .input,
            .payload = .{ .input = copy },
        });
        try self.input_lookup.put(copy, id);
        return id;
    }

    pub fn addConstant(self: *Graph, value: f32) !NodeId {
        return try self.appendNode(.{
            .kind = .constant,
            .payload = .{ .constant = value },
        });
    }

    pub fn addUnary(self: *Graph, kind: NodeKind, operand: NodeId) !NodeId {
        return try self.appendNode(.{
            .kind = kind,
            .payload = .{ .unary = operand },
        });
    }

    pub fn addBinary(self: *Graph, kind: NodeKind, lhs: NodeId, rhs: NodeId) !NodeId {
        return try self.appendNode(.{
            .kind = kind,
            .payload = .{ .binary = .{ .lhs = lhs, .rhs = rhs } },
        });
    }

    pub fn addCustom(self: *Graph, op: *const Operation, operands: []const NodeId) !NodeId {
        if (operands.len != op.arity) return Error.ArityMismatch;
        const dup = try self.allocator.dupe(NodeId, operands);
        errdefer self.allocator.free(dup);
        return try self.appendNode(.{
            .kind = .custom,
            .payload = .{ .custom = .{ .op = op, .inputs = dup } },
        });
    }

    fn node(self: *const Graph, id: NodeId) Node {
        return self.nodes.items[@as(usize, @intCast(id))];
    }

    fn copyNode(self: *Graph, source: *const Graph, id: NodeId, map: *std.AutoHashMap(NodeId, NodeId)) !NodeId {
        if (map.get(id)) |existing| return existing;
        const orig_node = source.node(id);
        const new_id = switch (orig_node.kind) {
            .input => try self.addInput(orig_node.payload.input),
            .constant => try self.addConstant(orig_node.payload.constant),
            .add, .sub, .mul, .div, .pow => blk: {
                const binary = orig_node.payload.binary;
                const lhs = try self.copyNode(source, binary.lhs, map);
                const rhs = try self.copyNode(source, binary.rhs, map);
                break :blk try self.addBinary(orig_node.kind, lhs, rhs);
            },
            .log, .exp, .sin, .cos => blk: {
                const operand = try self.copyNode(source, orig_node.payload.unary, map);
                break :blk try self.addUnary(orig_node.kind, operand);
            },
            .custom => blk: {
                const custom = orig_node.payload.custom;
                var buffer = try self.allocator.alloc(NodeId, custom.inputs.len);
                defer self.allocator.free(buffer);
                for (custom.inputs, 0..) |child, idx| {
                    buffer[idx] = try self.copyNode(source, child, map);
                }
                break :blk try self.addCustom(custom.op, buffer);
            },
        };
        try map.put(id, new_id);
        return new_id;
    }
};

pub const Expression = struct {
    allocator: std.mem.Allocator,
    registry: *Registry,
    graph: Graph,
    output: NodeId,

    pub fn parse(allocator: std.mem.Allocator, registry: *Registry, source: []const u8) Error!Expression {
        var parser: Parser = .init(allocator, registry, source);
        defer parser.deinit();
        const parsed = try parser.parse();
        return .{
            .allocator = allocator,
            .registry = registry,
            .graph = parsed.graph,
            .output = parsed.output,
        };
    }

    pub fn fromGraph(allocator: std.mem.Allocator, registry: *Registry, graph: Graph, output: NodeId) Expression {
        return .{
            .allocator = allocator,
            .registry = registry,
            .graph = graph,
            .output = output,
        };
    }

    pub fn deinit(self: *Expression) void {
        self.graph.deinit();
    }

    pub fn evaluate(self: *const Expression, inputs: std.StringHashMap(f32)) Error!f32 {
        var values = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        defer self.allocator.free(values);

        for (self.graph.nodes.items, 0..) |node, idx| {
            values[idx] = try evaluateNode(self, node, values, inputs);
        }
        return values[@intCast(self.output)];
    }

    pub fn symbolicGradient(self: *const Expression, variable: []const u8) Error!Expression {
        return SymbolicBuilder.build(self, variable);
    }

    pub fn numericGradient(self: *const Expression, variable: []const u8, inputs: std.StringHashMap(f32)) Error!f32 {
        var values = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        defer self.allocator.free(values);
        for (self.graph.nodes.items, 0..) |node, idx| {
            values[idx] = try evaluateNode(self, node, values, inputs);
        }
        var adjoints = try self.graphBackward(values);
        defer self.allocator.free(adjoints);

        const id = self.graph.input_lookup.get(variable) orelse return Error.UnknownVariable;
        return adjoints[@intCast(id)];
    }

    fn graphBackward(self: *const Expression, values: []const f32) Error![]f32 {
        var adjoints = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        @memset(adjoints, 0.0);
        adjoints[@intCast(self.output)] = 1.0;

        var idx = self.graph.nodes.items.len;
        while (idx > 0) {
            idx -= 1;
            const node = self.graph.nodes.items[idx];
            const grad = adjoints[idx];
            switch (node.kind) {
                .input, .constant => {},
                .add => {
                    const binary = node.payload.binary;
                    adjoints[@intCast(binary.lhs)] += grad;
                    adjoints[@intCast(binary.rhs)] += grad;
                },
                .sub => {
                    const binary = node.payload.binary;
                    adjoints[@intCast(binary.lhs)] += grad;
                    adjoints[@intCast(binary.rhs)] -= grad;
                },
                .mul => {
                    const binary = node.payload.binary;
                    adjoints[@intCast(binary.lhs)] += grad * values[@intCast(binary.rhs)];
                    adjoints[@intCast(binary.rhs)] += grad * values[@intCast(binary.lhs)];
                },
                .div => {
                    const binary = node.payload.binary;
                    const lhs_val = values[@intCast(binary.lhs)];
                    const rhs_val = values[@intCast(binary.rhs)];
                    adjoints[@intCast(binary.lhs)] += grad / rhs_val;
                    adjoints[@intCast(binary.rhs)] -= grad * lhs_val / (rhs_val * rhs_val);
                },
                .pow => {
                    const binary = node.payload.binary;
                    const base = values[@intCast(binary.lhs)];
                    const exponent = values[@intCast(binary.rhs)];
                    const result = values[idx];
                    adjoints[@intCast(binary.lhs)] += grad * exponent * std.math.pow(f32, base, exponent - 1.0);
                    if (base > 0.0) {
                        adjoints[@intCast(binary.rhs)] += grad * result * @log(base);
                    }
                },
                .log => {
                    const operand = node.payload.unary;
                    adjoints[@intCast(operand)] += grad / values[@intCast(operand)];
                },
                .exp => {
                    const operand = node.payload.unary;
                    adjoints[@intCast(operand)] += grad * values[idx];
                },
                .sin => {
                    const operand = node.payload.unary;
                    adjoints[@intCast(operand)] += grad * @cos(values[@intCast(operand)]);
                },
                .cos => {
                    const operand = node.payload.unary;
                    adjoints[@intCast(operand)] -= grad * @sin(values[@intCast(operand)]);
                },
                .custom => {
                    const custom = node.payload.custom;
                    var args = try self.allocator.alloc(f32, custom.inputs.len);
                    defer self.allocator.free(args);
                    for (custom.inputs, 0..) |child, arg_idx| {
                        args[arg_idx] = values[@intCast(child)];
                    }
                    for (custom.inputs, 0..) |child, arg_idx| {
                        adjoints[@intCast(child)] += custom.op.backward(grad, args, arg_idx);
                    }
                },
            }
        }
        return adjoints;
    }

    pub fn toString(self: *const Expression) Error![]u8 {
        var interner = std.AutoHashMap(NodeId, []const u8).init(self.allocator);
        defer {
            var it = interner.iterator();
            while (it.next()) |entry| self.allocator.free(entry.value_ptr.*);
            interner.deinit();
        }
        return renderNode(self, self.output, &interner);
    }
};

fn evaluateNode(expr: *const Expression, node: Node, values: []f32, inputs: std.StringHashMap(f32)) Error!f32 {
    return switch (node.kind) {
        .input => inputs.get(node.payload.input) orelse return Error.MissingInput,
        .constant => node.payload.constant,
        .add => blk: {
            const ids = node.payload.binary;
            break :blk values[@intCast(ids.lhs)] + values[@intCast(ids.rhs)];
        },
        .sub => blk: {
            const ids = node.payload.binary;
            break :blk values[@intCast(ids.lhs)] - values[@intCast(ids.rhs)];
        },
        .mul => blk: {
            const ids = node.payload.binary;
            break :blk values[@intCast(ids.lhs)] * values[@intCast(ids.rhs)];
        },
        .div => blk: {
            const ids = node.payload.binary;
            break :blk values[@intCast(ids.lhs)] / values[@intCast(ids.rhs)];
        },
        .pow => blk: {
            const ids = node.payload.binary;
            break :blk std.math.pow(f32, values[@intCast(ids.lhs)], values[@intCast(ids.rhs)]);
        },
        .log => @log(values[@intCast(node.payload.unary)]),
        .exp => @exp(values[@intCast(node.payload.unary)]),
        .sin => @sin(values[@intCast(node.payload.unary)]),
        .cos => @cos(values[@intCast(node.payload.unary)]),
        .custom => blk: {
            const custom = node.payload.custom;
            var args = try expr.allocator.alloc(f32, custom.inputs.len);
            defer expr.allocator.free(args);
            for (custom.inputs, 0..) |child, arg_idx| {
                args[arg_idx] = values[@intCast(child)];
            }
            break :blk custom.op.forward(args);
        },
    };
}

fn renderNode(expr: *const Expression, id: NodeId, cache: *std.AutoHashMap(NodeId, []const u8)) Error![]u8 {
    if (cache.get(id)) |existing| return expr.allocator.dupe(u8, existing);
    const node = expr.graph.node(id);
    const result = switch (node.kind) {
        .input => try expr.allocator.dupe(u8, node.payload.input),
        .constant => try std.fmt.allocPrint(expr.allocator, "{d}", .{node.payload.constant}),
        .add => try renderBinary(expr, cache, node.payload.binary, " + "),
        .sub => try renderBinary(expr, cache, node.payload.binary, " - "),
        .mul => try renderBinary(expr, cache, node.payload.binary, " * "),
        .div => try renderBinary(expr, cache, node.payload.binary, " / "),
        .pow => try renderBinary(expr, cache, node.payload.binary, " ^ "),
        .log => try renderCall(expr, cache, "log", node.payload.unary),
        .exp => try renderCall(expr, cache, "exp", node.payload.unary),
        .sin => try renderCall(expr, cache, "sin", node.payload.unary),
        .cos => try renderCall(expr, cache, "cos", node.payload.unary),
        .custom => blk: {
            const custom = node.payload.custom;
            var args = try expr.allocator.alloc([]const u8, custom.inputs.len);
            defer {
                for (args) |s| expr.allocator.free(s);
                expr.allocator.free(args);
            }
            for (custom.inputs, 0..) |child, idx| {
                args[idx] = try renderNode(expr, child, cache);
            }
            if (custom.op.printer) |printer| {
                break :blk try printer(expr.allocator, args);
            } else {
                const joined = try std.mem.join(expr.allocator, ", ", args);
                defer expr.allocator.free(joined);
                break :blk try std.fmt.allocPrint(expr.allocator, "{s}({s})", .{ custom.op.name, joined });
            }
        },
    };
    try cache.put(id, result);
    return expr.allocator.dupe(u8, result);
}

fn renderBinary(expr: *const Expression, cache: *std.AutoHashMap(NodeId, []const u8), binaries: BinaryInputs, op: []const u8) Error![]u8 {
    const left = try renderNode(expr, binaries.lhs, cache);
    defer expr.allocator.free(left);
    const right = try renderNode(expr, binaries.rhs, cache);
    defer expr.allocator.free(right);
    return std.fmt.allocPrint(expr.allocator, "({s}{s}{s})", .{ left, op, right });
}

fn renderCall(expr: *const Expression, cache: *std.AutoHashMap(NodeId, []const u8), name: []const u8, operand: NodeId) Error![]u8 {
    const inner = try renderNode(expr, operand, cache);
    defer expr.allocator.free(inner);
    return std.fmt.allocPrint(expr.allocator, "{s}({s})", .{ name, inner });
}

const SymbolicContext = struct {
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: *const Graph,
    grad_graph: *Graph,
    cache: std.AutoHashMap(NodeId, NodeId),
    gradients: *std.AutoHashMap(NodeId, NodeId),

    fn init(allocator: std.mem.Allocator, registry: *Registry, source: *const Graph, grad_graph: *Graph, gradients: *std.AutoHashMap(NodeId, NodeId)) SymbolicContext {
        return .{
            .allocator = allocator,
            .registry = registry,
            .source = source,
            .grad_graph = grad_graph,
            .cache = std.AutoHashMap(NodeId, NodeId).init(allocator),
            .gradients = gradients,
        };
    }

    fn deinit(self: *SymbolicContext) void {
        self.cache.deinit();
    }

    fn ensureNode(self: *SymbolicContext, id: NodeId) Error!NodeId {
        if (self.cache.get(id)) |existing| return existing;
        const new_id = try self.grad_graph.copyNode(self.source, id, &self.cache);
        return new_id;
    }

    fn constant(self: *SymbolicContext, value: f32) Error!NodeId {
        return self.grad_graph.addConstant(value);
    }

    fn unary(self: *SymbolicContext, kind: NodeKind, operand: NodeId) Error!NodeId {
        return self.grad_graph.addUnary(kind, operand);
    }

    fn binary(self: *SymbolicContext, kind: NodeKind, lhs: NodeId, rhs: NodeId) Error!NodeId {
        return self.grad_graph.addBinary(kind, lhs, rhs);
    }

    fn callCustom(self: *SymbolicContext, name: []const u8, operands: []const NodeId) Error!NodeId {
        const op = self.registry.get(name) orelse return Error.UnknownCustomOp;
        return self.grad_graph.addCustom(op, operands);
    }

    fn accumulate(self: *SymbolicContext, target: NodeId, grad_id: NodeId) Error!void {
        if (self.gradients.getPtr(target)) |existing| {
            const combined = try self.binary(.add, existing.*, grad_id);
            existing.* = combined;
        } else {
            try self.gradients.put(target, grad_id);
        }
    }
};

const SymbolicBuilder = struct {
    pub fn build(expr: *const Expression, variable: []const u8) Error!Expression {
        const allocator = expr.allocator;
        var grad_graph = Graph.init(allocator);
        errdefer grad_graph.deinit();

        var gradients = std.AutoHashMap(NodeId, NodeId).init(allocator);
        defer gradients.deinit();

        var ctx = SymbolicContext.init(allocator, expr.registry, &expr.graph, &grad_graph, &gradients);
        defer ctx.deinit();

        const input_id = expr.graph.input_lookup.get(variable) orelse return Error.UnknownVariable;

        const one = try grad_graph.addConstant(1.0);
        try gradients.put(expr.output, one);

        var idx = expr.graph.nodes.items.len;
        while (idx > 0) {
            idx -= 1;
            const node_id = @as(NodeId, @intCast(idx));
            const node = expr.graph.nodes.items[idx];
            const upstream = gradients.get(node_id) orelse continue;
            try propagate(&ctx, node_id, node, upstream);
        }

        const grad_output = gradients.get(input_id) orelse try grad_graph.addConstant(0.0);
        const output = grad_output;
        return Expression.fromGraph(allocator, expr.registry, grad_graph, output);
    }

    fn propagate(ctx: *SymbolicContext, node_id: NodeId, node: Node, upstream: NodeId) Error!void {
        switch (node.kind) {
            .input, .constant => return,
            .add => {
                const binary = node.payload.binary;
                try ctx.accumulate(binary.lhs, upstream);
                try ctx.accumulate(binary.rhs, upstream);
            },
            .sub => {
                const binary = node.payload.binary;
                try ctx.accumulate(binary.lhs, upstream);
                const neg_one = try ctx.constant(-1.0);
                const neg_grad = try ctx.binary(.mul, upstream, neg_one);
                try ctx.accumulate(binary.rhs, neg_grad);
            },
            .mul => {
                const binary = node.payload.binary;
                const lhs = try ctx.ensureNode(binary.lhs);
                const rhs = try ctx.ensureNode(binary.rhs);
                const grad_lhs = try ctx.binary(.mul, upstream, rhs);
                const grad_rhs = try ctx.binary(.mul, upstream, lhs);
                try ctx.accumulate(binary.lhs, grad_lhs);
                try ctx.accumulate(binary.rhs, grad_rhs);
            },
            .div => {
                const binary = node.payload.binary;
                const lhs = try ctx.ensureNode(binary.lhs);
                const rhs = try ctx.ensureNode(binary.rhs);
                const grad_lhs = try ctx.binary(.div, upstream, rhs);
                try ctx.accumulate(binary.lhs, grad_lhs);
                const neg_one = try ctx.constant(-1.0);
                const div_rhs = try ctx.binary(.div, lhs, try ctx.binary(.mul, rhs, rhs));
                const grad_rhs = try ctx.binary(.mul, upstream, try ctx.binary(.mul, neg_one, div_rhs));
                try ctx.accumulate(binary.rhs, grad_rhs);
            },
            .pow => {
                const binary = node.payload.binary;
                const base = try ctx.ensureNode(binary.lhs);
                const exponent = try ctx.ensureNode(binary.rhs);
                const one = try ctx.constant(1.0);
                const exponent_minus_one = try ctx.binary(.sub, exponent, one);
                const base_pow = try ctx.binary(.pow, base, exponent_minus_one);
                const grad_base = try ctx.binary(.mul, upstream, try ctx.binary(.mul, exponent, base_pow));
                try ctx.accumulate(binary.lhs, grad_base);

                const full_pow = try ctx.binary(.pow, base, exponent);
                const log_guard = try ctx.callCustom("log_guard", &.{base});
                const grad_exp = try ctx.binary(.mul, upstream, try ctx.binary(.mul, full_pow, log_guard));
                try ctx.accumulate(binary.rhs, grad_exp);
            },
            .log => {
                const operand = node.payload.unary;
                const copy = try ctx.ensureNode(operand);
                const one = try ctx.constant(1.0);
                const reciprocal = try ctx.binary(.div, one, copy);
                const grad_val = try ctx.binary(.mul, upstream, reciprocal);
                try ctx.accumulate(operand, grad_val);
            },
            .exp => {
                const operand = node.payload.unary;
                const copy = try ctx.ensureNode(node_id);
                const grad_val = try ctx.binary(.mul, upstream, copy);
                try ctx.accumulate(operand, grad_val);
            },
            .sin => {
                const operand = node.payload.unary;
                const copy = try ctx.ensureNode(operand);
                const cos_val = try ctx.unary(.cos, copy);
                const grad_val = try ctx.binary(.mul, upstream, cos_val);
                try ctx.accumulate(operand, grad_val);
            },
            .cos => {
                const operand = node.payload.unary;
                const copy = try ctx.ensureNode(operand);
                const sin_val = try ctx.unary(.sin, copy);
                const neg_one = try ctx.constant(-1.0);
                const grad_val = try ctx.binary(.mul, upstream, try ctx.binary(.mul, neg_one, sin_val));
                try ctx.accumulate(operand, grad_val);
            },
            .custom => {
                const custom = node.payload.custom;
                if (custom.op.symbolic) |rule| {
                    try rule(ctx, .{ .node_id = node_id, .inputs = custom.inputs, .upstream = upstream });
                } else {
                    return Error.SymbolicGradientUnsupported;
                }
            },
        }
    }
};

const Parser = struct {
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: []const u8,
    pos: usize,
    graph: Graph,

    fn init(allocator: std.mem.Allocator, registry: *Registry, source: []const u8) Parser {
        return .{
            .allocator = allocator,
            .registry = registry,
            .source = source,
            .pos = 0,
            .graph = Graph.init(allocator),
        };
    }

    fn deinit(self: *Parser) void {
        self.graph.deinit();
    }

    const ParseResult = struct {
        graph: Graph,
        output: NodeId,
    };

    fn parse(self: *Parser) Error!ParseResult {
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
        var node = try self.parseMulDiv();
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
        var node = try self.parsePower();
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
        var node = try self.parseUnary();
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

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn sigmoid_symbolic(ctx: *SymbolicContext, info: SymbolicOpInfo) Error!void {
    const input = info.inputs[0];
    const x = try ctx.ensureNode(input);
    const sig = try ctx.callCustom("sigmoid", &.{x});
    const one = try ctx.constant(1.0);
    const one_minus = try ctx.binary(.sub, one, sig);
    const local_grad = try ctx.binary(.mul, sig, one_minus);
    const grad = try ctx.binary(.mul, info.upstream, local_grad);
    try ctx.accumulate(input, grad);
}

fn relu_symbolic(ctx: *SymbolicContext, info: SymbolicOpInfo) Error!void {
    const input = info.inputs[0];
    const x = try ctx.ensureNode(input);
    const grad_expr = try ctx.callCustom("relu_grad", &.{x});
    const grad = try ctx.binary(.mul, info.upstream, grad_expr);
    try ctx.accumulate(input, grad);
}

const relu_op = Operation{
    .name = "relu",
    .arity = 1,
    .forward = struct {
        fn f(args: []const f32) f32 {
            return @max(0.0, args[0]);
        }
    }.f,
    .backward = struct {
        fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
            _ = arg_index;
            return if (args[0] > 0.0) grad_output else 0.0;
        }
    }.b,
    .symbolic = relu_symbolic,
    .printer = null,
};

const relu_grad_op = Operation{
    .name = "relu_grad",
    .arity = 1,
    .forward = struct {
        fn f(args: []const f32) f32 {
            return if (args[0] > 0.0) 1.0 else 0.0;
        }
    }.f,
    .backward = struct {
        fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
            _ = grad_output;
            _ = args;
            _ = arg_index;
            return 0.0;
        }
    }.b,
    .symbolic = null,
    .printer = null,
};

const sigmoid_op = Operation{
    .name = "sigmoid",
    .arity = 1,
    .forward = struct {
        fn f(args: []const f32) f32 {
            return sigmoid(args[0]);
        }
    }.f,
    .backward = struct {
        fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
            _ = arg_index;
            const s = sigmoid(args[0]);
            return grad_output * s * (1.0 - s);
        }
    }.b,
    .symbolic = sigmoid_symbolic,
    .printer = null,
};

const max_op = Operation{
    .name = "max",
    .arity = 2,
    .forward = struct {
        fn f(args: []const f32) f32 {
            return @max(args[0], args[1]);
        }
    }.f,
    .backward = struct {
        fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
            if (arg_index == 0) {
                return if (args[0] >= args[1]) grad_output else 0.0;
            } else {
                return if (args[1] > args[0]) grad_output else 0.0;
            }
        }
    }.b,
    .symbolic = null,
    .printer = null,
};

const log_guard_op = Operation{
    .name = "log_guard",
    .arity = 1,
    .forward = struct {
        fn f(args: []const f32) f32 {
            return if (args[0] > 0) @log(args[0]) else 0.0;
        }
    }.f,
    .backward = struct {
        fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
            _ = arg_index;
            if (args[0] > 0) {
                return grad_output / args[0];
            } else {
                return 0.0;
            }
        }
    }.b,
    .symbolic = null,
    .printer = null,
};

test "basic arithmetic" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    var expr = try Expression.parse(allocator, &registry, "x + y * 2");
    defer expr.deinit();

    var inputs = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 1.0);
    try inputs.put("y", 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try expr.evaluate(inputs), 0.0001);
}

test "symbolic gradient" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    var expr = try Expression.parse(allocator, &registry, "x ^ 2");
    defer expr.deinit();

    var grad_expr = try expr.symbolicGradient("x");
    defer grad_expr.deinit();

    var inputs = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), try grad_expr.evaluate(inputs), 0.0001);
}

test "numeric gradient matches" {
    const allocator = std.testing.allocator;
    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    var expr = try Expression.parse(allocator, &registry, "relu(x) * sigmoid(x)");
    defer expr.deinit();

    var inputs = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 2.0);

    const numeric = try expr.numericGradient("x", inputs);
    var grad_expr = try expr.symbolicGradient("x");
    defer grad_expr.deinit();
    const symbolic = try grad_expr.evaluate(inputs);
    try std.testing.expectApproxEqAbs(symbolic, numeric, 0.0001);
}
