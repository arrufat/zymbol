const std = @import("std");
const types = @import("types.zig");
const operation = @import("operation.zig");
const registry_mod = @import("registry.zig");
const graph_mod = @import("graph.zig");

pub const Error = types.Error;
pub const NodeId = types.NodeId;
pub const Graph = graph_mod.Graph;
pub const Node = graph_mod.Node;
pub const NodeKind = graph_mod.NodeKind;
pub const Operation = operation.Operation;
pub const Registry = registry_mod.Registry;
pub const SymbolicOpInfo = operation.SymbolicOpInfo;
const SymbolicRuleFn = operation.SymbolicRuleFn;

pub const BuildResult = struct {
    graph: Graph,
    output: NodeId,
};

pub fn buildSymbolicGradient(
    expr_graph: *const Graph,
    expr_output: NodeId,
    variable: []const u8,
    allocator: std.mem.Allocator,
    registry: *Registry,
) Error!BuildResult {
    var grad_graph: Graph = Graph.init(allocator);
    var needs_deinit = true;
    defer if (needs_deinit) grad_graph.deinit();

    var gradients = std.AutoHashMap(NodeId, NodeId).init(allocator);
    defer gradients.deinit();

    var ctx = SymbolicContext.init(allocator, registry, expr_graph, &grad_graph, &gradients);
    defer ctx.deinit();

    const input_id = expr_graph.input_lookup.get(variable) orelse return Error.UnknownVariable;

    const one = try grad_graph.addConstant(1.0);
    try gradients.put(expr_output, one);

    var idx: usize = expr_graph.nodes.items.len;
    while (idx > 0) {
        idx -= 1;
        const node_id = @as(NodeId, @intCast(idx));
        const node = expr_graph.nodes.items[idx];
        const upstream = gradients.get(node_id) orelse continue;
        try propagate(&ctx, node_id, node, upstream);
    }

    const grad_output = gradients.get(input_id) orelse try grad_graph.addConstant(0.0);
    const output = grad_output;

    needs_deinit = false;
    return .{ .graph = grad_graph, .output = output };
}

pub fn registerBuiltins(registry: *Registry) !void {
    try registry.register(relu_op);
    try registry.register(relu_grad_op);
    try registry.register(sigmoid_op);
    try registry.register(max_op);
    try registry.register(log_guard_op);
}

pub const SymbolicContext = struct {
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
                const ctx_any: *anyopaque = @ptrCast(ctx);
                try rule(ctx_any, .{ .node_id = node_id, .inputs = custom.inputs, .upstream = upstream });
            } else {
                return Error.SymbolicGradientUnsupported;
            }
        },
    }
}

fn relu_symbolic(ctx: *SymbolicContext, info: SymbolicOpInfo) Error!void {
    const input = info.inputs[0];
    const x = try ctx.ensureNode(input);
    const grad_expr = try ctx.callCustom("relu_grad", &.{x});
    const grad = try ctx.binary(.mul, info.upstream, grad_expr);
    try ctx.accumulate(input, grad);
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

fn relu_symbolic_dispatch(ctx_ptr: *anyopaque, info: SymbolicOpInfo) Error!void {
    const ctx_unaligned: *align(1) SymbolicContext = @ptrCast(ctx_ptr);
    const ctx: *SymbolicContext = @alignCast(ctx_unaligned);
    return relu_symbolic(ctx, info);
}

fn sigmoid_symbolic_dispatch(ctx_ptr: *anyopaque, info: SymbolicOpInfo) Error!void {
    const ctx_unaligned: *align(1) SymbolicContext = @ptrCast(ctx_ptr);
    const ctx: *SymbolicContext = @alignCast(ctx_unaligned);
    return sigmoid_symbolic(ctx, info);
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
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
    .symbolic = relu_symbolic_dispatch,
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
    .symbolic = sigmoid_symbolic_dispatch,
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
