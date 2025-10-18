const std = @import("std");
const types = @import("types.zig");
const graph_mod = @import("graph.zig");
const parser_mod = @import("parser.zig");
const registry_mod = @import("registry.zig");
const symbolic = @import("symbolic.zig");
const simplifier = @import("simplifier.zig");

pub const Error = types.Error;
pub const NodeId = types.NodeId;
pub const Graph = graph_mod.Graph;
pub const Node = graph_mod.Node;
pub const NodeKind = graph_mod.NodeKind;
pub const Registry = registry_mod.Registry;
pub const Parser = parser_mod.Parser;

pub const Expression = struct {
    allocator: std.mem.Allocator,
    registry: *Registry,
    graph: Graph,
    output: NodeId,

    pub fn parse(allocator: std.mem.Allocator, registry: *Registry, source: []const u8) Error!Expression {
        var parser = Parser.init(allocator, registry, source);
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
        var values: []f32 = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        defer self.allocator.free(values);

        for (self.graph.nodes.items, 0..) |node, idx| {
            values[idx] = try evaluateNode(self, node, values, inputs);
        }
        return values[@intCast(self.output)];
    }

    pub fn symbolicGradientRaw(self: *const Expression, variable: []const u8) Error!Expression {
        const result = try symbolic.buildSymbolicGradient(&self.graph, self.output, variable, self.allocator, self.registry);
        return Expression.fromGraph(self.allocator, self.registry, result.graph, result.output);
    }

    pub fn symbolicGradient(self: *const Expression, variable: []const u8) Error!Expression {
        var raw = try self.symbolicGradientRaw(variable);
        defer raw.deinit();
        return raw.simplify();
    }

    pub fn numericGradient(self: *const Expression, variable: []const u8, inputs: std.StringHashMap(f32)) Error!f32 {
        var values: []f32 = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        defer self.allocator.free(values);
        for (self.graph.nodes.items, 0..) |node, idx| {
            values[idx] = try evaluateNode(self, node, values, inputs);
        }
        var adjoints: []f32 = try self.graphBackward(values);
        defer self.allocator.free(adjoints);

        const id = self.graph.input_lookup.get(variable) orelse return Error.UnknownVariable;
        return adjoints[@intCast(id)];
    }

    pub fn toString(self: *const Expression) Error![]u8 {
        var interner: std.AutoHashMap(NodeId, []const u8) = std.AutoHashMap(NodeId, []const u8).init(self.allocator);
        defer {
            var it: std.AutoHashMap(NodeId, []const u8).Iterator = interner.iterator();
            while (it.next()) |entry| self.allocator.free(entry.value_ptr.*);
            interner.deinit();
        }
        return renderNode(self, self.output, &interner);
    }

    pub fn simplify(self: *const Expression) Error!Expression {
        const result = try simplifier.simplify(self.allocator, &self.graph, self.output);
        return Expression.fromGraph(self.allocator, self.registry, result.graph, result.output);
    }

    fn graphBackward(self: *const Expression, values: []const f32) Error![]f32 {
        var adjoints: []f32 = try self.allocator.alloc(f32, self.graph.nodes.items.len);
        @memset(adjoints, 0.0);
        adjoints[@intCast(self.output)] = 1.0;

        var idx: usize = self.graph.nodes.items.len;
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
                    var args: []f32 = try self.allocator.alloc(f32, custom.inputs.len);
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
            var args: []f32 = try expr.allocator.alloc(f32, custom.inputs.len);
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

fn renderBinary(expr: *const Expression, cache: *std.AutoHashMap(NodeId, []const u8), binaries: graph_mod.BinaryInputs, op: []const u8) Error![]u8 {
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
