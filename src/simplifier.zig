const std = @import("std");
const types = @import("types.zig");
const graph_mod = @import("graph.zig");

pub const Error = types.Error;
pub const NodeId = types.NodeId;
pub const NodeKind = graph_mod.NodeKind;
pub const Graph = graph_mod.Graph;
pub const Node = graph_mod.Node;

pub const SimplifyResult = struct {
    graph: Graph,
    output: NodeId,
};

pub fn simplify(
    allocator: std.mem.Allocator,
    source_graph: *const Graph,
    output: NodeId,
) Error!SimplifyResult {
    var target: Graph = Graph.init(allocator);
    var needs_deinit = true;
    defer if (needs_deinit) target.deinit();

    var simplifier = Simplifier.init(allocator, source_graph, &target);
    defer simplifier.deinit();

    const simplified_output = try simplifier.simplifyNode(output);
    needs_deinit = false;
    return .{ .graph = target, .output = simplified_output };
}

const Simplifier = struct {
    allocator: std.mem.Allocator,
    source: *const Graph,
    target: *Graph,
    cache: std.AutoHashMap(NodeId, NodeId),

    const epsilon: f32 = 1e-6;

    fn init(allocator: std.mem.Allocator, source: *const Graph, target: *Graph) Simplifier {
        return .{
            .allocator = allocator,
            .source = source,
            .target = target,
            .cache = std.AutoHashMap(NodeId, NodeId).init(allocator),
        };
    }

    fn deinit(self: *Simplifier) void {
        self.cache.deinit();
    }

    fn simplifyNode(self: *Simplifier, id: NodeId) Error!NodeId {
        if (self.cache.get(id)) |existing| return existing;
        const node = self.source.node(id);
        const simplified: NodeId = switch (node.kind) {
            .input => try self.target.addInput(node.payload.input),
            .constant => try self.target.addConstant(node.payload.constant),
            .add => blk: {
                const binary = node.payload.binary;
                const lhs = try self.simplifyNode(binary.lhs);
                const rhs = try self.simplifyNode(binary.rhs);
                break :blk try self.simplifyAdd(lhs, rhs);
            },
            .sub => blk: {
                const binary = node.payload.binary;
                const lhs = try self.simplifyNode(binary.lhs);
                const rhs = try self.simplifyNode(binary.rhs);
                break :blk try self.simplifySub(lhs, rhs);
            },
            .mul => blk: {
                const binary = node.payload.binary;
                const lhs = try self.simplifyNode(binary.lhs);
                const rhs = try self.simplifyNode(binary.rhs);
                break :blk try self.simplifyMul(lhs, rhs);
            },
            .div => blk: {
                const binary = node.payload.binary;
                const lhs = try self.simplifyNode(binary.lhs);
                const rhs = try self.simplifyNode(binary.rhs);
                break :blk try self.simplifyDiv(lhs, rhs);
            },
            .pow => blk: {
                const binary = node.payload.binary;
                const lhs = try self.simplifyNode(binary.lhs);
                const rhs = try self.simplifyNode(binary.rhs);
                break :blk try self.simplifyPow(lhs, rhs);
            },
            .log => try self.simplifyUnary(.log, try self.simplifyNode(node.payload.unary)),
            .exp => try self.simplifyUnary(.exp, try self.simplifyNode(node.payload.unary)),
            .sin => try self.simplifyUnary(.sin, try self.simplifyNode(node.payload.unary)),
            .cos => try self.simplifyUnary(.cos, try self.simplifyNode(node.payload.unary)),
            .tan => try self.simplifyUnary(.tan, try self.simplifyNode(node.payload.unary)),
            .custom => blk: {
                const custom = node.payload.custom;
                var simplified_inputs: []NodeId = try self.allocator.alloc(NodeId, custom.inputs.len);
                defer self.allocator.free(simplified_inputs);
                for (custom.inputs, 0..) |child, idx| {
                    simplified_inputs[idx] = try self.simplifyNode(child);
                }
                break :blk try self.target.addCustom(custom.op, simplified_inputs);
            },
        };
        try self.cache.put(id, simplified);
        return simplified;
    }

    fn simplifyAdd(self: *Simplifier, lhs: NodeId, rhs: NodeId) Error!NodeId {
        const lhs_const = self.constantValue(lhs);
        const rhs_const = self.constantValue(rhs);

        if (lhs_const) |lhs_val| {
            if (approxEqual(lhs_val, 0.0)) return rhs;
        }
        if (rhs_const) |rhs_val| {
            if (approxEqual(rhs_val, 0.0)) return lhs;
        }
        if (lhs == rhs) {
            const two = try self.target.addConstant(2.0);
            return self.simplifyMul(two, lhs);
        }

        var terms: std.ArrayList(NodeId) = .empty;
        defer terms.deinit(self.allocator);

        var const_sum: f32 = 0.0;
        var has_const = false;

        try self.collectAddTerms(lhs, &terms, &const_sum, &has_const);
        try self.collectAddTerms(rhs, &terms, &const_sum, &has_const);

        var result: ?NodeId = null;
        for (terms.items) |term| {
            result = if (result) |acc|
                try self.target.addBinary(.add, acc, term)
            else
                term;
        }

        if (has_const and !approxZero(const_sum)) {
            const const_node = try self.target.addConstant(const_sum);
            result = if (result) |acc|
                try self.target.addBinary(.add, acc, const_node)
            else
                const_node;
        }

        return result orelse try self.target.addConstant(0.0);
    }

    fn simplifySub(self: *Simplifier, lhs: NodeId, rhs: NodeId) Error!NodeId {
        const lhs_const = self.constantValue(lhs);
        const rhs_const = self.constantValue(rhs);

        if (rhs_const) |rhs_val| {
            if (approxEqual(rhs_val, 0.0)) return lhs;
        }
        if (lhs == rhs) {
            return self.target.addConstant(0.0);
        }
        if (lhs_const) |lhs_val| {
            if (rhs_const) |rhs_val| {
                const result = lhs_val - rhs_val;
                if (std.math.isFinite(result)) return self.target.addConstant(result);
            }
            if (approxEqual(lhs_val, 0.0)) {
                return self.simplifyNegate(rhs);
            }
        }
        return self.target.addBinary(.sub, lhs, rhs);
    }

    fn simplifyMul(self: *Simplifier, lhs: NodeId, rhs: NodeId) Error!NodeId {
        const lhs_const = self.constantValue(lhs);
        const rhs_const = self.constantValue(rhs);

        if (lhs_const) |lhs_val| {
            if (approxEqual(lhs_val, 0.0)) return lhs;
            if (approxEqual(lhs_val, 1.0)) return rhs;
        }
        if (rhs_const) |rhs_val| {
            if (approxEqual(rhs_val, 0.0)) return rhs;
            if (approxEqual(rhs_val, 1.0)) return lhs;
            if (approxEqual(rhs_val, -1.0)) return self.simplifyNegate(lhs);
        }
        if (lhs == rhs) {
            const two = try self.target.addConstant(2.0);
            return self.simplifyPow(lhs, two);
        }

        var factors: std.ArrayList(NodeId) = .empty;
        defer factors.deinit(self.allocator);

        var const_product: f32 = 1.0;
        var has_const = false;
        var zero_detected = false;

        try self.collectMulTerms(lhs, &factors, &const_product, &has_const, &zero_detected);
        try self.collectMulTerms(rhs, &factors, &const_product, &has_const, &zero_detected);

        if (zero_detected or approxZero(const_product)) {
            return self.target.addConstant(0.0);
        }

        var negative = false;
        var result: ?NodeId = null;

        if (has_const) {
            if (approxOne(const_product)) {
                // no-op
            } else if (approxEqual(const_product, -1.0)) {
                negative = true;
            } else {
                const const_node = try self.target.addConstant(const_product);
                result = const_node;
            }
        }

        for (factors.items) |factor| {
            result = if (result) |acc|
                try self.target.addBinary(.mul, acc, factor)
            else
                factor;
        }

        const final = result orelse {
            if (negative) {
                return self.target.addConstant(-1.0);
            }
            return self.target.addConstant(1.0);
        };

        if (negative) {
            return self.simplifyNegate(final);
        }
        return final;
    }

    fn simplifyDiv(self: *Simplifier, lhs: NodeId, rhs: NodeId) Error!NodeId {
        const lhs_const = self.constantValue(lhs);
        const rhs_const = self.constantValue(rhs);

        if (lhs_const) |lhs_val| {
            if (approxEqual(lhs_val, 0.0)) return lhs;
        }
        if (rhs_const) |rhs_val| {
            if (approxEqual(rhs_val, 1.0)) return lhs;
            if (approxEqual(rhs_val, 0.0)) return self.target.addBinary(.div, lhs, rhs);
            if (approxEqual(rhs_val, -1.0)) return self.simplifyNegate(lhs);
        }
        if (lhs == rhs) {
            if (lhs_const) |lhs_val| {
                if (!approxEqual(lhs_val, 0.0)) return self.target.addConstant(1.0);
            } else {
                return self.target.addConstant(1.0);
            }
        }

        const rhs_node = self.target.node(rhs);
        if (rhs_node.kind == .div) {
            const inner = rhs_node.payload.binary;
            if (self.constantValue(inner.lhs)) |inner_lhs| {
                if (approxEqual(inner_lhs, 1.0)) {
                    return self.simplifyMul(lhs, inner.rhs);
                }
            }
        }

        if (lhs_const) |lhs_val| {
            if (rhs_const) |rhs_val| {
                if (!approxEqual(rhs_val, 0.0)) {
                    const result = lhs_val / rhs_val;
                    if (std.math.isFinite(result)) return self.target.addConstant(result);
                }
            }
        }
        return self.target.addBinary(.div, lhs, rhs);
    }

    fn simplifyPow(self: *Simplifier, base: NodeId, exponent: NodeId) Error!NodeId {
        const base_const = self.constantValue(base);
        const exponent_const = self.constantValue(exponent);

        if (exponent_const) |exp_val| {
            if (approxEqual(exp_val, 1.0)) return base;
            if (approxEqual(exp_val, 0.0)) return self.target.addConstant(1.0);
        }
        if (base_const) |base_val| {
            if (approxEqual(base_val, 0.0)) {
                if (exponent_const) |exp_val| {
                    if (exp_val > 0.0) return base;
                }
            }
            if (approxEqual(base_val, 1.0)) return self.target.addConstant(1.0);
        }
        if (base_const) |base_val| {
            if (exponent_const) |exp_val| {
                const result = std.math.pow(f32, base_val, exp_val);
                if (std.math.isFinite(result)) return self.target.addConstant(result);
            }
        }
        return self.target.addBinary(.pow, base, exponent);
    }

    fn simplifyUnary(self: *Simplifier, kind: NodeKind, operand: NodeId) Error!NodeId {
        if (self.constantValue(operand)) |value| {
            const folded = switch (kind) {
                .log => @log(value),
                .exp => @exp(value),
                .sin => @sin(value),
                .cos => @cos(value),
                .tan => @tan(value),
                else => value,
            };
            if (std.math.isFinite(folded)) return self.target.addConstant(folded);
        }
        return self.target.addUnary(kind, operand);
    }

    fn simplifyNegate(self: *Simplifier, node_id: NodeId) Error!NodeId {
        if (self.constantValue(node_id)) |value| {
            return self.target.addConstant(-value);
        }
        const neg_one = try self.target.addConstant(-1.0);
        return self.target.addBinary(.mul, neg_one, node_id);
    }

    fn constantValue(self: *Simplifier, id: NodeId) ?f32 {
        const node = self.target.node(id);
        return switch (node.kind) {
            .constant => node.payload.constant,
            else => null,
        };
    }

    fn collectAddTerms(
        self: *Simplifier,
        id: NodeId,
        terms: *std.ArrayList(NodeId),
        const_sum: *f32,
        has_const: *bool,
    ) Error!void {
        const node = self.target.node(id);
        switch (node.kind) {
            .add => {
                const binary = node.payload.binary;
                try self.collectAddTerms(binary.lhs, terms, const_sum, has_const);
                try self.collectAddTerms(binary.rhs, terms, const_sum, has_const);
            },
            .constant => {
                const current = if (has_const.*) const_sum.* else 0.0;
                const_sum.* = current + node.payload.constant;
                has_const.* = true;
            },
            else => try terms.append(self.allocator, id),
        }
    }

    fn collectMulTerms(
        self: *Simplifier,
        id: NodeId,
        factors: *std.ArrayList(NodeId),
        const_product: *f32,
        has_const: *bool,
        zero_detected: *bool,
    ) Error!void {
        if (zero_detected.*) return;
        const node = self.target.node(id);
        switch (node.kind) {
            .mul => {
                const binary = node.payload.binary;
                try self.collectMulTerms(binary.lhs, factors, const_product, has_const, zero_detected);
                try self.collectMulTerms(binary.rhs, factors, const_product, has_const, zero_detected);
            },
            .constant => {
                const value = node.payload.constant;
                if (approxZero(value)) {
                    zero_detected.* = true;
                    return;
                }
                const current = if (has_const.*) const_product.* else 1.0;
                const_product.* = current * value;
                has_const.* = true;
            },
            else => try factors.append(self.allocator, id),
        }
    }
};

fn approxEqual(lhs: f32, rhs: f32) bool {
    return std.math.approxEqAbs(f32, lhs, rhs, Simplifier.epsilon);
}

fn approxZero(value: f32) bool {
    return approxEqual(value, 0.0);
}

fn approxOne(value: f32) bool {
    return approxEqual(value, 1.0);
}
