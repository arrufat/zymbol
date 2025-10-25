const std = @import("std");
const types = @import("types.zig");
const operation = @import("operation.zig");

pub const NodeId = types.NodeId;
pub const Error = types.Error;
pub const Operation = operation.Operation;

pub const NodeKind = enum {
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
    tan,
    sinh,
    cosh,
    tanh,
    custom,
};

pub const BinaryInputs = struct { lhs: NodeId, rhs: NodeId };

pub const Node = struct {
    kind: NodeKind,
    payload: Payload,

    pub const Payload = union(enum) {
        none,
        input: []const u8,
        constant: f32,
        unary: NodeId,
        binary: BinaryInputs,
        custom: struct { op: *const Operation, inputs: []NodeId },
    };
};

pub const Graph = struct {
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

    pub fn addCustom(self: *Graph, op: *const Operation, operands: []const NodeId) Error!NodeId {
        if (operands.len != op.arity) return Error.ArityMismatch;
        const dup = try self.allocator.dupe(NodeId, operands);
        errdefer self.allocator.free(dup);
        return try self.appendNode(.{
            .kind = .custom,
            .payload = .{ .custom = .{ .op = op, .inputs = dup } },
        });
    }

    pub fn node(self: *const Graph, id: NodeId) Node {
        return self.nodes.items[@as(usize, @intCast(id))];
    }

    pub fn copyNode(self: *Graph, source: *const Graph, id: NodeId, map: *std.AutoHashMap(NodeId, NodeId)) !NodeId {
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
            .log, .exp, .sin, .cos, .tan, .sinh, .cosh, .tanh => blk: {
                const operand = try self.copyNode(source, orig_node.payload.unary, map);
                break :blk try self.addUnary(orig_node.kind, operand);
            },
            .custom => blk: {
                const custom = orig_node.payload.custom;
                var buffer: []NodeId = try self.allocator.alloc(NodeId, custom.inputs.len);
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
