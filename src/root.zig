const std = @import("std");

// Computational graph node
const NodeId = u32;

const NodeType = enum {
    input,
    constant,
    add,
    sub,
    mul,
    div,
    pow,
    log, // Natural logarithm - needed for power gradients
    custom, // For user-defined operations
};

// Custom operation definition
pub const CustomOp = struct {
    name: []const u8,
    arity: u8, // Number of arguments (1 for unary, 2 for binary, etc.)

    // Forward evaluation: computes the operation
    forward: *const fn (args: []const f32) f32,

    // Backward: computes gradient contributions
    // Given: grad_output (gradient flowing back), args (input values), arg_index (which input we're computing grad for)
    // Returns: gradient contribution to that input
    backward: *const fn (grad_output: f32, args: []const f32, arg_index: usize) f32,

    // Optional: symbolic differentiation (returns expression as string)
    // If null, uses numeric backward
    toString: ?*const fn (args: []const []const u8, allocator: std.mem.Allocator) error{OutOfMemory}![]u8 = null,
    toStringGrad: ?*const fn (args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) error{OutOfMemory}![]u8 = null,
};

const Node = struct {
    node_type: NodeType,
    // For binary ops: [left_id, right_id]
    // For unary ops: [operand_id]
    // For custom ops: variable length stored separately
    // For input/constant: unused
    inputs: [2]NodeId,
    custom_inputs: ?[]NodeId, // For custom ops with arity > 2
    // Only used for constant nodes
    value: f32,
    // For input nodes
    name: []const u8,
    // For custom operations
    custom_op_name: ?[]const u8,
};

pub const OpRegistry = struct {
    ops: std.StringHashMap(CustomOp),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) OpRegistry {
        const registry = OpRegistry{
            .ops = std.StringHashMap(CustomOp).init(allocator),
            .allocator = allocator,
        };
        return registry;
    }

    pub fn deinit(self: *OpRegistry) void {
        self.ops.deinit();
    }

    pub fn register(self: *OpRegistry, op: CustomOp) !void {
        try self.ops.put(op.name, op);
    }

    pub fn get(self: *OpRegistry, name: []const u8) ?CustomOp {
        return self.ops.get(name);
    }

    // Register built-in operations that were previously hardcoded
    pub fn registerBuiltins(self: *OpRegistry) !void {
        // Sin
        try self.register(.{
            .name = "sin",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return @sin(args[0]);
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = arg_index;
                    return grad_output * @cos(args[0]);
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "sin({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "cos({s})", .{args[0]});
                }
            }.tsg,
        });

        // Cos
        try self.register(.{
            .name = "cos",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return @cos(args[0]);
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = arg_index;
                    return -grad_output * @sin(args[0]);
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "cos({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "(-1 * sin({s}))", .{args[0]});
                }
            }.tsg,
        });

        // Exp
        try self.register(.{
            .name = "exp",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return @exp(args[0]);
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = arg_index;
                    return grad_output * @exp(args[0]);
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "exp({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "exp({s})", .{args[0]});
                }
            }.tsg,
        });

        // Log
        try self.register(.{
            .name = "log",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return @log(args[0]);
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = arg_index;
                    return grad_output / args[0];
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "log({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "(1 / {s})", .{args[0]});
                }
            }.tsg,
        });

        // ReLU - now it's just another custom op!
        try self.register(.{
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
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "relu({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "relu_grad({s})", .{args[0]});
                }
            }.tsg,
        });

        // ReLU gradient helper
        try self.register(.{
            .name = "relu_grad",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return if (args[0] > 0.0) 1.0 else 0.0;
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = args;
                    _ = arg_index;
                    _ = grad_output;
                    return 0.0;
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "relu_grad({s})", .{args[0]});
                }
            }.ts,
        });

        // Sigmoid
        try self.register(.{
            .name = "sigmoid",
            .arity = 1,
            .forward = struct {
                fn f(args: []const f32) f32 {
                    return 1.0 / (1.0 + @exp(-args[0]));
                }
            }.f,
            .backward = struct {
                fn b(grad_output: f32, args: []const f32, arg_index: usize) f32 {
                    _ = arg_index;
                    const s = 1.0 / (1.0 + @exp(-args[0]));
                    return grad_output * s * (1.0 - s);
                }
            }.b,
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "sigmoid({s})", .{args[0]});
                }
            }.ts,
            .toStringGrad = struct {
                fn tsg(args: []const []const u8, arg_index: usize, allocator: std.mem.Allocator) ![]u8 {
                    _ = arg_index;
                    return try std.fmt.allocPrint(allocator, "(sigmoid({s}) * (1 - sigmoid({s})))", .{ args[0], args[0] });
                }
            }.tsg,
        });

        // Max (binary)
        try self.register(.{
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
            .toString = struct {
                fn ts(args: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
                    return try std.fmt.allocPrint(allocator, "max({s}, {s})", .{ args[0], args[1] });
                }
            }.ts,
        });
    }
};

const ComputationGraph = struct {
    nodes: std.ArrayList(Node),
    allocator: std.mem.Allocator,
    input_map: std.StringHashMap(NodeId),
    output_id: NodeId,
    registry: *OpRegistry,
    allocated_strings: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, registry: *OpRegistry) ComputationGraph {
        return .{
            .nodes = .empty,
            .allocator = allocator,
            .input_map = std.StringHashMap(NodeId).init(allocator),
            .output_id = 0,
            .registry = registry,
            .allocated_strings = .empty,
        };
    }

    pub fn initFromString(allocator: std.mem.Allocator, registry: *OpRegistry, expr_str: []const u8) !ComputationGraph {
        var graph = ComputationGraph.init(allocator, registry);

        const tokens = try tokenize(allocator, expr_str);
        defer allocator.free(tokens);

        const output = try Parser.parse(&graph, tokens);
        graph.output_id = output;

        return graph;
    }

    pub fn deinit(self: *ComputationGraph) void {
        // Free custom_inputs arrays
        for (self.nodes.items) |node| {
            if (node.custom_inputs) |inputs| {
                self.allocator.free(inputs);
            }
        }
        // Free allocated strings
        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }
        self.allocated_strings.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.input_map.deinit();
    }

    pub fn toString(self: *ComputationGraph) ![]u8 {
        return try self.nodeToString(self.output_id);
    }

    fn nodeToString(self: *ComputationGraph, node_id: NodeId) error{ OutOfMemory, MissingCustomOpName, UnknownCustomOp }![]u8 {
        const node = self.nodes.items[node_id];

        return switch (node.node_type) {
            .input => try std.fmt.allocPrint(self.allocator, "{s}", .{node.name}),
            .constant => blk: {
                if (@floor(node.value) == node.value and @abs(node.value) < 1000000) {
                    break :blk try std.fmt.allocPrint(self.allocator, "{d:.0}", .{node.value});
                } else {
                    break :blk try std.fmt.allocPrint(self.allocator, "{d}", .{node.value});
                }
            },
            .add => blk: {
                const left = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(left);
                const right = try self.nodeToString(node.inputs[1]);
                defer self.allocator.free(right);
                break :blk try std.fmt.allocPrint(self.allocator, "({s} + {s})", .{ left, right });
            },
            .sub => blk: {
                const left = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(left);
                const right = try self.nodeToString(node.inputs[1]);
                defer self.allocator.free(right);
                break :blk try std.fmt.allocPrint(self.allocator, "({s} - {s})", .{ left, right });
            },
            .mul => blk: {
                const left = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(left);
                const right = try self.nodeToString(node.inputs[1]);
                defer self.allocator.free(right);
                break :blk try std.fmt.allocPrint(self.allocator, "({s} * {s})", .{ left, right });
            },
            .div => blk: {
                const left = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(left);
                const right = try self.nodeToString(node.inputs[1]);
                defer self.allocator.free(right);
                break :blk try std.fmt.allocPrint(self.allocator, "({s} / {s})", .{ left, right });
            },
            .pow => blk: {
                const left = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(left);
                const right = try self.nodeToString(node.inputs[1]);
                defer self.allocator.free(right);
                break :blk try std.fmt.allocPrint(self.allocator, "({s} ^ {s})", .{ left, right });
            },
            .log => blk: {
                const arg = try self.nodeToString(node.inputs[0]);
                defer self.allocator.free(arg);
                break :blk try std.fmt.allocPrint(self.allocator, "log({s})", .{arg});
            },
            .custom => blk: {
                const op_name = node.custom_op_name orelse return error.MissingCustomOpName;
                const op = self.registry.get(op_name) orelse return error.UnknownCustomOp;

                if (op.toString) |toStringFn| {
                    // Get string representations of all inputs
                    const input_ids = if (node.custom_inputs) |ci| ci else node.inputs[0..op.arity];
                    var arg_strs = try self.allocator.alloc([]const u8, input_ids.len);
                    defer {
                        for (arg_strs) |s| self.allocator.free(s);
                        self.allocator.free(arg_strs);
                    }

                    for (input_ids, 0..) |input_id, i| {
                        arg_strs[i] = try self.nodeToString(input_id);
                    }

                    break :blk try toStringFn(arg_strs, self.allocator);
                } else {
                    // Fallback: generic representation
                    break :blk try std.fmt.allocPrint(self.allocator, "{s}(...)", .{op_name});
                }
            },
        };
    }

    pub fn addInput(self: *ComputationGraph, name: []const u8) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        // Duplicate the name string to avoid use-after-free
        const name_copy = try self.allocator.dupe(u8, name);
        try self.allocated_strings.append(self.allocator, name_copy);
        try self.nodes.append(self.allocator, .{
            .node_type = .input,
            .inputs = undefined,
            .custom_inputs = null,
            .value = undefined,
            .name = name_copy,
            .custom_op_name = null,
        });
        try self.input_map.put(name_copy, id);
        return id;
    }

    pub fn addConstant(self: *ComputationGraph, val: f32) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        try self.nodes.append(self.allocator, .{
            .node_type = .constant,
            .inputs = undefined,
            .custom_inputs = null,
            .value = val,
            .name = "",
            .custom_op_name = null,
        });
        return id;
    }

    pub fn addUnaryOp(self: *ComputationGraph, op: NodeType, operand: NodeId) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        try self.nodes.append(self.allocator, .{
            .node_type = op,
            .inputs = .{ operand, 0 },
            .custom_inputs = null,
            .value = undefined,
            .name = "",
            .custom_op_name = null,
        });
        return id;
    }

    pub fn addBinaryOp(self: *ComputationGraph, op: NodeType, left: NodeId, right: NodeId) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        try self.nodes.append(self.allocator, .{
            .node_type = op,
            .inputs = .{ left, right },
            .custom_inputs = null,
            .value = undefined,
            .name = "",
            .custom_op_name = null,
        });
        return id;
    }

    pub fn addCustomOp(self: *ComputationGraph, op_name: []const u8, operands: []const NodeId) !NodeId {
        const id = @as(NodeId, @intCast(self.nodes.items.len));
        const op = self.registry.get(op_name) orelse return error.UnknownCustomOp;

        // Validate operands.len matches op.arity
        if (operands.len != op.arity) {
            return error.ArityMismatch;
        }

        const inputs_copy = if (op.arity <= 2) null else try self.allocator.dupe(NodeId, operands);

        // Duplicate the op_name string to avoid use-after-free
        const op_name_copy = try self.allocator.dupe(u8, op_name);
        try self.allocated_strings.append(self.allocator, op_name_copy);

        try self.nodes.append(self.allocator, .{
            .node_type = .custom,
            .inputs = if (op.arity <= 2) .{ operands[0], if (operands.len > 1) operands[1] else 0 } else undefined,
            .custom_inputs = inputs_copy,
            .value = undefined,
            .name = "",
            .custom_op_name = op_name_copy,
        });
        return id;
    }

    pub fn forward(self: *ComputationGraph, inputs: std.StringHashMap(f32)) ![]f32 {
        var values = try self.allocator.alloc(f32, self.nodes.items.len);

        for (self.nodes.items, 0..) |node, i| {
            values[i] = switch (node.node_type) {
                .input => inputs.get(node.name) orelse return error.MissingInput,
                .constant => node.value,
                .add => values[node.inputs[0]] + values[node.inputs[1]],
                .sub => values[node.inputs[0]] - values[node.inputs[1]],
                .mul => values[node.inputs[0]] * values[node.inputs[1]],
                .div => values[node.inputs[0]] / values[node.inputs[1]],
                .pow => std.math.pow(f32, values[node.inputs[0]], values[node.inputs[1]]),
                .log => @log(values[node.inputs[0]]),
                .custom => blk: {
                    const op_name = node.custom_op_name orelse return error.MissingCustomOpName;
                    const op = self.registry.get(op_name) orelse return error.UnknownCustomOp;

                    const input_ids = if (node.custom_inputs) |ci| ci else node.inputs[0..op.arity];
                    var args = try self.allocator.alloc(f32, input_ids.len);
                    defer self.allocator.free(args);

                    for (input_ids, 0..) |input_id, j| {
                        args[j] = values[input_id];
                    }

                    break :blk op.forward(args);
                },
            };
        }

        return values;
    }

    pub fn backward(self: *ComputationGraph, values: []f32) ![]f32 {
        var adjoints = try self.allocator.alloc(f32, self.nodes.items.len);
        @memset(adjoints, 0.0);

        adjoints[self.output_id] = 1.0;

        var i = self.output_id;
        while (i > 0) : (i -= 1) {
            const node = self.nodes.items[i];
            const grd = adjoints[i];

            switch (node.node_type) {
                .input, .constant => {},
                .add => {
                    adjoints[node.inputs[0]] += grd;
                    adjoints[node.inputs[1]] += grd;
                },
                .sub => {
                    adjoints[node.inputs[0]] += grd;
                    adjoints[node.inputs[1]] -= grd;
                },
                .mul => {
                    const a = values[node.inputs[0]];
                    const b = values[node.inputs[1]];
                    adjoints[node.inputs[0]] += grd * b;
                    adjoints[node.inputs[1]] += grd * a;
                },
                .div => {
                    const a = values[node.inputs[0]];
                    const b = values[node.inputs[1]];
                    adjoints[node.inputs[0]] += grd / b;
                    adjoints[node.inputs[1]] -= grd * a / (b * b);
                },
                .pow => {
                    const a = values[node.inputs[0]];
                    const b = values[node.inputs[1]];
                    const result = values[i];
                    adjoints[node.inputs[0]] += grd * b * std.math.pow(f32, a, b - 1.0);
                    adjoints[node.inputs[1]] += grd * result * @log(a);
                },
                .log => {
                    const a = values[node.inputs[0]];
                    adjoints[node.inputs[0]] += grd / a;
                },
                .custom => {
                    const op_name = node.custom_op_name orelse continue;
                    const op = self.registry.get(op_name) orelse continue;

                    const input_ids = if (node.custom_inputs) |ci| ci else node.inputs[0..op.arity];
                    var args = try self.allocator.alloc(f32, input_ids.len);
                    defer self.allocator.free(args);

                    for (input_ids, 0..) |input_id, j| {
                        args[j] = values[input_id];
                    }

                    for (input_ids, 0..) |input_id, j| {
                        adjoints[input_id] += op.backward(grd, args, j);
                    }
                },
            }
        }

        return adjoints;
    }

    pub fn gradient(self: *ComputationGraph, wrt: []const u8, inputs: std.StringHashMap(f32)) !f32 {
        const values = try self.forward(inputs);
        defer self.allocator.free(values);

        const adjoints = try self.backward(values);
        defer self.allocator.free(adjoints);

        const input_id = self.input_map.get(wrt) orelse return error.UnknownInput;
        return adjoints[input_id];
    }

    pub fn symbolicGrad(self: *ComputationGraph, wrt: []const u8) !ComputationGraph {
        var grad_graph = ComputationGraph.init(self.allocator, self.registry);
        errdefer grad_graph.deinit();

        var grad_node_map = std.AutoHashMap(NodeId, NodeId).init(self.allocator);
        defer grad_node_map.deinit();

        var orig_to_new = std.AutoHashMap(NodeId, NodeId).init(self.allocator);
        defer orig_to_new.deinit();

        for (self.nodes.items, 0..) |node, idx| {
            if (node.node_type == .input) {
                const new_id = try grad_graph.addInput(node.name);
                try orig_to_new.put(@intCast(idx), new_id);
            }
        }

        const one = try grad_graph.addConstant(1.0);
        try grad_node_map.put(self.output_id, one);

        var i = self.output_id;
        while (i > 0) : (i -= 1) {
            const node = self.nodes.items[i];
            const grad_of_node = grad_node_map.get(i) orelse continue;

            switch (node.node_type) {
                .input, .constant => {},
                .add => {
                    try accumulateGradient(&grad_graph, &grad_node_map, node.inputs[0], grad_of_node);
                    try accumulateGradient(&grad_graph, &grad_node_map, node.inputs[1], grad_of_node);
                },
                .sub => {
                    try accumulateGradient(&grad_graph, &grad_node_map, node.inputs[0], grad_of_node);

                    const neg_grad = try grad_graph.addBinaryOp(.mul, grad_of_node, try grad_graph.addConstant(-1.0));
                    try accumulateGradient(&grad_graph, &grad_node_map, node.inputs[1], neg_grad);
                },
                .mul => {
                    const a_id = node.inputs[0];
                    const b_id = node.inputs[1];

                    const a_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, a_id);
                    const b_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, b_id);

                    const grad_a = try grad_graph.addBinaryOp(.mul, grad_of_node, b_node_in_grad);
                    try accumulateGradient(&grad_graph, &grad_node_map, a_id, grad_a);

                    const grad_b = try grad_graph.addBinaryOp(.mul, grad_of_node, a_node_in_grad);
                    try accumulateGradient(&grad_graph, &grad_node_map, b_id, grad_b);
                },
                .div => {
                    const a_id = node.inputs[0];
                    const b_id = node.inputs[1];

                    const a_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, a_id);
                    const b_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, b_id);

                    const grad_a = try grad_graph.addBinaryOp(.div, grad_of_node, b_node_in_grad);
                    try accumulateGradient(&grad_graph, &grad_node_map, a_id, grad_a);

                    const b_squared = try grad_graph.addBinaryOp(.mul, b_node_in_grad, b_node_in_grad);
                    const neg_a = try grad_graph.addBinaryOp(.mul, a_node_in_grad, try grad_graph.addConstant(-1.0));
                    const neg_a_over_b2 = try grad_graph.addBinaryOp(.div, neg_a, b_squared);
                    const grad_b = try grad_graph.addBinaryOp(.mul, grad_of_node, neg_a_over_b2);
                    try accumulateGradient(&grad_graph, &grad_node_map, b_id, grad_b);
                },
                .pow => {
                    const a_id = node.inputs[0];
                    const b_id = node.inputs[1];

                    const a_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, a_id);
                    const b_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, b_id);

                    // Gradient w.r.t. base: grad * b * a^(b-1)
                    const b_minus_1 = try grad_graph.addBinaryOp(.sub, b_node_in_grad, try grad_graph.addConstant(1.0));
                    const a_pow_b_minus_1 = try grad_graph.addBinaryOp(.pow, a_node_in_grad, b_minus_1);
                    const deriv_a = try grad_graph.addBinaryOp(.mul, b_node_in_grad, a_pow_b_minus_1);
                    const grad_a = try grad_graph.addBinaryOp(.mul, grad_of_node, deriv_a);
                    try accumulateGradient(&grad_graph, &grad_node_map, a_id, grad_a);

                    // Gradient w.r.t. exponent: grad * a^b * ln(a)
                    const a_pow_b = try grad_graph.addBinaryOp(.pow, a_node_in_grad, b_node_in_grad);
                    const log_a = try grad_graph.addUnaryOp(.log, a_node_in_grad);
                    const deriv_b = try grad_graph.addBinaryOp(.mul, a_pow_b, log_a);
                    const grad_b = try grad_graph.addBinaryOp(.mul, grad_of_node, deriv_b);
                    try accumulateGradient(&grad_graph, &grad_node_map, b_id, grad_b);
                },
                .log => {
                    const a_id = node.inputs[0];
                    const a_node_in_grad = try getOrCopyNode(self, &grad_graph, &orig_to_new, a_id);

                    // Gradient of log(a) w.r.t. a is 1/a
                    const deriv = try grad_graph.addBinaryOp(.div, try grad_graph.addConstant(1.0), a_node_in_grad);
                    const grad_a = try grad_graph.addBinaryOp(.mul, grad_of_node, deriv);
                    try accumulateGradient(&grad_graph, &grad_node_map, a_id, grad_a);
                },
                .custom => {
                    const op_name = node.custom_op_name orelse continue;
                    const op = self.registry.get(op_name) orelse continue;

                    const input_ids = if (node.custom_inputs) |ci| ci else node.inputs[0..op.arity];

                    // Copy input nodes to gradient graph
                    var input_nodes_in_grad = try self.allocator.alloc(NodeId, input_ids.len);
                    defer self.allocator.free(input_nodes_in_grad);

                    for (input_ids, 0..) |input_id, j| {
                        input_nodes_in_grad[j] = try getOrCopyNode(self, &grad_graph, &orig_to_new, input_id);
                    }

                    // If the operation has symbolic gradient, use it
                    if (op.toStringGrad) |toStringGradFn| {
                        // For each input, compute its gradient symbolically
                        for (input_ids, 0..) |input_id, j| {
                            // Get string representations of all inputs from the gradient graph
                            var arg_strs = try self.allocator.alloc([]const u8, input_ids.len);
                            defer {
                                for (arg_strs) |s| self.allocator.free(s);
                                self.allocator.free(arg_strs);
                            }

                            for (input_nodes_in_grad, 0..) |grad_node_id, k| {
                                arg_strs[k] = try grad_graph.nodeToString(grad_node_id);
                            }

                            // Call toStringGrad to get the gradient expression
                            const grad_expr = try toStringGradFn(arg_strs, j, self.allocator);
                            defer self.allocator.free(grad_expr);

                            // Parse the gradient expression to build the subgraph
                            const tokens = try tokenize(self.allocator, grad_expr);
                            defer self.allocator.free(tokens);

                            var sub_parser = Parser{
                                .tokens = tokens,
                                .pos = 0,
                                .graph = &grad_graph,
                            };
                            const local_grad_node = try sub_parser.parseExpression();

                            const grad_input = try grad_graph.addBinaryOp(.mul, grad_of_node, local_grad_node);
                            try accumulateGradient(&grad_graph, &grad_node_map, input_id, grad_input);
                        }
                    } else {
                        // Custom op lacks toStringGrad - cannot build symbolic gradient.
                        // The numeric backward() is still available via ComputationGraph.gradient()
                        // which uses the .backward implementation correctly.
                        return error.SymbolicGradientUnsupported;
                    }
                },
            }
        }

        const input_id = self.input_map.get(wrt) orelse return error.UnknownInput;
        const grad_output = grad_node_map.get(input_id) orelse try grad_graph.addConstant(0.0);

        grad_graph.output_id = grad_output;
        return grad_graph;
    }

    pub fn eval(self: *ComputationGraph, inputs: std.StringHashMap(f32)) !f32 {
        const values = try self.forward(inputs);
        defer self.allocator.free(values);
        return values[self.output_id];
    }
};

fn accumulateGradient(
    grad_graph: *ComputationGraph,
    grad_map: *std.AutoHashMap(NodeId, NodeId),
    node_id: NodeId,
    new_grad: NodeId,
) !void {
    if (grad_map.get(node_id)) |existing_grad| {
        const sum = try grad_graph.addBinaryOp(.add, existing_grad, new_grad);
        try grad_map.put(node_id, sum);
    } else {
        try grad_map.put(node_id, new_grad);
    }
}

fn getOrCopyNode(
    orig_graph: *const ComputationGraph,
    grad_graph: *ComputationGraph,
    orig_to_new: *std.AutoHashMap(NodeId, NodeId),
    orig_id: NodeId,
) !NodeId {
    if (orig_to_new.get(orig_id)) |new_id| {
        return new_id;
    }

    const orig_node = orig_graph.nodes.items[orig_id];

    const new_id = switch (orig_node.node_type) {
        .input => blk: {
            const id = try grad_graph.addInput(orig_node.name);
            break :blk id;
        },
        .constant => try grad_graph.addConstant(orig_node.value),
        .add, .sub, .mul, .div, .pow => blk: {
            const left = try getOrCopyNode(orig_graph, grad_graph, orig_to_new, orig_node.inputs[0]);
            const right = try getOrCopyNode(orig_graph, grad_graph, orig_to_new, orig_node.inputs[1]);
            break :blk try grad_graph.addBinaryOp(orig_node.node_type, left, right);
        },
        .log => blk: {
            const operand = try getOrCopyNode(orig_graph, grad_graph, orig_to_new, orig_node.inputs[0]);
            break :blk try grad_graph.addUnaryOp(.log, operand);
        },
        .custom => blk: {
            const op_name = orig_node.custom_op_name orelse return error.MissingCustomOpName;
            const op = orig_graph.registry.get(op_name) orelse return error.UnknownCustomOp;

            const input_ids = if (orig_node.custom_inputs) |ci| ci else orig_node.inputs[0..op.arity];
            var new_inputs = try grad_graph.allocator.alloc(NodeId, input_ids.len);
            defer grad_graph.allocator.free(new_inputs);

            for (input_ids, 0..) |input_id, j| {
                new_inputs[j] = try getOrCopyNode(orig_graph, grad_graph, orig_to_new, input_id);
            }

            break :blk try grad_graph.addCustomOp(op_name, new_inputs);
        },
    };

    try orig_to_new.put(orig_id, new_id);
    return new_id;
}

// Tokenizer and Parser remain mostly the same, but now handle custom ops
const Token = union(enum) {
    number: f32,
    identifier: []const u8,
    plus,
    minus,
    star,
    slash,
    caret,
    lparen,
    rparen,
    comma,
};

fn tokenize(allocator: std.mem.Allocator, input: []const u8) ![]Token {
    var tokens: std.ArrayList(Token) = .empty;
    var i: usize = 0;

    while (i < input.len) {
        const c = input[i];

        if (std.ascii.isWhitespace(c)) {
            i += 1;
            continue;
        }

        if (std.ascii.isDigit(c) or c == '.') {
            var end = i;
            while (end < input.len and (std.ascii.isDigit(input[end]) or input[end] == '.')) {
                end += 1;
            }
            const num = try std.fmt.parseFloat(f32, input[i..end]);
            try tokens.append(allocator, .{ .number = num });
            i = end;
            continue;
        }

        if (std.ascii.isAlphabetic(c) or c == '_') {
            var end = i;
            while (end < input.len and (std.ascii.isAlphanumeric(input[end]) or input[end] == '_')) {
                end += 1;
            }
            const ident = input[i..end];
            try tokens.append(allocator, .{ .identifier = ident });
            i = end;
            continue;
        }

        switch (c) {
            '+' => try tokens.append(allocator, .plus),
            '-' => try tokens.append(allocator, .minus),
            '*' => try tokens.append(allocator, .star),
            '/' => try tokens.append(allocator, .slash),
            '^' => try tokens.append(allocator, .caret),
            '(' => try tokens.append(allocator, .lparen),
            ')' => try tokens.append(allocator, .rparen),
            ',' => try tokens.append(allocator, .comma),
            else => return error.UnexpectedCharacter,
        }
        i += 1;
    }

    return try tokens.toOwnedSlice(allocator);
}

const ParserError = error{
    UnexpectedEndOfInput,
    UnexpectedToken,
    UnexpectedCharacter,
    ExpectedCommaOrRParen,
    ExpectedRightParen,
    UnknownFunction,
    UnknownCustomOp,
    OutOfMemory,
    InvalidCharacter,
    ArityMismatch,
};

const Parser = struct {
    tokens: []Token,
    pos: usize,
    graph: *ComputationGraph,

    fn parse(graph: *ComputationGraph, tokens: []Token) ParserError!NodeId {
        var parser = Parser{
            .tokens = tokens,
            .pos = 0,
            .graph = graph,
        };
        return try parser.parseExpression();
    }

    fn parseExpression(self: *Parser) ParserError!NodeId {
        return try self.parseAddSub();
    }

    fn parseAddSub(self: *Parser) ParserError!NodeId {
        var left = try self.parseMulDiv();

        while (self.pos < self.tokens.len) {
            const token = self.tokens[self.pos];
            switch (token) {
                .plus => {
                    self.pos += 1;
                    const right = try self.parseMulDiv();
                    left = try self.graph.addBinaryOp(.add, left, right);
                },
                .minus => {
                    self.pos += 1;
                    const right = try self.parseMulDiv();
                    left = try self.graph.addBinaryOp(.sub, left, right);
                },
                else => break,
            }
        }

        return left;
    }

    fn parseMulDiv(self: *Parser) ParserError!NodeId {
        var left = try self.parsePower();

        while (self.pos < self.tokens.len) {
            const token = self.tokens[self.pos];
            switch (token) {
                .star => {
                    self.pos += 1;
                    const right = try self.parsePower();
                    left = try self.graph.addBinaryOp(.mul, left, right);
                },
                .slash => {
                    self.pos += 1;
                    const right = try self.parsePower();
                    left = try self.graph.addBinaryOp(.div, left, right);
                },
                else => break,
            }
        }

        return left;
    }

    fn parsePower(self: *Parser) ParserError!NodeId {
        var left = try self.parseUnary();

        if (self.pos < self.tokens.len) {
            const token = self.tokens[self.pos];
            if (token == .caret) {
                self.pos += 1;
                const right = try self.parseUnary();
                left = try self.graph.addBinaryOp(.pow, left, right);
            }
        }

        return left;
    }

    fn parseUnary(self: *Parser) ParserError!NodeId {
        if (self.pos >= self.tokens.len) return error.UnexpectedEndOfInput;

        const token = self.tokens[self.pos];

        // Check if it's a function call (identifier followed by lparen)
        if (token == .identifier) {
            if (self.pos + 1 < self.tokens.len and self.tokens[self.pos + 1] == .lparen) {
                const func_name = token.identifier;
                self.pos += 2; // Skip identifier and lparen

                // Parse arguments
                var args: std.ArrayList(NodeId) = .empty;
                defer args.deinit(self.graph.allocator);

                if (self.pos < self.tokens.len and self.tokens[self.pos] != .rparen) {
                    while (true) {
                        const arg = try self.parseExpression();
                        try args.append(self.graph.allocator, arg);

                        if (self.pos >= self.tokens.len) return error.UnexpectedEndOfInput;

                        if (self.tokens[self.pos] == .comma) {
                            self.pos += 1;
                            continue;
                        } else if (self.tokens[self.pos] == .rparen) {
                            break;
                        } else {
                            return error.ExpectedCommaOrRParen;
                        }
                    }
                }

                if (self.pos >= self.tokens.len or self.tokens[self.pos] != .rparen) {
                    return error.ExpectedRightParen;
                }
                self.pos += 1;

                // Try to find this as a custom operation
                if (self.graph.registry.get(func_name)) |_| {
                    return try self.graph.addCustomOp(func_name, args.items);
                } else {
                    return error.UnknownFunction;
                }
            }
        }

        return try self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) !NodeId {
        if (self.pos >= self.tokens.len) return error.UnexpectedEndOfInput;

        const token = self.tokens[self.pos];
        self.pos += 1;

        switch (token) {
            .number => |n| return try self.graph.addConstant(n),
            .identifier => |id| {
                if (self.graph.input_map.get(id)) |node_id| {
                    return node_id;
                }
                return try self.graph.addInput(id);
            },
            .lparen => {
                const expr = try self.parseExpression();
                if (self.pos >= self.tokens.len or self.tokens[self.pos] != .rparen) {
                    return error.ExpectedRightParen;
                }
                self.pos += 1;
                return expr;
            },
            else => return error.UnexpectedToken,
        }
    }
};

pub fn parse(allocator: std.mem.Allocator, registry: *OpRegistry, expr_str: []const u8) !ComputationGraph {
    return try ComputationGraph.initFromString(allocator, registry, expr_str);
}

pub fn grad(graph: *ComputationGraph, wrt: []const u8) !ComputationGraph {
    return try graph.symbolicGrad(wrt);
}
