const std = @import("std");
const types = @import("types.zig");

pub const NodeId = types.NodeId;
pub const Error = types.Error;

pub const SymbolicOpInfo = struct {
    node_id: NodeId,
    inputs: []const NodeId,
    upstream: NodeId,
};

pub const SymbolicRuleFn = *const fn (ctx: *anyopaque, info: SymbolicOpInfo) Error!void;
pub const PrinterFn = *const fn (allocator: std.mem.Allocator, args: []const []const u8) Error![]u8;

pub const Operation = struct {
    name: []const u8,
    arity: u8,
    forward: *const fn (args: []const f32) f32,
    backward: *const fn (grad_output: f32, args: []const f32, arg_index: usize) f32,
    symbolic: ?SymbolicRuleFn = null,
    printer: ?PrinterFn = null,
};
