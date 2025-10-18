const std = @import("std");
const operation = @import("operation.zig");

pub const Operation = operation.Operation;

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
        var it: std.StringHashMapUnmanaged(Operation).Iterator = self.ops.iterator();
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
};
