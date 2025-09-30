const std = @import("std");

const zymbol = @import("root.zig");
const OpRegistry = zymbol.OpRegistry;
const parse = zymbol.parse;
const grad = zymbol.grad;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create and setup registry
    var registry = OpRegistry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    // Example 1: ReLU is now just another operation!
    std.debug.print("Example 1: f(x) = relu(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "relu(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();

        try inputs.put("x", 2.0);
        std.debug.print("  f(2) = {d}\n", .{try f.eval(inputs)});

        try inputs.put("x", -1.0);
        std.debug.print("  f(-1) = {d}\n", .{try f.eval(inputs)});

        var df = try grad(&f, "x");
        defer df.deinit();

        const df_str = try df.toString();
        defer allocator.free(df_str);
        std.debug.print("  df/dx = {s}\n", .{df_str});

        try inputs.put("x", 2.0);
        std.debug.print("  df/dx(2) = {d} (expected: 1)\n", .{try df.eval(inputs)});

        try inputs.put("x", -1.0);
        std.debug.print("  df/dx(-1) = {d} (expected: 0)\n\n", .{try df.eval(inputs)});
    }

    // Example 2: Sigmoid
    std.debug.print("Example 2: f(x) = sigmoid(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "sigmoid(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 0.0);

        std.debug.print("  f(0) = {d} (expected: 0.5)\n", .{try f.eval(inputs)});

        var df = try grad(&f, "x");
        defer df.deinit();

        std.debug.print("  df/dx(0) = {d} (expected: 0.25)\n\n", .{try df.eval(inputs)});
    }

    // Example 3: Combining operations
    std.debug.print("Example 3: f(x) = relu(x) * sigmoid(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "relu(x) * sigmoid(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 2.0);

        std.debug.print("  f(2) = {d}\n", .{try f.eval(inputs)});

        const grad_val = try f.gradient("x", inputs);
        std.debug.print("  df/dx(2) = {d}\n\n", .{grad_val});
    }

    // Example 4: Binary custom op - max
    std.debug.print("Example 4: f(x, y) = max(x, y)\n", .{});
    {
        var f = try parse(allocator, &registry, "max(x, y)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x, y) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 3.0);
        try inputs.put("y", 5.0);

        std.debug.print("  f(3, 5) = {d}\n", .{try f.eval(inputs)});
        std.debug.print("  df/dx(3, 5) = {d} (expected: 0)\n", .{try f.gradient("x", inputs)});
        std.debug.print("  df/dy(3, 5) = {d} (expected: 1)\n", .{try f.gradient("y", inputs)});
    }
}
