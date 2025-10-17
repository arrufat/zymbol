const std = @import("std");
const zymbol = @import("zymbol");

const Registry = zymbol.Registry;
const Expression = zymbol.Expression;
const Sample = struct { key: []const u8, value: f32 };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    demo(allocator, &registry, "relu(x)", "x", &[_]Sample{
        .{ .key = "x", .value = 2.0 },
        .{ .key = "x", .value = -1.0 },
    });
    demo(allocator, &registry, "sigmoid(x)", "x", &[_]Sample{
        .{ .key = "x", .value = 0.0 },
    });
    demo(allocator, &registry, "relu(x) * sigmoid(x)", "x", &[_]Sample{
        .{ .key = "x", .value = 2.0 },
    });
    demo(allocator, &registry, "x ^ 2", "x", &[_]Sample{
        .{ .key = "x", .value = 3.0 },
    });
    demo(allocator, &registry, "x ^ y", "x", &[_]Sample{
        .{ .key = "x", .value = 3.0 },
        .{ .key = "y", .value = 2.0 },
    });
}

fn demo(
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: []const u8,
    variable: []const u8,
    samples: []const Sample,
) void {
    std.debug.print("\nf(x) = {s}\n", .{source});

    var expr = Expression.parse(allocator, registry, source) catch |err| {
        std.debug.print("  parse error: {s}\n", .{@errorName(err)});
        return;
    };
    defer expr.deinit();

    const canonical = expr.toString() catch |err| {
        std.debug.print("  print error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(canonical);
    std.debug.print("  canonical: {s}\n", .{canonical});

    var inputs = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    for (samples) |entry| {
        inputs.put(entry.key, entry.value) catch |err| {
            std.debug.print("  put error: {s}\n", .{@errorName(err)});
            return;
        };
    }

    const value = expr.evaluate(inputs) catch |err| {
        std.debug.print("  eval error: {s}\n", .{@errorName(err)});
        return;
    };
    std.debug.print("  eval = {d}\n", .{value});

    const numeric = expr.numericGradient(variable, inputs) catch |err| {
        std.debug.print("  numeric grad error: {s}\n", .{@errorName(err)});
        return;
    };

    const grad_expr_result = expr.symbolicGradient(variable) catch |err| {
        std.debug.print("  symbolic grad error: {s}\n", .{@errorName(err)});
        return;
    };
    var grad_expr = grad_expr_result;
    defer grad_expr.deinit();

    const grad_str = grad_expr.toString() catch |err| {
        std.debug.print("  grad print error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(grad_str);

    const grad_val = grad_expr.evaluate(inputs) catch |err| {
        std.debug.print("  grad eval error: {s}\n", .{@errorName(err)});
        return;
    };

    std.debug.print("  symbolic grad: {s}\n", .{grad_str});
    std.debug.print("  numeric grad:  {d}\n", .{numeric});
    std.debug.print("  grad eval:     {d}\n", .{grad_val});
}
