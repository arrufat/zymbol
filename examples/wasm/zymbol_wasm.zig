const std = @import("std");
const zymbol = @import("zymbol");
const Registry = zymbol.Registry;
const Expression = zymbol.Expression;

const wasm_allocator = std.heap.wasm_allocator;

var output_buffer: std.ArrayList(u8) = .empty;
var output_len: usize = 0;
var zero_byte: u8 = 0;

fn setOutput(message: []const u8) !void {
    try output_buffer.resize(wasm_allocator, message.len + 1);
    @memcpy(output_buffer.items[0..message.len], message);
    output_buffer.items[message.len] = 0;
    output_len = message.len;
}

fn setErrorMessage(err_msg: []const u8) !void {
    const formatted = try std.fmt.allocPrint(wasm_allocator, "error: {s}", .{err_msg});
    defer wasm_allocator.free(formatted);
    try setOutput(formatted);
}

fn setError(err: anyerror) !void {
    try setErrorMessage(@errorName(err));
}

fn trimmedOrDefault(slice: []const u8, default_value: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, slice, " \t\r\n");
    return if (trimmed.len == 0) default_value else trimmed;
}

fn derive(expr: []const u8, variable: []const u8, simplify: bool) !void {
    var arena = std.heap.ArenaAllocator.init(wasm_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var registry = Registry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    var expression = try Expression.parse(allocator, &registry, expr);
    defer expression.deinit();

    var gradient = if (simplify)
        try expression.symbolicGradient(variable)
    else
        try expression.symbolicGradientRaw(variable);
    defer gradient.deinit();

    const grad_str = try gradient.toString();
    try setOutput(grad_str);
}

pub export fn zymbol_derive(expr_ptr: [*]const u8, expr_len: usize, var_ptr: [*]const u8, var_len: usize, simplify_flag: u32) i32 {
    const expr_slice: []const u8 = if (expr_len == 0) "" else expr_ptr[0..expr_len];
    const var_slice: []const u8 = if (var_len == 0) "" else var_ptr[0..var_len];

    const expr = std.mem.trim(u8, expr_slice, " \t\r\n");
    const variable = trimmedOrDefault(var_slice, "x");
    const simplify = simplify_flag != 0;

    if (expr.len == 0) {
        setErrorMessage("expression is empty") catch {};
        return -1;
    }

    if (variable.len == 0) {
        setErrorMessage("variable is empty") catch {};
        return -1;
    }

    if (derive(expr, variable, simplify)) |_| {
        return 0;
    } else |err| {
        setError(err) catch {};
        return -1;
    }
}

pub export fn zymbol_result_ptr() [*]const u8 {
    return if (output_len == 0)
        @as([*]const u8, @ptrCast(&zero_byte))
    else
        output_buffer.items.ptr;
}

pub export fn zymbol_result_len() usize {
    return output_len;
}

pub export fn zymbol_clear() void {
    output_buffer.clearAndFree(wasm_allocator);
    output_len = 0;
}

pub export fn zymbol_alloc(size: usize) ?[*]u8 {
    if (size == 0) return null;
    const slice = wasm_allocator.alloc(u8, size) catch return null;
    return slice.ptr;
}

pub export fn zymbol_free(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    wasm_allocator.free(ptr[0..size]);
}

pub fn panic(message: []const u8, stack_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = stack_trace;
    _ = ret_addr;
    setErrorMessage(message) catch {};
    while (true) {}
}
