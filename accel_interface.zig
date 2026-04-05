const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkBackwardFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        futhark.futhark_context_config_set_device(cfg, "0");
        futhark.futhark_context_config_set_default_thread_block_size(cfg, 256);
        futhark.futhark_context_config_set_default_grid_size(cfg, 128);
        futhark.futhark_context_config_set_default_tile_size(cfg, 32);

        const ctx = futhark.futhark_context_new(cfg);
        futhark.futhark_context_config_free(cfg);

        if (ctx == null) return AccelError.FutharkContextFailed;

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }
};

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0 };
        }

        var ptr: ?*anyopaque = null;
        const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
        if (err != cuda.cudaSuccess) {
            return AccelError.CudaHostAllocFailed;
        }

        return Self{
            .ptr = ptr,
            .size = size,
        };
    }

    pub fn free(self: *Self) void {
        if (self.ptr) |p| {
            _ = cuda.cudaFreeHost(p);
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        if (self.size == 0) return &[_]T{};
        const count = self.size / @sizeOf(T);
        if (count == 0) return &[_]T{};
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const f32) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn newZeros(ctx: *FutharkContext, n: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (n == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, n) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, zeros.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn valuesInto(self: *Self, ctx: *FutharkContext, out: []f32) AccelError!void {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (out.len != self.len) return AccelError.InvalidDimensions;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, out.ptr) != 0) {
            return AccelError.FutharkValuesFailed;
        }
    }

    pub fn toSlice(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const data = allocator.alloc(f32, self.len) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, data.ptr) != 0) {
            allocator.free(data);
            return AccelError.FutharkValuesFailed;
        }
        return data;
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    weights_s: FutharkArray1DF32,
    weights_t: FutharkArray1DF32,
    bias_s: FutharkArray1DF32,
    bias_t: FutharkArray1DF32,
    velocity_s: FutharkArray1DF32,
    velocity_t: FutharkArray1DF32,
    dims_arr: FutharkArray1DI64,
    model_dim: usize,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const dim_sq = model_dim * model_dim;

        var weights_s = try FutharkArray1DF32.newZeros(&ctx, dim_sq, std.heap.page_allocator);
        errdefer weights_s.free(&ctx);

        var weights_t = try FutharkArray1DF32.newZeros(&ctx, dim_sq, std.heap.page_allocator);
        errdefer weights_t.free(&ctx);

        var bias_s = try FutharkArray1DF32.newZeros(&ctx, model_dim, std.heap.page_allocator);
        errdefer bias_s.free(&ctx);

        var bias_t = try FutharkArray1DF32.newZeros(&ctx, model_dim, std.heap.page_allocator);
        errdefer bias_t.free(&ctx);

        var velocity_s = try FutharkArray1DF32.newZeros(&ctx, dim_sq, std.heap.page_allocator);
        errdefer velocity_s.free(&ctx);

        var velocity_t = try FutharkArray1DF32.newZeros(&ctx, dim_sq, std.heap.page_allocator);
        errdefer velocity_t.free(&ctx);

        const dims_data = [_]i64{ @intCast(model_dim), @intCast(model_dim) };
        var dims_arr = try FutharkArray1DI64.newFromSlice(&ctx, &dims_data);
        errdefer dims_arr.free(&ctx);

        return Self{
            .ctx = ctx,
            .weights_s = weights_s,
            .weights_t = weights_t,
            .bias_s = bias_s,
            .bias_t = bias_t,
            .velocity_s = velocity_s,
            .velocity_t = velocity_t,
            .dims_arr = dims_arr,
            .model_dim = model_dim,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        self.dims_arr.free(&self.ctx);
        self.velocity_t.free(&self.ctx);
        self.velocity_s.free(&self.ctx);
        self.bias_t.free(&self.ctx);
        self.bias_s.free(&self.ctx);
        self.weights_t.free(&self.ctx);
        self.weights_s.free(&self.ctx);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn setWeightsS(self: *Self, data: []const f32, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_s.free(&self.ctx);
        self.weights_s = try FutharkArray1DF32.newFromSlice(&self.ctx, data);
    }

    pub fn setWeightsT(self: *Self, data: []const f32, rows: usize, cols: usize) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;

        self.weights_t.free(&self.ctx);
        self.weights_t = try FutharkArray1DF32.newFromSlice(&self.ctx, data);
    }

    pub fn setBiasS(self: *Self, data: []const f32) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (data.len != self.model_dim) return AccelError.InvalidDimensions;

        self.bias_s.free(&self.ctx);
        self.bias_s = try FutharkArray1DF32.newFromSlice(&self.ctx, data);
    }

    pub fn setBiasT(self: *Self, data: []const f32) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (data.len != self.model_dim) return AccelError.InvalidDimensions;

        self.bias_t.free(&self.ctx);
        self.bias_t = try FutharkArray1DF32.newFromSlice(&self.ctx, data);
    }

    pub fn forward(self: *Self, input_data: []const f32) AccelError!FutharkArray1DF32 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        var input_arr = try FutharkArray1DF32.newFromSlice(&self.ctx, input_data);
        defer input_arr.free(&self.ctx);

        var output: ?*futhark.struct_futhark_f32_1d = null;
        const result = futhark.futhark_entry_rsf_forward(
            self.ctx.ctx,
            &output,
            input_arr.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.bias_s.arr,
            self.bias_t.arr,
            self.dims_arr.arr,
        );

        if (result != 0) return AccelError.FutharkForwardFailed;
        if (output == null) return AccelError.NullPointer;

        return FutharkArray1DF32{ .arr = output, .len = input_data.len };
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        var gpu_out = try self.forward(input.data);
        defer gpu_out.free(&self.ctx);

        const shape = [_]usize{ rows, cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;

        if (futhark.futhark_values_f32_1d(self.ctx.ctx, gpu_out.arr, result.data.ptr) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return result;
    }

    pub fn backward(self: *Self, grad_output: []const f32, input_data: []const f32) AccelError!RSFBackwardResult {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        var grad_arr = try FutharkArray1DF32.newFromSlice(&self.ctx, grad_output);
        defer grad_arr.free(&self.ctx);

        var input_arr = try FutharkArray1DF32.newFromSlice(&self.ctx, input_data);
        defer input_arr.free(&self.ctx);

        var out_grad_input: ?*futhark.struct_futhark_f32_1d = null;
        var out_grad_ws: ?*futhark.struct_futhark_f32_1d = null;
        var out_grad_wt: ?*futhark.struct_futhark_f32_1d = null;
        var out_grad_bs: ?*futhark.struct_futhark_f32_1d = null;
        var out_grad_bt: ?*futhark.struct_futhark_f32_1d = null;

        const result = futhark.futhark_entry_rsf_backward(
            self.ctx.ctx,
            &out_grad_input,
            &out_grad_ws,
            &out_grad_wt,
            &out_grad_bs,
            &out_grad_bt,
            grad_arr.arr,
            input_arr.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.dims_arr.arr,
        );

        if (result != 0) return AccelError.FutharkBackwardFailed;
        if (out_grad_input == null or out_grad_ws == null or out_grad_wt == null) return AccelError.NullPointer;
        if (out_grad_bs == null or out_grad_bt == null) return AccelError.NullPointer;

        return RSFBackwardResult{
            .grad_input = FutharkArray1DF32{ .arr = out_grad_input, .len = input_data.len },
            .grad_ws = FutharkArray1DF32{ .arr = out_grad_ws, .len = self.model_dim * self.model_dim },
            .grad_wt = FutharkArray1DF32{ .arr = out_grad_wt, .len = self.model_dim * self.model_dim },
            .grad_bs = FutharkArray1DF32{ .arr = out_grad_bs, .len = self.model_dim },
            .grad_bt = FutharkArray1DF32{ .arr = out_grad_bt, .len = self.model_dim },
        };
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }
};

pub const RSFBackwardResult = struct {
    grad_input: FutharkArray1DF32,
    grad_ws: FutharkArray1DF32,
    grad_wt: FutharkArray1DF32,
    grad_bs: FutharkArray1DF32,
    grad_bt: FutharkArray1DF32,

    pub fn free(self: *RSFBackwardResult, ctx: *FutharkContext) void {
        self.grad_input.free(ctx);
        self.grad_ws.free(ctx);
        self.grad_wt.free(ctx);
        self.grad_bs.free(ctx);
        self.grad_bt.free(ctx);
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }

    pub fn softmax(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fi = try FutharkArray1DF32.fromTensor(&self.ctx, input);
        defer fi.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_1d = null;
        if (futhark.futhark_entry_apply_softmax(self.ctx.ctx, &out_arr, fi.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray1DF32{ .arr = out_arr, .len = input.shape.dims[0] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }

    pub fn layerNorm(self: *Self, input: *const core_tensor.Tensor, gamma: *const core_tensor.Tensor, beta: *const core_tensor.Tensor, eps: f32, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fi = try FutharkArray1DF32.fromTensor(&self.ctx, input);
        defer fi.free(&self.ctx);
        var fg = try FutharkArray1DF32.fromTensor(&self.ctx, gamma);
        defer fg.free(&self.ctx);
        var fb = try FutharkArray1DF32.fromTensor(&self.ctx, beta);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_1d = null;
        if (futhark.futhark_entry_apply_layer_norm(self.ctx.ctx, &out_arr, fi.arr, fg.arr, fb.arr, eps) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray1DF32{ .arr = out_arr, .len = input.shape.dims[0] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }

    pub fn relu(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fi = try FutharkArray1DF32.fromTensor(&self.ctx, input);
        defer fi.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_1d = null;
        if (futhark.futhark_entry_apply_relu(self.ctx.ctx, &out_arr, fi.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray1DF32{ .arr = out_arr, .len = input.shape.dims[0] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }

    pub fn gelu(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fi = try FutharkArray1DF32.fromTensor(&self.ctx, input);
        defer fi.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_1d = null;
        if (futhark.futhark_entry_apply_gelu(self.ctx.ctx, &out_arr, fi.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray1DF32{ .arr = out_arr, .len = input.shape.dims[0] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const GPULayerWeights = struct {
    ws: FutharkArray1DF32,
    wt: FutharkArray1DF32,
    bs: FutharkArray1DF32,
    bt: FutharkArray1DF32,

    pub fn initFromCPU(ctx: *FutharkContext, s_weight: []const f32, t_weight: []const f32, s_bias: []const f32, t_bias: []const f32) AccelError!GPULayerWeights {
        var ws = try FutharkArray1DF32.newFromSlice(ctx, s_weight);
        errdefer ws.free(ctx);
        var wt = try FutharkArray1DF32.newFromSlice(ctx, t_weight);
        errdefer wt.free(ctx);
        var bs = try FutharkArray1DF32.newFromSlice(ctx, s_bias);
        errdefer bs.free(ctx);
        var bt = try FutharkArray1DF32.newFromSlice(ctx, t_bias);
        errdefer bt.free(ctx);
        return GPULayerWeights{ .ws = ws, .wt = wt, .bs = bs, .bt = bt };
    }

    pub fn deinit(self: *GPULayerWeights, ctx: *FutharkContext) void {
        self.ws.free(ctx);
        self.wt.free(ctx);
        self.bs.free(ctx);
        self.bt.free(ctx);
    }
};

pub fn forwardLayerGPU(ctx: *FutharkContext, input: FutharkArray1DF32, lw: *const GPULayerWeights, dims: *const FutharkArray1DI64) AccelError!FutharkArray1DF32 {
    if (ctx.ctx == null) return AccelError.NullPointer;
    if (input.arr == null) return AccelError.NullPointer;
    var output: ?*futhark.struct_futhark_f32_1d = null;
    const result = futhark.futhark_entry_rsf_forward(
        ctx.ctx,
        &output,
        input.arr,
        lw.ws.arr,
        lw.wt.arr,
        lw.bs.arr,
        lw.bt.arr,
        dims.arr,
    );
    if (result != 0) return AccelError.FutharkForwardFailed;
    if (output == null) return AccelError.NullPointer;
    return FutharkArray1DF32{ .arr = output, .len = input.len };
}
