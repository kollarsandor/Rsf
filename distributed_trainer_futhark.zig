const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray1DF32 = accel.FutharkArray1DF32;
const PinnedMemory = accel.PinnedMemory;

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    max_line_size: usize = 10 * 1024 * 1024,
    checkpoint_version: u32 = 2,
};

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: RSFAccelerator,
    model_dim: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: usize,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        if (model_dim == 0) return error.InvalidModelDim;
        if (local_batch_size == 0) return error.InvalidBatchSize;
        if (coordinator.world_size == 0) return error.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;
        if (coordinator.world_size > 1 and config.momentum != 0.0) return error.UnsupportedDistributedMomentum;

        const vocab = &[_][]const u8{
            "a",     "about",   "all",   "also",  "and",   "as",    "at",
            "be",    "because", "but",   "by",    "can",   "come",  "could",
            "day",   "do",      "even",  "find",  "first", "for",   "from",
            "get",   "give",    "go",    "have",  "he",    "her",   "here",
            "him",   "his",     "how",   "i",     "if",    "in",    "into",
            "it",    "its",     "just",  "know",  "like",  "look",  "make",
            "man",   "many",    "me",    "more",  "my",    "new",   "no",
            "not",   "now",     "of",    "on",    "one",   "only",  "or",
            "other", "our",     "out",   "people", "say",  "see",   "she",
            "so",    "some",    "take",  "tell",  "than",  "that",  "the",
            "their", "them",    "then",  "there", "these", "they",  "thing",
            "think", "this",    "those", "time",  "to",    "two",   "up",
            "use",   "very",    "want",  "way",   "we",    "well",  "what",
            "when",  "which",   "who",   "will",  "with",  "would", "year",
            "you",   "your",
        };
        if (model_dim < vocab.len) return error.ModelDimTooSmallForVocabulary;
        const empty_anchors: []const []const u8 = &.{};

        var tokenizer = try MGT.init(allocator, vocab, empty_anchors);
        errdefer tokenizer.deinit();

        var accelerator = try RSFAccelerator.init(model_dim);
        errdefer accelerator.deinit();

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer,
            .accelerator = accelerator,
            .model_dim = model_dim,
            .vocab_size = vocab.len,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
        };
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch {};
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }

    fn hasDatasetText(self: *DistributedTrainerFuthark, line: []const u8) bool {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{ .allocate = .alloc_always },
        ) catch return false;
        defer parsed.deinit();

        return switch (parsed.value) {
            .object => |obj| blk: {
                const text_value = obj.get("text") orelse break :blk false;
                break :blk switch (text_value) {
                    .string => |text| text.len > 0,
                    else => false,
                };
            },
            else => false,
        };
    }

    fn extractDatasetText(self: *DistributedTrainerFuthark, line: []const u8) !?[]const u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{ .allocate = .alloc_always },
        ) catch return null;
        defer parsed.deinit();

        return switch (parsed.value) {
            .object => |obj| blk: {
                const text_value = obj.get("text") orelse break :blk null;
                break :blk switch (text_value) {
                    .string => |text| if (text.len > 0)
                        try self.allocator.dupe(u8, text)
                    else
                        null,
                    else => null,
                };
            },
            else => null,
        };
    }

    fn appendDatasetRange(
        self: *DistributedTrainerFuthark,
        dataset_path: []const u8,
        start_valid_index: usize,
        count: usize,
        samples: *std.ArrayList([]const u8),
    ) !void {
        if (count == 0) {
            return;
        }

        const end_valid_index = start_valid_index + count;
        var appended: usize = 0;
        var valid_index: usize = 0;

        const load_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| {
            std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
            return err;
        };
        defer load_file.close();

        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();

        while (try load_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);

            const maybe_text = try self.extractDatasetText(line);
            if (maybe_text) |text_copy| {
                errdefer self.allocator.free(text_copy);

                if (valid_index >= start_valid_index and valid_index < end_valid_index) {
                    try samples.append(text_copy);
                    appended += 1;
                    valid_index += 1;
                    if (appended == count) {
                        return;
                    }
                } else {
                    self.allocator.free(text_copy);
                    valid_index += 1;
                }
            }
        }

        return error.UnexpectedEndOfFile;
    }

    fn readWeightsFlat(self: *DistributedTrainerFuthark, arr: *FutharkArray1DF32) ![]f32 {
        return arr.toSlice(&self.accelerator.ctx, self.allocator);
    }

    fn averageDeltaInPlace(self: *DistributedTrainerFuthark, delta: []f32) !void {
        if (delta.len == 0 or self.coordinator.world_size <= 1) {
            return;
        }

        const byte_count = try std.math.mul(usize, delta.len, @sizeOf(f32));
        const delta_dev = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(delta_dev);

        try self.coordinator.copyHostToDevice(delta_dev, delta, byte_count);
        try self.coordinator.allReduceFloat32(delta_dev, delta_dev, delta.len);
        try self.coordinator.copyDeviceToHost(delta, delta_dev, byte_count);
        try self.coordinator.synchronize();

        const inv_world: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
        for (delta) |*value| {
            value.* *= inv_world;
        }
    }

    fn applyDelta(self: *DistributedTrainerFuthark, base: []const f32, delta: []const f32, which: enum { s, t }) !void {
        if (base.len != delta.len) {
            return error.InvalidWeightsShape;
        }

        var merged = try self.allocator.alloc(f32, base.len);
        defer self.allocator.free(merged);

        for (base, delta, 0..) |base_value, delta_value, idx| {
            merged[idx] = base_value + delta_value;
        }

        switch (which) {
            .s => try self.accelerator.setWeightsS(merged, self.model_dim, self.model_dim),
            .t => try self.accelerator.setWeightsT(merged, self.model_dim, self.model_dim),
        }
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return error.InvalidWorldSize;
        if (self.coordinator.rank >= self.coordinator.world_size) return error.InvalidRank;

        var total_line_count: usize = 0;
        var valid_sample_count: usize = 0;

        {
            const count_file = std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only }) catch |err| {
                std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
                return err;
            };
            defer count_file.close();

            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();

            while (try count_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
                defer self.allocator.free(line);
                total_line_count += 1;
                if (self.hasDatasetText(line)) {
                    valid_sample_count += 1;
                }
            }
        }

        if (valid_sample_count == 0) {
            std.debug.print("[Rank {d}] ERROR: Dataset does not contain any usable samples\n", .{self.coordinator.rank});
            return error.EmptyDataset;
        }

        const samples_per_rank = try std.math.divCeil(usize, valid_sample_count, self.coordinator.world_size);
        const logical_start = try std.math.mul(usize, self.coordinator.rank, samples_per_rank);
        const start_valid_index = logical_start % valid_sample_count;
        const primary_count = @min(samples_per_rank, valid_sample_count - start_valid_index);
        const wrap_count = samples_per_rank - primary_count;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| {
                self.allocator.free(sample);
            }
            samples.deinit();
        }

        try self.appendDatasetRange(dataset_path, start_valid_index, primary_count, &samples);
        if (wrap_count > 0) {
            try self.appendDatasetRange(dataset_path, 0, wrap_count, &samples);
        }

        if (samples.items.len != samples_per_rank) {
            return error.InvalidDatasetPartition;
        }

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples per rank from {d} usable samples across {d} lines\n", .{
                self.coordinator.rank,
                samples.items.len,
                valid_sample_count,
                total_line_count,
            });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return error.InvalidBatchSize;

        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        var batch_start: usize = 0;
        while (batch_start < samples.len) {
            const remaining = samples.len - batch_start;
            const batch_len = @min(self.local_batch_size, remaining);
            const batch_end = batch_start + batch_len;
            const batch = samples[batch_start..batch_end];

            const loss = try self.trainStepFuthark(batch);
            total_loss += loss;
            num_batches += 1;

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, loss });
            }

            self.global_step +|= 1;
            batch_start = batch_end;
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        const loss_and_count_dev = try self.coordinator.allocDeviceMemory(2 * @sizeOf(f32));
        defer self.coordinator.freeDeviceMemory(loss_and_count_dev);

        try self.coordinator.copyHostToDevice(loss_and_count_dev, std.mem.asBytes(&loss_and_count), 2 * @sizeOf(f32));
        try self.coordinator.allReduceFloat32(loss_and_count_dev, loss_and_count_dev, 2);
        try self.coordinator.copyDeviceToHost(std.mem.asBytes(&loss_and_count), loss_and_count_dev, 2 * @sizeOf(f32));
        try self.coordinator.synchronize();

        const global_loss_sum = loss_and_count[0];
        const global_batch_count = loss_and_count[1];

        if (global_batch_count < 1.0) {
            std.debug.print("[WARNING] No batches processed across all ranks\n", .{});
            return 0.0;
        }

        return global_loss_sum / global_batch_count;
    }

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !f32 {
        if (batch.len == 0) return 0.0;

        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| {
                list.deinit();
            }
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            errdefer token_list.deinit();
            try self.tokenizer.encode(text, &token_list);
            try token_lists.append(token_list);
        }

        var max_seq_len: usize = 0;
        for (token_lists.items) |list| {
            max_seq_len = @max(max_seq_len, list.items.len);
        }

        if (max_seq_len == 0) return 0.0;

        const batch_size = batch.len;
        const batch_rows = try std.math.mul(usize, batch_size, max_seq_len);
        const data_elements = try std.math.mul(usize, batch_rows, self.model_dim);
        const data_size = try std.math.mul(usize, data_elements, @sizeOf(f32));

        var pinned_mem = try PinnedMemory.alloc(data_size);
        defer pinned_mem.free();

        const input_f32_data = pinned_mem.asSlice(f32) orelse return error.AllocationFailed;
        if (input_f32_data.len != data_elements) {
            return error.InvalidPinnedMemorySize;
        }
        @memset(input_f32_data, @as(f32, 0.0));

        var batch_idx: usize = 0;
        while (batch_idx < token_lists.items.len) : (batch_idx += 1) {
            const list = token_lists.items[batch_idx].items;
            var seq_idx: usize = 0;
            while (seq_idx < list.len) : (seq_idx += 1) {
                const token_index: usize = @intCast(list[seq_idx]);
                if (token_index >= self.vocab_size) return error.InvalidToken;

                const row_offset = try std.math.mul(usize, batch_idx, max_seq_len);
                const row_index = try std.math.add(usize, row_offset, seq_idx);
                const base_idx = try std.math.mul(usize, row_index, self.model_dim);
                const final_idx = try std.math.add(usize, base_idx, token_index);
                if (final_idx >= input_f32_data.len) return error.IndexOutOfBounds;

                input_f32_data[final_idx] = 1.0;
            }
        }

        var output = try self.accelerator.forward(input_f32_data);
        defer output.free(&self.accelerator.ctx);

        const output_data = try output.toSlice(&self.accelerator.ctx, self.allocator);
        defer self.allocator.free(output_data);

        var loss: f32 = 0.0;
        var count: usize = 0;
        for (output_data, input_f32_data) |pred, target| {
            const diff = pred - target;
            loss += diff * diff;
            count += 1;
        }
        if (count > 0) loss /= @floatFromInt(count);

        const grad_data = try self.allocator.alloc(f32, data_elements);
        defer self.allocator.free(grad_data);
        const inv_count: f32 = if (count > 0) 2.0 / @as(f32, @floatFromInt(count)) else 0.0;
        for (output_data, input_f32_data, 0..) |pred, target, idx| {
            grad_data[idx] = (pred - target) * inv_count;
        }

        var bwd = try self.accelerator.backward(grad_data, input_f32_data);
        defer bwd.free(&self.accelerator.ctx);

        const grad_ws = try bwd.grad_ws.toSlice(&self.accelerator.ctx, self.allocator);
        defer self.allocator.free(grad_ws);
        const grad_wt = try bwd.grad_wt.toSlice(&self.accelerator.ctx, self.allocator);
        defer self.allocator.free(grad_wt);

        const weights_s_before = try self.readWeightsFlat(&self.accelerator.weights_s);
        defer self.allocator.free(weights_s_before);
        const weights_t_before = try self.readWeightsFlat(&self.accelerator.weights_t);
        defer self.allocator.free(weights_t_before);

        const lr = self.learning_rate;
        for (grad_ws, 0..) |g, i| {
            grad_ws[i] = -lr * g;
        }
        for (grad_wt, 0..) |g, i| {
            grad_wt[i] = -lr * g;
        }

        try self.averageDeltaInPlace(grad_ws);
        try self.averageDeltaInPlace(grad_wt);

        try self.applyDelta(weights_s_before, grad_ws, .s);
        try self.applyDelta(weights_t_before, grad_wt, .t);
        try self.accelerator.sync();

        return loss;
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            return;
        }

        try self.coordinator.synchronize();
        try self.accelerator.sync();

        const file = std.fs.createFileAbsolute(path, .{ .mode = 0o600 }) catch |err| {
            std.debug.print("Failed to create checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var buffered_writer = std.io.bufferedWriter(file.writer());
        var writer = buffered_writer.writer();

        try writer.writeInt(u32, self.config.checkpoint_version, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.global_step)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.learning_rate)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.momentum)), .little);

        const weights_s_flat = try self.readWeightsFlat(&self.accelerator.weights_s);
        defer self.allocator.free(weights_s_flat);

        for (weights_s_flat) |weight| {
            try writer.writeInt(u32, @as(u32, @bitCast(weight)), .little);
        }

        const weights_t_flat = try self.readWeightsFlat(&self.accelerator.weights_t);
        defer self.allocator.free(weights_t_flat);

        for (weights_t_flat) |weight| {
            try writer.writeInt(u32, @as(u32, @bitCast(weight)), .little);
        }

        try buffered_writer.flush();
        try file.sync();

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const file = std.fs.openFileAbsolute(path, .{ .mode = .read_only }) catch |err| {
            std.debug.print("Failed to open checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var reader = file.reader();

        const version = try reader.readInt(u32, .little);
        if (version != self.config.checkpoint_version) {
            return error.CheckpointVersionMismatch;
        }

        self.global_step = @intCast(try reader.readInt(u64, .little));
        const saved_model_dim = try reader.readInt(u64, .little);
        const saved_learning_rate_bits = try reader.readInt(u32, .little);
        const saved_momentum_bits = try reader.readInt(u32, .little);

        if (saved_model_dim != self.model_dim) {
            return error.ModelDimMismatch;
        }

        const saved_learning_rate: f32 = @bitCast(saved_learning_rate_bits);
        const saved_momentum: f32 = @bitCast(saved_momentum_bits);
        if (self.coordinator.world_size > 1 and saved_momentum != 0.0) {
            return error.UnsupportedDistributedMomentum;
        }
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;

        const weight_count = try std.math.mul(usize, self.model_dim, self.model_dim);
        const s_weights = try self.allocator.alloc(f32, weight_count);
        defer self.allocator.free(s_weights);

        for (s_weights) |*w| {
            const bits = try reader.readInt(u32, .little);
            w.* = @bitCast(bits);
        }

        const t_weights = try self.allocator.alloc(f32, weight_count);
        defer self.allocator.free(t_weights);

        for (t_weights) |*w| {
            const bits = try reader.readInt(u32, .little);
            w.* = @bitCast(bits);
        }

        try self.accelerator.setWeightsS(s_weights, self.model_dim, self.model_dim);
        try self.accelerator.setWeightsT(t_weights, self.model_dim, self.model_dim);
        try self.accelerator.sync();

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }
};