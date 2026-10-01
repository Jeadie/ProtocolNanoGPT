# ===== FROZEN: the dataloader — do not edit between these markers =====
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos:][:batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
# ===== END FROZEN =====
# ===== FROZEN: the wire frame — do not edit between these markers =====
class _WirePack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t: Tensor):
        # compress activations -> (payload, count), sent to the NEXT stage
        ctx.meta = (t.shape, t.dtype)
        payload, n = pack(t, backward=False)
        return payload, torch.tensor([float(n)], dtype=torch.float64, device=t.device)

    @staticmethod
    def backward(ctx, g_payload: Tensor, g_used: Tensor):
        # decompress the activation gradients received FROM the next stage
        # (the meter already zeroed g_payload past g_used)
        return unpack(g_payload, ctx.meta, backward=True)

class _WireUnpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, payload: Tensor, used: Tensor, meta):
        # decompress the activations received FROM the previous stage
        # (`used` was already consumed by the meter: zero + bill)
        return unpack(payload, meta, backward=False)

    @staticmethod
    def backward(ctx, g: Tensor):
        # compress activation gradients -> (payload, count), sent to the PREVIOUS stage
        payload, n = pack(g, backward=True)   # the count travels as the gradient of `used`
        return payload, torch.tensor([float(n)], dtype=torch.float64, device=g.device), None

# Keep out of torch.compile: the compiler (Inductor) cannot generate code for byte-level
# dtype reinterpretation (view(torch.uint8) <-> view(torch.bfloat16)) or data-dependent sizes.
@torch.compiler.disable
def wire_pack(t: Tensor):
    return _WirePack.apply(t)

@torch.compiler.disable
def wire_unpack(x: Tensor, used: Tensor, meta) -> Tensor:
    return _WireUnpack.apply(x, used, meta)
# ===== END FROZEN =====
            # ===== FROZEN: the wire recv — do not edit this line =====
            x = wire_unpack(x, used, self.wire_meta)
            # ===== END FROZEN =====
        # ===== FROZEN: the wire send — do not edit this line =====
        return wire_pack(x)
        # ===== END FROZEN =====
# ===== FROZEN: the meter — do not edit between these markers =====
WAN_MODE = os.environ.get("WAN_MODE", "off")
assert WAN_MODE in ("on", "off"), f"WAN_MODE must be 'on' or 'off', got {WAN_MODE!r}"
LINK_BANDWIDTH = 200e6
LINK_LATENCY = 0.05
simulating = WAN_MODE == "on"
TARGET_LOSS = 3.276

class SimPipelineStage(PipelineStage):
    """Receiver-side link simulation, schedule-agnostic via the overridden _retrieve_recv_* methods. A message ending in a 1D float64 counts vector is a frame: the meter zeroes each payload past its count and bills only the counts (_billable), so under-declaring destroys the sender's own data. Each message carries an 8-byte sent_at stamp so the delay is billed from the sender's clock, letting compute and communication overlap on the receive side."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrives_at = 0.0     # wall clock at which the last message on this rank's inbound wire lands
        self.sent_at = {}         # chunk key -> float64[1] CUDA buffer holding the sender's wall clock
        self.wire_bits = 0        # payload bits billed against this rank's inbound wire

    def _with_sent_at(self, ops, key=None):
        if simulating and ops:
            t = (torch.tensor([time.time()], dtype=torch.float64, device=self.device) if key is None
                 else self.sent_at.setdefault(key, torch.empty(1, dtype=torch.float64, device=self.device)))
            ops = ops + [dist.P2POp(ops[0].op, t, group=self.group, group_peer=ops[0].group_peer)]
        return ops

    def get_fwd_send_ops(self, fwd_chunk_id):
        return self._with_sent_at(super().get_fwd_send_ops(fwd_chunk_id))

    def get_bwd_send_ops(self, bwd_chunk_id):
        return self._with_sent_at(super().get_bwd_send_ops(bwd_chunk_id))

    def get_fwd_recv_ops(self, fwd_chunk_id):
        return self._with_sent_at(super().get_fwd_recv_ops(fwd_chunk_id), ("f", fwd_chunk_id))

    def get_bwd_recv_ops(self, bwd_chunk_id):
        return self._with_sent_at(super().get_bwd_recv_ops(bwd_chunk_id), ("b", bwd_chunk_id))

    def _billable(self, tensors):
        framed = (len(tensors) >= 2 and tensors[-1] is not None
                  and tensors[-1].dtype == torch.float64 and tensors[-1].ndim == 1
                  and tensors[-1].numel() == len(tensors) - 1)
        if not framed:
            return sum(t.element_size() * t.numel() for t in tensors if t is not None)
        *slots, used = tensors
        with torch.no_grad():
            counts = used.detach().to(torch.int64).tolist()
            total = used.numel() * used.element_size()
            for t, k in zip(slots, counts):
                if t is None:
                    continue
                flat = t.detach().view(torch.uint8).view(-1)
                k = min(max(int(k), 0), flat.numel())
                flat[k:].zero_()   # destroy every byte past the declared count: only billed bytes are readable
                total += k
        return total

    def add_delay(self, nbytes, key):
        evt = torch.cuda.Event(); evt.record(); evt.synchronize()   # payload and sent_at have landed
        total_bits = 8 * nbytes
        self.wire_bits += total_bits
        sent_at = self.sent_at.pop(key).item() if key in self.sent_at else time.time()
        starts_at = max(self.arrives_at, sent_at)        # data is ready AND the wire is free
        self.arrives_at = starts_at + total_bits / LINK_BANDWIDTH + LINK_LATENCY
        time.sleep(max(0.0, self.arrives_at - time.time()))

    def _retrieve_recv_activations(self, fwd_chunk_id):
        acts = super()._retrieve_recv_activations(fwd_chunk_id)
        nbytes = self._billable(list(acts))
        if simulating:
            self.add_delay(nbytes, ("f", fwd_chunk_id))
        return acts

    def _retrieve_recv_grads(self, bwd_chunk_id):
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        nbytes = self._billable(list(grads))
        if simulating:
            self.add_delay(nbytes, ("b", bwd_chunk_id))
        return grads

    def forward_one_chunk(self, fwd_chunk_id, *args, **kwargs):
        out = super().forward_one_chunk(fwd_chunk_id, *args, **kwargs)
        if simulating:
            # 1F1B's mixed send/recv batching stalls the pipe without this
            evt = torch.cuda.Event(); evt.record(); evt.synchronize()
        return out

    def backward_one_chunk(self, bwd_chunk_id, *args, **kwargs):
        out = super().backward_one_chunk(bwd_chunk_id, *args, **kwargs)
        if simulating:
            evt = torch.cuda.Event(); evt.record(); evt.synchronize()
        return out

# ===== END FROZEN =====
# ===== FROZEN: self-logging — do not edit between these markers =====
# rule 4's mechanism: the logfile must contain everything needed to reproduce the run,
# starting with this file's own source
if rank == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if rank == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)
print0(code)
# ===== END FROZEN =====
# ===== FROZEN: problem size — do not edit between these markers =====
val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024 # tokens per step (512 sequences)
assert world_size == 8, "the model is cut across eight nodes; fewer ranks means fewer metered boundaries"
# ===== END FROZEN =====
# ===== FROZEN: the data — do not edit this line =====
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))
# ===== END FROZEN =====
# ===== FROZEN: parameter cap — do not edit between these markers =====
BASELINE_PARAMS = 203_828_352
num_params = sum(p.numel() for p in model.parameters())
print0(f"total params: {num_params:,} (baseline {BASELINE_PARAMS:,})", console=True)
assert num_params <= BASELINE_PARAMS, f"rule 1: {num_params:,} params exceeds the baseline {BASELINE_PARAMS:,}"
# ===== END FROZEN =====
# ===== FROZEN: meter wrapper — do not edit between these markers =====
stage = SimPipelineStage(stage_mod, rank, world_size, device, input_args=stage_in, output_args=stage_out)
# ===== END FROZEN =====
    # ===== FROZEN: data + clock — do not edit between these markers =====
    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    # start the clock
    training_time = 0
    last_val_step = 0
    windows = []  # per-window step_avg of the timing run
    dist.barrier()
    t0 = time.perf_counter()
    # ===== END FROZEN =====
        # ===== FROZEN: val cadence + clock stop — do not edit between these markers =====
        val_step_freq = 10 if WAN_MODE == "on" else (125 if step / train_steps < 0.9 else 25)
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            simulating = False  # clock is stopped: run val at full link speed
            # ===== END FROZEN =====
            # ===== FROZEN: scoring + clock restart — do not edit between these markers =====
            val_loss /= val_tokens
            dist.broadcast(val_loss, src=world_size - 1)  # loss materializes on the last rank
            print0(f"step:{step}/{train_steps} val_loss:{val_loss.item():.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            if step == train_steps:
                reached = "REACHED" if float(val_loss) <= TARGET_LOSS else "MISSED"
                print0(f"TARGET {TARGET_LOSS} {reached} final val_loss:{float(val_loss):.5f}",
                       console=True)
            # steady state: the three latest windows agree within 1% (window 1 is compile and never counts).
            if WAN_MODE == "on" and step > val_step_freq:
                windows.append(step_avg)
                wire = torch.tensor([float(stage.wire_bits)], dtype=torch.float64, device=device)
                dist.all_reduce(wire)   # each rank counts only its own inbound wire
                if len(windows) >= 3 and max(windows[-3:]) / min(windows[-3:]) - 1 < 0.01:
                    steady = sorted(windows[-3:])[1]
                    T = train_steps * steady
                    bytes_per_token = wire.item() / 8 / (world_size - 1) / (batch_size * step)
                    print0(f"STEADY at step:{step} step_avg:{1000*steady:.2f}ms"
                           + f" bytes_per_token:{bytes_per_token:.1f}")
                    print0(f"T = {train_steps} steps x {steady:.4f}s = {T:.1f}s", console=True)
                    break
            # start the clock again
            simulating = WAN_MODE == "on"
            dist.barrier()
            t0 = time.perf_counter()
            # ===== END FROZEN =====
