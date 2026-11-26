"""Merged-LoRA inference measurement, shared by all downstream evaluations."""
import json
import time
from datetime import datetime
from pathlib import Path

import torch

MB = 1024 ** 2


def add_inference_arguments(parser):
    parser.add_argument('--warmup_batches', type=int, default=3,
                        help='Warmup forwards on the first batch; excluded from inference measurements.')


def output_directory(root, task, requested=None):
    path = Path(requested).expanduser() if requested else Path('results') / f'{task}_inference_{datetime.now():%Y%m%d_%H%M%S_%f}'
    path = (path if path.is_absolute() else Path(root) / path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_for_inference(model):
    """Merge in memory only; never write to the original base/adapter files."""
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    merged = isinstance(model, PeftModel)
    model.eval()
    if merged:
        model = model.merge_and_unload(safe_merge=True)
        if any(isinstance(layer, BaseTunerLayer) for layer in model.modules()):
            raise RuntimeError('LoRA tuner layers remain after merge_and_unload')
        print('LoRA merged into base model in memory (merge_and_unload).')
    model.eval()
    model._inference_lora_merged = merged
    return model


class InferenceMeter:
    def __init__(self, model, device, batch_size, warmup_batches=3, **metadata):
        if batch_size <= 0 or warmup_batches < 0:
            raise ValueError('batch_size must be positive; warmup_batches must be nonnegative')
        self.model = model
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.metadata = metadata
        self.result = None

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _logits(output):
        return output.logits if hasattr(output, 'logits') else output

    def predict(self, loader, prepare, input_key='input_ids'):
        """Yield original CPU batch and CPU logits; each sample is scored exactly once.

        CUDA events time forward only. Pipeline wall time also covers loader/tokenization,
        host/device copies and CPU prediction collection. Metrics and file exports are excluded.
        """
        self.model.eval()
        # Separate iterator: warmup does not consume any measured evaluation examples.
        try:
            first = next(iter(loader))
        except StopIteration:
            raise ValueError('Cannot benchmark an empty evaluation dataset')
        with torch.inference_mode():
            kwargs = prepare(first)
            for _ in range(self.warmup_batches):
                output = self.model(**kwargs)
                del output
            del kwargs, first
            self._sync()
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            baseline = torch.cuda.memory_allocated(self.device) / MB if self.cuda else None
            events = []
            seconds = 0.0
            samples = batches = tokens = 0
            wall_start = time.perf_counter()
            for batch in loader:
                kwargs = prepare(batch)
                if self.cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(self.device))
                else:
                    start = time.perf_counter()
                output = self.model(**kwargs)
                if self.cuda:
                    end.record(torch.cuda.current_stream(self.device))
                    events.append((start, end))
                else:
                    seconds += time.perf_counter() - start
                logits = self._logits(output).detach().float().cpu()
                samples += len(batch[input_key])
                batches += 1
                tokens += batch[input_key].numel()
                del output, kwargs
                yield batch, logits
                del logits
            self._sync()
            wall_seconds = time.perf_counter() - wall_start
            if self.cuda:
                seconds = sum(start.elapsed_time(end) for start, end in events) / 1000.0
            if seconds <= 0:
                raise RuntimeError('Nonpositive forward timing')
            self.result = dict(
                self.metadata,
                device=str(self.device),
                gpu_name=torch.cuda.get_device_name(self.device) if self.cuda else None,
                precision=str(next(self.model.parameters()).dtype),
                lora_merged=bool(getattr(self.model, '_inference_lora_merged', False)),
                batch_size=self.batch_size, warmup_batches=self.warmup_batches,
                samples=samples, batches=batches, padded_input_tokens=tokens,
                inference_time_s=seconds,
                throughput_batch_per_s=batches / seconds,
                throughput_samples_per_s=samples / seconds,
                inference_pipeline_time_s=wall_seconds,
                pipeline_throughput_batch_per_s=batches / wall_seconds,
                pipeline_throughput_samples_per_s=samples / wall_seconds,
                peak_allocated_mb=torch.cuda.max_memory_allocated(self.device) / MB if self.cuda else None,
                peak_reserved_mb=torch.cuda.max_memory_reserved(self.device) / MB if self.cuda else None,
                model_baseline_allocated_mb=baseline,
                torch_version=str(torch.__version__),
                timing_scope='Forward only: CUDA events or CPU wall clock; excludes loading, merging, warmup, input transfers, metrics, exports',
                pipeline_scope='Whole measured data loop: loading/tokenization, transfers, forward, CPU output collection; excludes model loading/merging/warmup/metrics/file export',
                memory_scope='Peak current-process PyTorch allocator after warmup, including model weights; excludes CUDA context and other processes; reserved includes cached allocations',
            )

    def save(self, output_dir):
        if self.result is None:
            raise RuntimeError('Consume the complete prediction iterator before saving measurements')
        path = Path(output_dir) / 'inference_efficiency.json'
        path.write_text(json.dumps(self.result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        def mb(value):
            return 'N/A (CPU)' if value is None else f'{value:.1f} MB'
        print('\n===== 推理效率（预热后，仅本进程）=====', flush=True)
        print(f"LoRA merged: {self.result['lora_merged']}; precision: {self.result['precision']}; batch: {self.batch_size}")
        print(f"纯前向: {self.result['inference_time_s']:.3f}s; "
              f"{self.result['throughput_batch_per_s']:.3f} batch/s; "
              f"{self.result['throughput_samples_per_s']:.1f} samples/s")
        print(f"含取数据/传输/收集预测: {self.result['inference_pipeline_time_s']:.3f}s; "
              f"{self.result['pipeline_throughput_samples_per_s']:.1f} samples/s")
        print(f"显存峰值: allocated {mb(self.result['peak_allocated_mb'])}; reserved {mb(self.result['peak_reserved_mb'])}")
        print('Inference measurements:', path, flush=True)
        return self.result
