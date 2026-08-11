"""Mocked multi-item contracts for all isolated inference adapters."""

import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from conftest import make_model

from peste.adapters import create_adapter
from peste.adapters.base import AdapterOutputError
from peste.adapters.nemo import NemoRnntAdapter
from peste.adapters.transformers import (
    TransformersCTCAdapter,
    TransformersCTCOutputError,
    TransformersQwenAdapter,
    TransformersWhisperAdapter,
    _upgrade_legacy_whisper_generation_config,
)


class FakeTensor:
    def __init__(
        self,
        value: Any = None,
        shape: tuple[int, ...] = (2, 4),
        *,
        floating: bool = True,
        dtype: str = "fp32",
    ) -> None:
        self.value = value
        self.shape = shape
        self.floating = floating
        self.dtype = dtype
        self.moves: list[tuple[Any, Any]] = []

    def to(self, device: Any, dtype: Any = None) -> "FakeTensor":
        self.moves.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype
        return self

    def is_floating_point(self) -> bool:
        return self.floating

    def __getitem__(self, key: Any) -> "FakeTensor":
        if isinstance(self.value, list):
            return FakeTensor(value=self.value[key])
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.value


class FakeBatch(dict[str, FakeTensor]):
    def to(self, device: Any, dtype: Any = None) -> "FakeBatch":
        for value in self.values():
            value.to(device, dtype if value.is_floating_point() else None)
        return self


class FakeGenerated:
    def __init__(self, count: int = 2) -> None:
        self.rows = [FakeTensor(value=index) for index in range(count)]

    def __iter__(self) -> Any:
        return iter(self.rows)


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] = {}
        self.device = "cuda"
        self.dtype = "bf16"
        self.eval_called = False
        self.generation_config = SimpleNamespace(
            lang_to_id={"<|fa|>": 1}, task_to_id={"transcribe": 2}
        )

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, *args: Any, **kwargs: Any) -> FakeGenerated:
        self.generate_kwargs = kwargs
        source = kwargs.get("input_ids", kwargs.get("input_features", FakeTensor()))
        count = source.shape[0]
        return FakeGenerated(count)

    def parameters(self) -> list[Any]:
        return [SimpleNamespace(numel=lambda: 10)]


class Factory:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        self.args = args
        self.kwargs = kwargs
        return self.value


def _fake_torch(monkeypatch: Any) -> ModuleType:
    module = ModuleType("torch")
    module.float16 = "fp16"  # type: ignore[attr-defined]
    module.bfloat16 = "bf16"  # type: ignore[attr-defined]
    module.float32 = "fp32"  # type: ignore[attr-defined]
    module.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def _make_snapshot(model: Any, cache_directory: Path) -> Path:
    snapshot = (
        cache_directory
        / f"models--{model.repository.replace('/', '--')}"
        / "snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    return snapshot


def _audio_batch(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for index, path in enumerate(paths, start=1):
        sf.write(path, np.zeros(160 * index, dtype=np.float32), 16_000)
    return paths


def test_whisper_uses_padded_ordered_batch(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    model = FakeModel()

    class Processor:
        def __call__(self, audio: Any, **kwargs: Any) -> FakeBatch:
            self.audio = audio
            self.kwargs = kwargs
            return FakeBatch(
                input_features=FakeTensor(shape=(len(audio), 4)),
                attention_mask=FakeTensor(floating=False, dtype="int64"),
            )

        def batch_decode(self, output: FakeGenerated, skip_special_tokens: bool) -> list[str]:
            labels = ("اول", "دوم")
            return [labels[row.value] for row in output.rows]

    processor = Processor()
    transformers = ModuleType("transformers")
    model_factory = Factory(model)
    processor_factory = Factory(processor)
    transformers.AutoModelForSpeechSeq2Seq = model_factory  # type: ignore[attr-defined]
    transformers.AutoProcessor = processor_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = make_model()
    snapshot = _make_snapshot(spec, tmp_path)
    adapter = TransformersWhisperAdapter(spec, tmp_path)
    adapter.load()

    results = adapter.transcribe_batch(_audio_batch(tmp_path))

    assert [result.text for result in results] == ["اول", "دوم"]
    assert processor.kwargs == {
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "padding": True,
    }
    assert len(processor.audio) == 2
    assert model_factory.args == (str(snapshot),)
    assert model_factory.kwargs["dtype"] == "fp16"
    assert model.generate_kwargs["input_features"].moves == [("cuda", "fp16")]
    assert model.generate_kwargs["attention_mask"].moves == [("cuda", None)]
    assert adapter.transcribe_batch(_audio_batch(tmp_path)[:1]) == results[:1]


def test_whisper_rejects_output_cardinality(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    adapter = TransformersWhisperAdapter(make_model(), tmp_path)
    adapter.model = FakeModel()

    class Processor:
        def __call__(self, *args: Any, **kwargs: Any) -> FakeBatch:
            return FakeBatch(input_features=FakeTensor())

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            return ["only one"]

    adapter.processor = Processor()
    with pytest.raises(AdapterOutputError, match="2 audio paths but returned 1"):
        adapter.transcribe_batch(_audio_batch(tmp_path))


def test_legacy_whisper_generation_config_is_completed() -> None:
    token_ids = {"<|fa|>": 10, "<|transcribe|>": 20, "<|notimestamps|>": 30}
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: token_ids[token],
        unk_token_id=0,
    )
    model = SimpleNamespace(generation_config=SimpleNamespace())
    _upgrade_legacy_whisper_generation_config(
        SimpleNamespace(tokenizer=tokenizer), model, "fa", "legacy-whisper"
    )
    assert model.generation_config.lang_to_id == {"<|fa|>": 10}
    assert model.generation_config.task_to_id == {"transcribe": 20}


def test_qwen_uses_native_padded_batch_and_ordered_decode(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    model = FakeModel()

    class Processor:
        def __init__(self) -> None:
            self.decoded: list[int] = []

        def apply_transcription_request(self, **kwargs: Any) -> FakeBatch:
            self.request_kwargs = kwargs
            self.batch = FakeBatch(
                input_ids=FakeTensor(shape=(len(kwargs["audio"]), 4), floating=False),
                audio=FakeTensor(shape=(len(kwargs["audio"]), 4)),
            )
            return self.batch

        def decode(self, output: FakeTensor, **kwargs: Any) -> str:
            self.decode_kwargs = kwargs
            self.decoded.append(output.value)
            return ("اول", "دوم")[output.value]

    processor = Processor()
    transformers = ModuleType("transformers")
    model_factory = Factory(model)
    transformers.AutoModelForMultimodalLM = model_factory  # type: ignore[attr-defined]
    transformers.AutoProcessor = Factory(processor)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = make_model("transformers-qwen", dtype="bfloat16")
    _make_snapshot(spec, tmp_path)
    adapter = TransformersQwenAdapter(spec, tmp_path)
    adapter.load()
    paths = _audio_batch(tmp_path)

    results = adapter.transcribe_batch(paths)

    assert [result.text for result in results] == ["اول", "دوم"]
    assert processor.request_kwargs == {
        "audio": [str(path) for path in paths],
        "language": "fa",
        "padding": True,
    }
    assert processor.decode_kwargs == {"return_format": "transcription_only"}
    assert model.generate_kwargs["max_new_tokens"] == 256
    assert processor.batch["input_ids"].moves == [("cuda", None)]
    assert processor.batch["audio"].moves == [("cuda", "bf16")]
    assert adapter.transcribe_batch(paths[:1]) == results[:1]


def test_qwen_rejects_output_cardinality(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)

    class Model(FakeModel):
        def generate(self, *args: Any, **kwargs: Any) -> FakeGenerated:
            return FakeGenerated(1)

    class Processor:
        def apply_transcription_request(self, **kwargs: Any) -> FakeBatch:
            return FakeBatch(input_ids=FakeTensor(shape=(2, 4), floating=False))

        def decode(self, output: FakeTensor, **kwargs: Any) -> str:
            return "only one"

    adapter = TransformersQwenAdapter(make_model("transformers-qwen"), tmp_path)
    adapter.model = Model()
    adapter.processor = Processor()
    with pytest.raises(AdapterOutputError, match="2 audio paths but returned 1"):
        adapter.transcribe_batch([tmp_path / "first.wav", tmp_path / "second.wav"])


def test_transformers_ctc_pads_masks_moves_dtype_and_decodes_in_order(
    monkeypatch: Any, tmp_path: Path
) -> None:
    torch = _fake_torch(monkeypatch)
    inference_state = {"active": False}

    @contextlib.contextmanager
    def inference_mode() -> Any:
        inference_state["active"] = True
        try:
            yield
        finally:
            inference_state["active"] = False

    predicted_rows = [FakeTensor(value=[10, 11, 12, 13]), FakeTensor(value=[20, 21, 22, 23])]
    predicted_ids = predicted_rows
    torch.inference_mode = inference_mode  # type: ignore[attr-defined]
    torch.argmax = lambda tensor, dim: predicted_ids  # type: ignore[attr-defined]
    input_values = FakeTensor(dtype="source-fp32")
    attention_mask = FakeTensor(value=[4, 2], floating=False, dtype="int64")
    attention_mask.sum = lambda dim: FakeTensor(value=[4, 2])  # type: ignore[attr-defined]

    class Model(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = "fp32"

        def __call__(self, **kwargs: Any) -> Any:
            self.call_kwargs = kwargs
            self.inference_seen = inference_state["active"]
            return SimpleNamespace(logits=FakeTensor(value="logits"))

        def _get_feat_extract_output_lengths(self, lengths: FakeTensor) -> FakeTensor:
            self.input_lengths = lengths
            return FakeTensor(value=[4, 2])

    class Processor:
        def __call__(self, audio: Any, **kwargs: Any) -> dict[str, FakeTensor]:
            self.audio = audio
            self.batch_size = len(audio)
            self.call_kwargs = kwargs
            return {"input_values": input_values, "attention_mask": attention_mask}

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            self.decode_args = args
            self.decode_kwargs = kwargs
            return ["اول <unk>", "دوم"][: self.batch_size]

    model = Model()
    processor = Processor()
    transformers = ModuleType("transformers")
    transformers.AutoModelForCTC = Factory(model)  # type: ignore[attr-defined]
    transformers.AutoProcessor = Factory(processor)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = make_model("transformers-ctc", dtype="float32")
    _make_snapshot(spec, tmp_path)
    adapter = TransformersCTCAdapter(spec, tmp_path)
    adapter.load()

    results = adapter.transcribe_batch(_audio_batch(tmp_path))

    assert [result.text for result in results] == ["اول <unk>", "دوم"]
    assert processor.call_kwargs == {
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "padding": True,
    }
    assert len(processor.audio) == 2
    assert input_values.moves == [("cuda", "fp32")]
    assert attention_mask.moves == [("cuda", None)]
    assert model.call_kwargs == {
        "input_values": input_values,
        "attention_mask": attention_mask,
    }
    assert model.inference_seen is True
    assert processor.decode_args == ([[10, 11, 12, 13], [20, 21]],)
    assert adapter.transcribe_batch(_audio_batch(tmp_path)[:1]) == results[:1]


@pytest.mark.parametrize("decoded", [[], ["one"], ["one", "two", "three"]])
def test_transformers_ctc_rejects_invalid_output_cardinality(
    monkeypatch: Any, tmp_path: Path, decoded: list[str]
) -> None:
    torch = _fake_torch(monkeypatch)
    torch.argmax = lambda tensor, dim: [FakeTensor(value=[1]), FakeTensor(value=[2])]  # type: ignore[attr-defined]
    adapter = TransformersCTCAdapter(make_model("transformers-ctc", dtype="float32"), tmp_path)
    adapter.model = SimpleNamespace(
        device="cuda",
        dtype="fp32",
        __call__=lambda **kwargs: SimpleNamespace(logits=FakeTensor()),
    )

    class CallableModel:
        device = "cuda"
        dtype = "fp32"

        def __call__(self, **kwargs: Any) -> Any:
            return SimpleNamespace(logits=FakeTensor())

        def _get_feat_extract_output_lengths(self, lengths: FakeTensor) -> FakeTensor:
            return FakeTensor(value=[1, 1])

    adapter.model = CallableModel()

    class Processor:
        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, FakeTensor]:
            attention_mask = FakeTensor(floating=False)
            attention_mask.sum = lambda dim: FakeTensor(value=[1, 1])  # type: ignore[attr-defined]
            return {"input_values": FakeTensor(), "attention_mask": attention_mask}

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            return decoded

    adapter.processor = Processor()
    with pytest.raises(TransformersCTCOutputError, match="expected 2 decoded results"):
        adapter.transcribe_batch(_audio_batch(tmp_path))


def test_transformers_ctc_registry_selection(tmp_path: Path) -> None:
    adapter = create_adapter(make_model("transformers-ctc", dtype="float32"), tmp_path)
    assert isinstance(adapter, TransformersCTCAdapter)


def test_nemo_uses_native_batch_and_checks_cardinality(monkeypatch: Any, tmp_path: Path) -> None:
    checkpoint = tmp_path / "models--organization--model" / "snapshots" / ("a" * 40) / "model.nemo"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    class NemoModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.disable_cuda_graphs_called = False
            self.decoding = SimpleNamespace(
                decoding=SimpleNamespace(disable_cuda_graphs=self.disable_cuda_graphs)
            )

        def disable_cuda_graphs(self) -> bool:
            self.disable_cuda_graphs_called = True
            return True

        def change_decoding_strategy(self, **kwargs: Any) -> None:
            self.decoder_kwargs = kwargs

        def transcribe(self, paths: list[str], batch_size: int, verbose: bool) -> list[Any]:
            self.transcribe_args = (paths, batch_size, verbose)
            return [SimpleNamespace(text="اول"), SimpleNamespace(text="دوم")][: len(paths)]

    model = NemoModel()

    class Factory:
        def restore_from(self, *args: Any, **kwargs: Any) -> NemoModel:
            self.args = args
            self.kwargs = kwargs
            return model

    factory = Factory()
    module = ModuleType("nemo.collections.asr.models")
    module.ASRModel = factory  # type: ignore[attr-defined]
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = NemoRnntAdapter(make_model("nemo-rnnt", dtype="float32"), tmp_path)
    adapter.load()
    paths = _audio_batch(tmp_path)

    results = adapter.transcribe_batch(paths)

    assert [result.text for result in results] == ["اول", "دوم"]
    assert model.decoder_kwargs == {"decoder_type": "rnnt"}
    assert model.disable_cuda_graphs_called
    assert model.transcribe_args == ([str(path) for path in paths], 2, False)
    assert factory.kwargs["map_location"] == "cuda"
    assert adapter.transcribe_batch(paths[:1]) == results[:1]
    model.transcribe = lambda paths, batch_size, verbose: [SimpleNamespace(text="only")]
    with pytest.raises(AdapterOutputError, match="2 audio paths but returned 1"):
        adapter.transcribe_batch(paths)
