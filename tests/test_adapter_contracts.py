"""Mocked contract checks for all isolated inference adapters."""

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
from peste.adapters.nemo import NemoRnntAdapter
from peste.adapters.transformers import (
    TransformersCTCAdapter,
    TransformersCTCOutputError,
    TransformersQwenAdapter,
    TransformersWhisperAdapter,
    _upgrade_legacy_whisper_generation_config,
)
from peste.adapters.vibevoice import VibeVoiceAdapter
from peste.prefetch import VIBEVOICE_TOKENIZER_REPOSITORY, VIBEVOICE_TOKENIZER_REVISION


class FakeTensor:
    def __init__(
        self,
        value: Any = None,
        shape: tuple[int, ...] = (1, 2),
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


class FakeBatch(dict[str, FakeTensor]):
    def to(self, device: Any, dtype: Any = None) -> "FakeBatch":
        for value in self.values():
            value.to(device, dtype)
        return self


class FakeGenerated:
    def __getitem__(self, key: Any) -> "FakeGenerated":
        return self


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
        return FakeGenerated()

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


def test_whisper_contract(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    model = FakeModel()
    processor = SimpleNamespace(
        __call__=lambda *args, **kwargs: None,
        batch_decode=lambda output, skip_special_tokens: ["متن"],
    )

    class CallableProcessor:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(input_features=FakeTensor())

        def batch_decode(self, output: Any, skip_special_tokens: bool) -> list[str]:
            return ["متن"]

    callable_processor = CallableProcessor()
    transformers = ModuleType("transformers")
    model_factory = Factory(model)
    processor_factory = Factory(callable_processor)
    transformers.AutoModelForSpeechSeq2Seq = model_factory  # type: ignore[attr-defined]
    transformers.AutoProcessor = processor_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(160, dtype=np.float32), 16_000)
    spec = make_model()
    snapshot = _make_snapshot(spec, tmp_path)
    adapter = TransformersWhisperAdapter(spec, tmp_path)
    adapter.load()
    assert adapter.transcribe(audio).text == "متن"
    assert model_factory.kwargs["dtype"] == "fp16"
    assert processor_factory.kwargs == {"local_files_only": True}
    assert processor_factory.args == (str(snapshot),)
    assert model.generate_kwargs == {
        "language": "fa",
        "task": "transcribe",
        "max_new_tokens": 444,
        "return_timestamps": False,
    }
    assert callable_processor.kwargs["sampling_rate"] == 16_000
    del processor


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

    assert model.generation_config.is_multilingual is True
    assert model.generation_config.lang_to_id == {"<|fa|>": 10}
    assert model.generation_config.task_to_id == {"transcribe": 20}
    assert model.generation_config.no_timestamps_token_id == 30


def test_qwen_contract(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    model = FakeModel()

    class Processor:
        def __init__(self) -> None:
            self.request_kwargs: dict[str, Any] = {}
            self.decode_kwargs: dict[str, Any] = {}

        def apply_transcription_request(self, **kwargs: Any) -> FakeBatch:
            self.request_kwargs = kwargs
            return FakeBatch(input_ids=FakeTensor(shape=(1, 4)), audio=FakeTensor())

        def decode(self, output: Any, **kwargs: Any) -> list[str]:
            self.decode_kwargs = kwargs
            return ["متن"]

    processor = Processor()
    transformers = ModuleType("transformers")
    model_factory = Factory(model)
    transformers.AutoModelForMultimodalLM = model_factory  # type: ignore[attr-defined]
    transformers.AutoProcessor = Factory(processor)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = make_model("transformers-qwen", dtype="bfloat16")
    snapshot = _make_snapshot(spec, tmp_path)
    adapter = TransformersQwenAdapter(spec, tmp_path)
    adapter.load()
    result = adapter.transcribe(tmp_path / "audio.wav")
    assert result.text == "متن"
    assert processor.request_kwargs["language"] == "fa"
    assert processor.decode_kwargs == {"return_format": "transcription_only"}
    assert model.generate_kwargs["max_new_tokens"] == 256
    assert model_factory.kwargs["dtype"] == "bf16"
    assert model_factory.kwargs["local_files_only"] is True
    assert model_factory.args == (str(snapshot),)


def test_transformers_ctc_contract(monkeypatch: Any, tmp_path: Path) -> None:
    torch = _fake_torch(monkeypatch)
    inference_state = {"active": False}
    argmax_call: dict[str, Any] = {}
    predicted_ids = FakeTensor(value="predicted")

    @contextlib.contextmanager
    def inference_mode() -> Any:
        inference_state["active"] = True
        try:
            yield
        finally:
            inference_state["active"] = False

    def argmax(tensor: Any, *, dim: int) -> FakeTensor:
        argmax_call.update(tensor=tensor, dim=dim)
        return predicted_ids

    torch.inference_mode = inference_mode  # type: ignore[attr-defined]
    torch.argmax = argmax  # type: ignore[attr-defined]

    input_values = FakeTensor(dtype="source-fp32")
    attention_mask = FakeTensor(floating=False, dtype="int64")
    logits = FakeTensor(value="logits")

    class CTCModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = "fp32"
            self.call_kwargs: dict[str, Any] = {}
            self.inference_mode_seen = False

        def __call__(self, **kwargs: Any) -> Any:
            self.call_kwargs = kwargs
            self.inference_mode_seen = inference_state["active"]
            return SimpleNamespace(logits=logits)

        def parameters(self) -> list[Any]:
            return [SimpleNamespace(numel=lambda: 10), SimpleNamespace(numel=lambda: 7)]

    class CTCProcessor:
        def __init__(self) -> None:
            self.audio: Any = None
            self.call_kwargs: dict[str, Any] = {}
            self.decode_args: tuple[Any, ...] = ()
            self.decode_kwargs: dict[str, Any] = {}

        def __call__(self, audio: Any, **kwargs: Any) -> dict[str, FakeTensor]:
            self.audio = audio
            self.call_kwargs = kwargs
            return {"input_values": input_values, "attention_mask": attention_mask}

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            self.decode_args = args
            self.decode_kwargs = kwargs
            return ["متن <unk>"]

    model = CTCModel()
    processor = CTCProcessor()
    model_factory = Factory(model)
    processor_factory = Factory(processor)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCTC = model_factory  # type: ignore[attr-defined]
    transformers.AutoProcessor = processor_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(220, dtype=np.float32), 22_050)
    spec = make_model("transformers-ctc", dtype="float32")
    snapshot = _make_snapshot(spec, tmp_path)
    adapter = TransformersCTCAdapter(spec, tmp_path)

    adapter.load()
    result = adapter.transcribe(audio)

    assert result.text == "متن <unk>"
    assert processor_factory.args == (str(snapshot),)
    assert processor_factory.kwargs == {"local_files_only": True}
    assert model_factory.args == (str(snapshot),)
    assert model_factory.kwargs == {
        "dtype": "fp32",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    assert model.device == "cuda"
    assert model.eval_called is True
    assert processor.audio.dtype == np.float32
    assert processor.call_kwargs == {"sampling_rate": 22_050, "return_tensors": "pt"}
    assert input_values.moves == [("cuda", "fp32")]
    assert attention_mask.moves == [("cuda", None)]
    assert attention_mask.dtype == "int64"
    assert model.call_kwargs == {
        "input_values": input_values,
        "attention_mask": attention_mask,
    }
    assert model.inference_mode_seen is True
    assert argmax_call == {"tensor": logits, "dim": -1}
    assert processor.decode_args == (predicted_ids,)
    assert processor.decode_kwargs == {
        "group_tokens": True,
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    assert adapter.parameter_count == 17

    adapter.close()

    assert adapter.model is None
    assert adapter.processor is None
    assert adapter.parameter_count == 0


@pytest.mark.parametrize("decoded", [[], ["first", "second"]])
def test_transformers_ctc_rejects_invalid_output_cardinality(
    monkeypatch: Any, tmp_path: Path, decoded: list[str]
) -> None:
    torch = _fake_torch(monkeypatch)
    torch.argmax = lambda tensor, dim: FakeTensor()  # type: ignore[attr-defined]
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(160, dtype=np.float32), 16_000)
    adapter = TransformersCTCAdapter(make_model("transformers-ctc", dtype="float32"), tmp_path)

    class Model:
        device = "cuda"
        dtype = "fp32"

        def __call__(self, **kwargs: Any) -> Any:
            return SimpleNamespace(logits=FakeTensor())

    adapter.model = Model()

    class Processor:
        def __call__(self, audio: Any, **kwargs: Any) -> dict[str, FakeTensor]:
            return {"input_values": FakeTensor()}

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            return decoded

    adapter.processor = Processor()

    with pytest.raises(TransformersCTCOutputError, match="expected exactly one decoded result"):
        adapter.transcribe(audio)


def test_transformers_ctc_registry_selection(tmp_path: Path) -> None:
    adapter = create_adapter(make_model("transformers-ctc", dtype="float32"), tmp_path)

    assert isinstance(adapter, TransformersCTCAdapter)


def test_vibevoice_contract(monkeypatch: Any, tmp_path: Path) -> None:
    _fake_torch(monkeypatch)
    model = FakeModel()

    class Processor:
        pad_id = 0
        tokenizer = SimpleNamespace(eos_token_id=1)

        def __call__(self, **kwargs: Any) -> FakeBatch:
            self.call_kwargs = kwargs
            return FakeBatch(input_ids=FakeTensor(shape=(1, 3)), audio=FakeTensor())

        def decode(self, output: Any, **kwargs: Any) -> str:
            return "raw"

        def post_process_transcription(self, value: str) -> list[dict[str, str]]:
            return [{"text": "سلام"}, {"text": "دنیا"}]

    processor = Processor()
    model_module = ModuleType("vibevoice.modular.modeling_vibevoice_asr")
    model_factory = Factory(model)
    model_module.VibeVoiceASRForConditionalGeneration = model_factory  # type: ignore[attr-defined]
    processor_module = ModuleType("vibevoice.processor.vibevoice_asr_processor")
    processor_factory = Factory(processor)
    processor_module.VibeVoiceASRProcessor = processor_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vibevoice", ModuleType("vibevoice"))
    monkeypatch.setitem(sys.modules, "vibevoice.modular", ModuleType("vibevoice.modular"))
    monkeypatch.setitem(sys.modules, "vibevoice.processor", ModuleType("vibevoice.processor"))
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)
    tokenizer = (
        tmp_path
        / f"models--{VIBEVOICE_TOKENIZER_REPOSITORY.replace('/', '--')}"
        / "snapshots"
        / VIBEVOICE_TOKENIZER_REVISION
    )
    tokenizer.mkdir(parents=True)
    spec = make_model("vibevoice", dtype="bfloat16")
    model_snapshot = _make_snapshot(spec, tmp_path)
    adapter = VibeVoiceAdapter(spec, tmp_path)
    adapter.load()
    result = adapter.transcribe(tmp_path / "audio.wav")
    assert result.text == "سلام دنیا"
    assert processor_factory.kwargs["language_model_pretrained_name"] == str(tokenizer)
    assert processor.call_kwargs["audio"] == [str(tmp_path / "audio.wav")]
    assert model.generate_kwargs["max_new_tokens"] == 512
    assert model.generate_kwargs["num_beams"] == 1
    assert model_factory.kwargs["attn_implementation"] == "sdpa"
    assert model_factory.args == (str(model_snapshot),)


def test_nemo_contract(monkeypatch: Any, tmp_path: Path) -> None:
    checkpoint = tmp_path / "models--organization--model" / "snapshots" / ("a" * 40) / "model.nemo"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    class NemoModel(FakeModel):
        def change_decoding_strategy(self, **kwargs: Any) -> None:
            self.decoder_kwargs = kwargs

        def transcribe(self, paths: list[str], batch_size: int) -> list[Any]:
            self.transcribe_args = (paths, batch_size)
            return [SimpleNamespace(text="متن")]

    model = NemoModel()
    factory = SimpleNamespace(restore_from=lambda *args, **kwargs: model)
    module = ModuleType("nemo.collections.asr.models")
    module.ASRModel = factory  # type: ignore[attr-defined]
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = NemoRnntAdapter(make_model("nemo-rnnt", dtype="float32"), tmp_path)
    adapter.load()
    assert adapter.transcribe(tmp_path / "audio.wav").text == "متن"
    assert model.decoder_kwargs == {"decoder_type": "rnnt"}
    assert model.transcribe_args == ([str(tmp_path / "audio.wav")], 1)
