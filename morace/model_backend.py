import gc
import math
import os
from morace.generation_runtime import science_env_get
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_ACTIVE_MODEL_KEY = None
_ACTIVE_MODEL = None
_ACTIVE_TOKENIZER = None
_ACTIVE_CONTEXT_LIMIT = None


def _model_path_env_suffix(model_name: str) -> str:
    return model_name.upper().replace(".", "").replace("-", "_")


def _env_value(primary_key: str, legacy_key: str, default: str = "") -> str:
    return science_env_get(primary_key) or science_env_get(legacy_key) or default


def trust_remote_code_enabled() -> bool:
    """Return whether checkpoint-provided Python code may be executed."""
    value = _env_value("MORACE_TRUST_REMOTE_CODE", "LLM4SD_TRUST_REMOTE_CODE", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_torch():
    try:
        import torch
    except (ImportError, NameError) as exc:
        raise RuntimeError(
            "PyTorch could not be imported. Install the pinned requirements and run "
            "`python scripts/verify_repository.py --runtime` before model generation."
        ) from exc
    return torch


def _load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Local model generation requires transformers. Install the dependencies in requirements.txt."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _visible_cuda_device_count() -> int:
    torch = _load_torch()
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        visible = [item.strip() for item in env_value.split(",") if item.strip() and item.strip() != "-1"]
        if visible:
            return len(visible)
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0
    return 0


def normalize_model_name(model_name: str) -> str:
    lowered = (model_name or "").strip().lower()
    if lowered in {"qwen", "qwen3.5", "qwen3.5-9b", "qwen3-9b"}:
        return "qwen3.5-9b"
    if lowered in {"qwen3.5-27b", "qwen35-27b", "qwen3_5_27b"}:
        return "qwen3.5-27b"
    if lowered in {"qwen2.5-3b", "qwen2.5-3b-instruct", "qwen25-3b", "qwen25-3b-instruct"}:
        return "qwen2.5-3b-instruct"
    if lowered in {"qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "qwen25-05b", "qwen25-0.5b-instruct"}:
        return "qwen2.5-0.5b-instruct"
    if lowered in {"chemdfm", "chemdfm-13b", "chemdfm-v1.0-13b"}:
        return "chemdfm"
    if lowered in {"chemdfm-v1.5-8b", "chemdfm-8b", "chemdfm-v15-8b"}:
        return "chemdfm-v1.5-8b"
    if lowered in {"glm-4-9b", "glm-4-9b-chat", "glm4-9b", "chatglm", "chatglm-4-9b"}:
        return "glm-4-9b"
    if lowered in {"gemma-4-e4b-it", "gemma4-e4b-it", "gemma4"}:
        return "gemma-4-e4b-it"
    if lowered in {"gemma-4-e2b-it", "gemma4-e2b-it", "gemma-e2b-it"}:
        return "gemma-4-e2b-it"
    if lowered in {"llama-3-8b-instruct", "llama3-8b", "llama-3-8b", "meta-llama-3-8b-instruct"}:
        return "llama-3-8b-instruct"
    if lowered in {"llama-3.1-8b-instruct", "llama31-8b", "llama-31-8b-instruct"}:
        return "llama-3.1-8b-instruct"
    if lowered in {"deepseek-llm-7b-chat", "deepseek-7b-chat", "deepseekllm-7b-chat", "deepseek-llm7b-chat"}:
        return "deepseek-llm-7b-chat"
    if lowered in {"baichuan2-7b-chat", "baichuan-7b-chat", "baichuan2-7b", "baichuan27bchat"}:
        return "baichuan2-7b-chat"
    if lowered in {"galactica", "galactica-6.7b", "galactica67"}:
        return "galactica-6.7b"
    return lowered


def resolve_model_path(model_name: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return explicit_path
    normalized = normalize_model_name(model_name)
    suffix = _model_path_env_suffix(normalized)
    specific_path = _env_value(f"MORACE_MODEL_PATH_{suffix}", f"LLM4SD_MODEL_PATH_{suffix}")
    if specific_path:
        return specific_path
    generic_path = _env_value("MORACE_MODEL_PATH", "LLM4SD_MODEL_PATH")
    if generic_path:
        return generic_path
    return model_name


def infer_model_family(model_name: str, model_path: str) -> str:
    name = f"{model_name} {model_path}".lower()
    if "qwen" in name:
        return "qwen"
    if "chemdfm" in name:
        return "chemdfm"
    if "glm" in name or "chatglm" in name:
        return "glm"
    if "gemma" in name:
        return "gemma"
    if "llama" in name:
        return "llama"
    if "deepseek" in name:
        return "deepseek"
    if "baichuan" in name:
        return "baichuan"
    if "galactica" in name:
        return "galactica"
    return "generic"


def is_non_qwen_family(model_family: str) -> bool:
    return model_family not in {"qwen"}


def _release_active_model() -> None:
    global _ACTIVE_MODEL_KEY, _ACTIVE_MODEL, _ACTIVE_TOKENIZER, _ACTIVE_CONTEXT_LIMIT
    torch = _load_torch()
    if _ACTIVE_MODEL is not None:
        del _ACTIVE_MODEL
    if _ACTIVE_TOKENIZER is not None:
        del _ACTIVE_TOKENIZER
    _ACTIVE_MODEL_KEY = None
    _ACTIVE_MODEL = None
    _ACTIVE_TOKENIZER = None
    _ACTIVE_CONTEXT_LIMIT = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


@dataclass
class ModelSpec:
    name: str
    path: str


class LocalTextGenerator:
    def __init__(self, model_name: str, model_path: Optional[str] = None, seed: int = 42):
        self.model_name = normalize_model_name(model_name)
        self.model_path = resolve_model_path(self.model_name, explicit_path=model_path)
        self.model_family = infer_model_family(self.model_name, self.model_path)
        self.seed = int(seed)
        self._generation_index = 0
        self.tokenizer = None
        self.model = None
        self.max_context_tokens = 2048

    def _next_generation_seed(self, torch) -> int:
        generation_seed = self.seed + self._generation_index
        self._generation_index += 1
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)
        return generation_seed

    def _infer_context_limit(self) -> int:
        candidates = []
        if self.model is not None and getattr(self.model, "config", None) is not None:
            config = self.model.config
            for field in ("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"):
                value = getattr(config, field, None)
                if isinstance(value, int) and value > 0:
                    candidates.append(value)
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 10**6:
            candidates.append(model_max_length)
        return min(candidates) if candidates else 2048

    def load(self) -> None:
        global _ACTIVE_MODEL_KEY, _ACTIVE_MODEL, _ACTIVE_TOKENIZER, _ACTIVE_CONTEXT_LIMIT
        torch = _load_torch()
        AutoModelForCausalLM, AutoTokenizer = _load_transformers()
        trust_remote_code = trust_remote_code_enabled()
        key = (
            self.model_path, self.model_family, trust_remote_code,
            _env_value("MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_LLAMA_ALLOW_AUTO_DEVICE_MAP", "0"),
            _env_value("MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_BAICHUAN_ALLOW_AUTO_DEVICE_MAP", "0"),
        )
        if _ACTIVE_MODEL_KEY == key and _ACTIVE_MODEL is not None and _ACTIVE_TOKENIZER is not None:
            self.model = _ACTIVE_MODEL
            self.tokenizer = _ACTIVE_TOKENIZER
            self.max_context_tokens = _ACTIVE_CONTEXT_LIMIT or 2048
            return

        if _ACTIVE_MODEL is not None and _ACTIVE_MODEL_KEY != key:
            _release_active_model()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        force_single_gpu_explicit_placement = (
            self.model_family in {"llama", "baichuan"}
            and torch.cuda.is_available()
            and _visible_cuda_device_count() == 1
            and (
                (
                    self.model_family == "llama"
                    and _env_value("MORACE_LLAMA_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_LLAMA_ALLOW_AUTO_DEVICE_MAP", "0") != "1"
                )
                or (
                    self.model_family == "baichuan"
                    and _env_value("MORACE_BAICHUAN_ALLOW_AUTO_DEVICE_MAP", "LLM4SD_BAICHUAN_ALLOW_AUTO_DEVICE_MAP", "0") != "1"
                )
            )
        )
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if force_single_gpu_explicit_placement:
            # For selected 7B/8B chat models on a single visible GPU, explicit
            # full-GPU placement is more stable than accelerate's auto
            # device_map, which may offload to CPU even when the card has enough
            # headroom.
            load_kwargs["device_map"] = {"": 0}
            load_kwargs["low_cpu_mem_usage"] = True
        else:
            load_kwargs["device_map"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                **load_kwargs,
            )
        self.model.eval()
        self.max_context_tokens = self._infer_context_limit()
        _ACTIVE_MODEL_KEY = key
        _ACTIVE_MODEL = self.model
        _ACTIVE_TOKENIZER = self.tokenizer
        _ACTIVE_CONTEXT_LIMIT = self.max_context_tokens

    def unload(self) -> None:
        # Keep the active model cached within the current process so Stage1/Stage2
        # can reuse the same weights without reloading checkpoint shards.
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        system_prompt = (system_prompt or "").strip()
        user_prompt = (user_prompt or "").strip()

        def tokenizer_has_chat_template() -> bool:
            return bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template")

        if self.model_family == "qwen" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.model_family in {"chemdfm", "glm", "gemma", "llama", "deepseek", "baichuan", "generic"} and tokenizer_has_chat_template():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                pass
        if self.model_family == "glm":
            if system_prompt:
                return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            return f"<|user|>\n{user_prompt}\n<|assistant|>\n"
        if self.model_family == "gemma":
            instruction = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
            return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        if self.model_family == "llama":
            if system_prompt:
                return (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            return (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        if self.model_family == "deepseek":
            instruction = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
            return f"User: {instruction}\n\nAssistant:"
        if self.model_family == "baichuan":
            instruction = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
            return f"User: {instruction}\n\nAssistant:"
        if self.model_family == "chemdfm":
            if system_prompt:
                return f"Human: {system_prompt}\n\n{user_prompt}\nAssistant:"
            return f"Human: {user_prompt}\nAssistant:"
        if self.model_family == "galactica":
            instruction = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
            if _env_value("MORACE_GALACTICA_PLAIN_PROMPT", "LLM4SD_GALACTICA_PLAIN_PROMPT", "1") == "1":
                return f"{instruction}\n"
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"

    def _truncate_prompt_text(
        self,
        prompt_text: str,
        max_input_tokens: int,
        preserve_head_tokens: int,
        preserve_tail_tokens: int,
    ) -> str:
        if max_input_tokens <= 0:
            return prompt_text
        token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(token_ids) <= max_input_tokens:
            return prompt_text
        tail_tokens = min(max(0, preserve_tail_tokens), max_input_tokens // 2)
        head_tokens = min(max(0, preserve_head_tokens), max_input_tokens - tail_tokens)
        if head_tokens == 0:
            head_tokens = max_input_tokens - tail_tokens
        if tail_tokens > 0 and head_tokens + tail_tokens < max_input_tokens:
            head_tokens = max_input_tokens - tail_tokens
        clipped = token_ids[:head_tokens]
        if tail_tokens > 0:
            clipped = clipped + token_ids[-tail_tokens:]
        return self.tokenizer.decode(clipped, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _resolve_eos_token_id(self, stage2_protocol_profile: str = ""):
        eos_ids: List[int] = []
        base_eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(base_eos, int) and base_eos >= 0:
            eos_ids.append(base_eos)
        if str(stage2_protocol_profile or "").strip() == "llama_native":
            for token in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"):
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                except Exception:
                    token_id = None
                if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
                    eos_ids.append(token_id)
        if not eos_ids:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    def _resolve_stop_strings(
        self,
        stop_strings: Optional[Sequence[str]] = None,
        *,
        stage2_protocol_profile: str = "",
    ) -> List[str]:
        resolved: List[str] = []
        seen = set()
        for stop in stop_strings or ():
            normalized = str(stop or "")
            if normalized and normalized not in seen:
                seen.add(normalized)
                resolved.append(normalized)
        if str(stage2_protocol_profile or "").strip() == "llama_native":
            for stop in ("```", "\n```", "\r\n```"):
                if stop not in seen:
                    seen.add(stop)
                    resolved.append(stop)
        return resolved

    def generate_batch(
        self,
        prompts: Sequence[Tuple[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
        max_input_tokens: Optional[int] = None,
        preserve_head_tokens: int = 384,
        preserve_tail_tokens: int = 256,
        stop_strings: Optional[Sequence[str]] = None,
        stage2_protocol_profile: str = "",
    ) -> List[str]:
        self.load()
        torch = _load_torch()
        responses: List[str] = []
        do_sample = temperature > 0.0
        device = next(self.model.parameters()).device
        context_limit = getattr(self, "max_context_tokens", 2048)
        safety_margin = 16
        resolved_stop_strings = self._resolve_stop_strings(
            stop_strings,
            stage2_protocol_profile=stage2_protocol_profile,
        )
        eos_token_id = self._resolve_eos_token_id(stage2_protocol_profile=stage2_protocol_profile)

        for system_prompt, user_prompt in prompts:
            prompt_text = self._build_prompt(system_prompt, user_prompt)
            effective_input_cap = max_input_tokens
            if effective_input_cap is None:
                effective_input_cap = max(1, context_limit - max_new_tokens - safety_margin)
            else:
                effective_input_cap = min(effective_input_cap, max(1, context_limit - safety_margin))

            prompt_text = self._truncate_prompt_text(
                prompt_text,
                int(effective_input_cap),
                preserve_head_tokens=preserve_head_tokens,
                preserve_tail_tokens=preserve_tail_tokens,
            )

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
            input_len = int(inputs["input_ids"].shape[1])
            allowed_new_tokens = max(1, min(max_new_tokens, context_limit - input_len - safety_margin))
            self._next_generation_seed(torch)
            with torch.inference_mode():
                generate_kwargs = dict(
                    **inputs,
                    max_new_tokens=allowed_new_tokens,
                    do_sample=do_sample,
                    num_beams=max(1, num_beams),
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=eos_token_id,
                )
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p
                outputs = self.model.generate(**generate_kwargs)
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            if resolved_stop_strings:
                stop_positions = [
                    decoded.find(stop) for stop in resolved_stop_strings if stop and decoded.find(stop) != -1
                ]
                if stop_positions:
                    decoded = decoded[: min(stop_positions)]
            responses.append(decoded.strip())
        return responses


    def generate_structured(
        self,
        prompts: Sequence[Tuple[str, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
        max_input_tokens: Optional[int] = None,
        preserve_head_tokens: int = 384,
        preserve_tail_tokens: int = 256,
        stop_strings: Optional[Sequence[str]] = None,
        stage2_protocol_profile: str = "",
        return_metadata: bool = False,
    ) -> List[str] | List[Dict]:
        self.load()
        torch = _load_torch()
        results: List[Dict] = []
        do_sample = temperature > 0.0
        device = next(self.model.parameters()).device
        context_limit = getattr(self, "max_context_tokens", 2048)
        safety_margin = 16
        backend_name = science_env_get("STAGE2_BACKEND", "prompt_only")
        resolved_stop_strings = self._resolve_stop_strings(
            stop_strings,
            stage2_protocol_profile=stage2_protocol_profile,
        )
        eos_token_id = self._resolve_eos_token_id(stage2_protocol_profile=stage2_protocol_profile)

        for system_prompt, user_prompt in prompts:
            prompt_text = self._build_prompt(system_prompt, user_prompt)
            effective_input_cap = max_input_tokens
            if effective_input_cap is None:
                effective_input_cap = max(1, context_limit - max_new_tokens - safety_margin)
            else:
                effective_input_cap = min(effective_input_cap, max(1, context_limit - safety_margin))

            prompt_text = self._truncate_prompt_text(
                prompt_text,
                int(effective_input_cap),
                preserve_head_tokens=preserve_head_tokens,
                preserve_tail_tokens=preserve_tail_tokens,
            )

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
            input_len = int(inputs["input_ids"].shape[1])
            allowed_new_tokens = max(1, min(max_new_tokens, context_limit - input_len - safety_margin))
            generation_seed = self._next_generation_seed(torch)
            with torch.inference_mode():
                generate_kwargs = dict(
                    **inputs,
                    max_new_tokens=allowed_new_tokens,
                    do_sample=do_sample,
                    num_beams=max(1, num_beams),
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p
                outputs = self.model.generate(**generate_kwargs)

            sequence = outputs.sequences[0]
            generated = sequence[input_len:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            if resolved_stop_strings:
                stop_positions = [
                    decoded.find(stop) for stop in resolved_stop_strings if stop and decoded.find(stop) != -1
                ]
                if stop_positions:
                    decoded = decoded[: min(stop_positions)]

            generated_token_count = int(generated.shape[0])
            avg_logprob = None
            logprob_norm = 0.0
            scores = getattr(outputs, "scores", None) or []
            if scores and generated_token_count > 0:
                token_logprobs = []
                steps = min(len(scores), generated_token_count)
                for step_idx in range(steps):
                    step_scores = scores[step_idx]
                    log_probs = torch.log_softmax(step_scores, dim=-1)
                    token_id = int(generated[step_idx].item())
                    token_logprobs.append(float(log_probs[0, token_id].item()))
                if token_logprobs:
                    avg_logprob = float(sum(token_logprobs) / len(token_logprobs))
                    logprob_norm = float(1.0 / (1.0 + math.exp(-avg_logprob)))

            results.append(
                {
                    "text": decoded.strip(),
                    "backend": backend_name,
                    "avg_logprob": avg_logprob,
                    "logprob_norm": logprob_norm,
                    "generated_token_count": generated_token_count,
                    "generation_seed": generation_seed,
                }
            )

        if return_metadata:
            return results
        return [item["text"] for item in results]

