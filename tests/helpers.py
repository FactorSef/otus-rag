"""Test helpers: fake embeddings and a sample corpus covering every supported format."""

import hashlib
import math
import textwrap
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from rag_kb.retrieval.bm25 import tokenize

DIM = 256


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-stems embedding: texts sharing words are close."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        vec[0] += 1e-3  # never a zero vector
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


SAMPLE_FILES = {
    "guide.md": """\
        # Klipper guide

        Intro paragraph about the firmware.

        ## Pressure advance

        Pressure advance compensates for pressure in the nozzle.
        Typical value for direct drive is 0.04.

        ```bash
        # not a heading inside a code fence
        echo tune
        ```

        ## Input shaper

        Input shaper reduces ringing on prints.
        """,
    "notes.txt": """\
        Журнал мастерской.

        Принтер «Сойка» откалиброван 12 марта, e-steps равен 97.3.
        """,
    "tools/purge.py": """\
        PURGE_LINE_LENGTH_MM = 143


        def purge_line(length=PURGE_LINE_LENGTH_MM):
            \"\"\"Generate purge line G-code.\"\"\"
            return f"G1 X{length} E15"


        class Calibrator:
            def run(self):
                return purge_line()
        """,
    "tools/temp.js": """\
        const MAX_BED_TEMP = 110;

        function clampBedTemp(t) {
          return Math.min(t, MAX_BED_TEMP);
        }
        """,
    "tools/fan.ts": """\
        export const FAN_KICKSTART_MS: number = 250;

        export function fanSpeed(percent: number): number {
          return Math.round((percent / 100) * 255);
        }
        """,
    "data/filaments.json": """\
        {
          "petg": {"name": "Smoky Quartz PETG", "nozzle_temp": 236, "bed_temp": 80},
          "pla": {"name": "Basic PLA", "nozzle_temp": 205, "bed_temp": 60}
        }
        """,
    "data/fleet.yaml": """\
        # Printer fleet
        soyka:
          model: Ender 3 V2
          e_steps: 97.3
        kestrel:
          model: Prusa MK4
          nozzle: 0.6
        """,
}


def write_files(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
    return root


class ScriptedChatModel(BaseChatModel):
    """Mock LLM: ``handler(system_prompt, user_prompt) -> reply``; records every call.

    Unlike ``FakeListChatModel`` the reply depends on the prompt, not on call order, so
    it stays deterministic when chunks are graded concurrently.
    """

    handler: Callable[[str, str], str]
    calls: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = messages[0].content if len(messages) > 1 else ""
        user = messages[-1].content
        self.calls.append((system, user))
        reply = AIMessage(self.handler(system, user))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def calls_to(self, system_prefix: str) -> list[str]:
        return [user for system, user in self.calls if system.startswith(system_prefix)]
