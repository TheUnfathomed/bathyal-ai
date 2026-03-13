"""GPT-5.4 agentic species classification fallback using OpenAI Agents SDK."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Literal, Sequence

from PIL import Image, ImageDraw
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_AGENT_TURNS = 3

MARINE_ID_SYSTEM_PROMPT = """\
You are a deep-sea marine biologist specializing in taxonomic identification from ROV \
and submersible imagery. You are analyzing imagery from underwater camera systems that \
may have challenging conditions: low light, blue/green color casts, motion blur, \
particulate backscatter, and organisms at unusual angles.

Your task: Identify the marine organism shown in the primary crop image.

Identification protocol:
1. Examine the PRIMARY CROP IMAGE first for diagnostic morphological features.
2. Use the FULL FRAME IMAGE for environmental context: depth zone indicators, substrate \
type, associated fauna, water conditions.
3. Consider the BioCLIP classifier's candidate rankings as a statistical hint, but \
override them when visual evidence contradicts. The classifier is limited to a fixed \
species set and may be wrong.
4. If you are not confident in your identification, use web search to look up reference \
images and diagnostic features for your top candidate species. Search for terms like \
"<species name> morphology identification" or "<species name> ROV imagery deep sea".
5. Work through the taxonomic hierarchy: first establish phylum/class, then narrow to \
order/family, then genus, then species only if diagnostic features are sufficient.

Morphological checklist for marine taxa:
- Fish: fin ray counts and placement, body shape (fusiform/laterally compressed/elongate), \
mouth position, eye size relative to head, coloration pattern, scale type
- Cnidarians: polyp structure, tentacle arrangement, colony form, skeletal structure
- Crustaceans: body segmentation, appendage count and form, carapace shape, antennae
- Echinoderms: symmetry type, arm count and form, surface texture, tube feet visibility
- Cephalopods: mantle shape, arm/tentacle count, fin position, chromatophore patterns
- Porifera: body form (encrusting/massive/branching), osculum visibility, surface texture
- Annelids: segmentation, parapodia presence, tube structure if present

When species-level identification is not possible from the available imagery, provide \
genus or family level identification instead. Never guess a species when only genus-level \
features are visible.

If the organism matches one of the known deployment species listed below, use that exact \
name. If it does not match any known species, identify it as accurately as possible using \
standard taxonomic nomenclature.
"""


class TaxonomicIdentification(BaseModel):
    species: str | None
    genus: str
    family: str
    common_name: str | None
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    morphological_features: list[str]
    habitat_notes: str | None


def is_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import agents  # noqa: F401
        return True
    except ImportError:
        return False


def _image_to_base64(image: Image.Image, max_long_edge: int = 2048, quality: int = 85) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_long_edge:
        scale = max_long_edge / max(w, h)
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_detection_highlight(
    frame: Image.Image, box: tuple[int, int, int, int]
) -> Image.Image:
    highlighted = frame.copy()
    draw = ImageDraw.Draw(highlighted)
    draw.rectangle(box, outline=(0, 255, 255), width=4)
    return highlighted


def _format_bioclip_candidates(candidates: Sequence[dict]) -> str:
    if not candidates:
        return "No BioCLIP candidates available."
    lines = ["BioCLIP classifier candidate rankings:"]
    for i, c in enumerate(candidates, 1):
        label = c.get("label", "unknown")
        prob = c.get("probability", 0.0)
        sim = c.get("centroid_similarity", 0.0)
        lines.append(f"  {i}. {label} (probability: {prob:.3f}, centroid_similarity: {sim:.3f})")
    return "\n".join(lines)


def _build_agent(known_species: Sequence[str], model: str = "gpt-5.4"):
    from agents import Agent, WebSearchTool

    system_prompt = (
        MARINE_ID_SYSTEM_PROMPT
        + "\nKnown deployment species:\n"
        + "\n".join(f"- {s}" for s in known_species)
    )

    return Agent(
        name="MarineSpeciesIdentifier",
        model=model,
        instructions=system_prompt,
        tools=[WebSearchTool()],
        output_type=TaxonomicIdentification,
    )


class VlmClassificationClient:
    def __init__(self, known_species: Sequence[str], model: str = "gpt-5.4") -> None:
        self.model = model
        self.known_species = list(known_species)
        self.agent = _build_agent(known_species, model)

    def classify(
        self,
        crop: Image.Image,
        frame: Image.Image,
        box: tuple[int, int, int, int],
        bioclip_candidates: Sequence[dict],
    ) -> dict | None:
        from agents import Runner

        crop_b64 = _image_to_base64(crop)
        highlighted_frame = draw_detection_highlight(frame, box)
        frame_b64 = _image_to_base64(highlighted_frame)
        candidates_text = _format_bioclip_candidates(bioclip_candidates)

        user_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            candidates_text
                            + "\n\nIdentify this organism. Use web search if you need to verify diagnostic features."
                            "\n\nThe first image is the cropped detection. "
                            "The second image is the full frame with the detection highlighted in cyan."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{crop_b64}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                ],
            },
        ]

        try:
            result = Runner.run_sync(
                self.agent,
                user_input,
                max_turns=MAX_AGENT_TURNS,
            )
        except Exception:
            logger.warning("GPT-5.4 agent classification failed", exc_info=True)
            return None

        parsed: TaxonomicIdentification | None = result.final_output_as(
            TaxonomicIdentification, raise_if_incorrect_type=False
        )
        if parsed is None:
            logger.warning("GPT-5.4 agent returned no structured output")
            return None

        best_label = parsed.species or parsed.genus
        return {
            "vlm_label": best_label,
            "vlm_species": parsed.species,
            "vlm_genus": parsed.genus,
            "vlm_family": parsed.family,
            "vlm_common_name": parsed.common_name,
            "vlm_confidence": parsed.confidence,
            "vlm_reasoning": parsed.reasoning,
            "vlm_morphological_features": parsed.morphological_features,
            "vlm_habitat_notes": parsed.habitat_notes,
            "vlm_model": self.model,
        }
