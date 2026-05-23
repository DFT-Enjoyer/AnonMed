from datasets import load_dataset
from tqdm.auto import tqdm
from anonmed.ml.core.types import (
    Role, ParticipantKind, Span, TextLine, TextDocument,
    AnnotationSetLine, AnnotationSet, Case
)
from anonmed.ml.data.base import Dataset

class RussianPIIDataset(Dataset):
    def __init__(self, sample_size: int = 2000, random_seed: int = 42):
        self.sample_size = sample_size
        self.random_seed = random_seed
        super().__init__()   # cases = field(init=False)

    def _load(self):
        ds = load_dataset("wolframko/russian-pii-66k", split="train")
        ds = ds.shuffle(seed=self.random_seed).select(range(min(self.sample_size, len(ds))))
        object.__setattr__(self, '_row_data', ds)

    def _convert(self):
        role = Role(name="text", kind=ParticipantKind.UNKNOWN)
        case_list = []
        for idx, record in enumerate(tqdm(self._row_data, desc="Converting")):
            text = record.get("source_text") or ""
            if not text.strip():
                continue
            anns = record.get("privacy_mask") or []
            # Собираем GIVENNAME и SURNAME
            name_spans = []
            for ann in anns:
                label = ann.get("label")
                if label not in ("GIVENNAME", "SURNAME"):
                    continue
                start = ann.get("start")
                end = ann.get("end")
                if start is None or end is None:
                    continue
                if start < 0 or end > len(text):
                    raise ValueError(f"Invalid span [{start},{end}] in text of length {len(text)}")
                if start >= end:
                    raise ValueError(f"Empty span [{start},{end}]")
                name_spans.append((int(start), int(end)))
            # Объединение соседних
            name_spans.sort(key=lambda x: x[0])
            merged = []
            i = 0
            while i < len(name_spans):
                s, e = name_spans[i]
                j = i + 1
                while j < len(name_spans) and name_spans[j][0] <= e + 2:
                    e = max(e, name_spans[j][1])
                    j += 1
                merged.append(Span(line_idx=0, begin=s, end=e, label="PER", data=text[s:e]))
                i = j
            line = TextLine(idx=0, role=role, text=text)
            doc = TextDocument(lines=(line,), sample_id=str(idx))
            target_line = AnnotationSetLine(idx=0, role=role, spans=merged)
            target = AnnotationSet(lines=(target_line,), idx=None)
            case_list.append(Case(document=doc, target=target))
        object.__setattr__(self, 'cases', tuple(case_list))
