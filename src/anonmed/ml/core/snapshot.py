import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from anonmed.ml.data.base import Dataset
from anonmed.ml.core.types import AnnotationSetLine, Case, Span, TextLine


class DatasetSnapshotWriter:
    def write_json(self, dataset: Dataset, path_to_file: str | Path) -> Path:
        self._validate_dataset(dataset)
        path = Path(path_to_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "samples_count": len(dataset.cases),
            "cases": [self._case_to_dict(case) for case in dataset.cases],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_parquet(self, dataset: Dataset, path_to_file: str | Path) -> Path:
        self._validate_dataset(dataset)
        path = Path(path_to_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = list(self._iter_flat_rows(dataset))
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return path

    def _validate_dataset(self, dataset: Dataset) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be Dataset, got {type(dataset).__name__}")

    def _case_to_dict(self, case: Case) -> dict[str, Any]:
        return {
            "document": {
                "sample_id": case.document.sample_id,
                "lines": [self._text_line_to_dict(line) for line in case.document.lines],
            },
            "target": {
                "idx": case.target.idx,
                "lines": [self._annotation_line_to_dict(line) for line in case.target.lines],
            },
        }

    def _text_line_to_dict(self, line: TextLine) -> dict[str, Any]:
        return {
            "idx": line.idx,
            "role": {"name": line.role.name, "kind": line.role.kind.value},
            "text": line.text,
        }

    def _annotation_line_to_dict(self, line: AnnotationSetLine) -> dict[str, Any]:
        return {
            "idx": line.idx,
            "role": {"name": line.role.name, "kind": line.role.kind.value},
            "spans": [self._span_to_dict(span) for span in line.spans],
        }

    def _span_to_dict(self, span: Span) -> dict[str, Any]:
        return {
            "line_idx": span.line_idx,
            "begin": span.begin,
            "end": span.end,
            "label": span.label,
            "data": span.data,
        }

    def _iter_flat_rows(self, dataset: Dataset):
        for case in dataset.cases:
            sample_id = case.document.sample_id
            target_idx = case.target.idx
            line_text_by_idx = {line.idx: line.text for line in case.document.lines}
            line_role_by_idx = {line.idx: line.role for line in case.document.lines}

            has_any_span = False
            for annotation_line in case.target.lines:
                for span in annotation_line.spans:
                    has_any_span = True
                    role = line_role_by_idx.get(annotation_line.idx, annotation_line.role)
                    yield {
                        "sample_id": sample_id,
                        "target_idx": target_idx,
                        "line_idx": annotation_line.idx,
                        "line_text": line_text_by_idx.get(annotation_line.idx),
                        "line_role_name": role.name,
                        "line_role_kind": role.kind.value,
                        "span_begin": span.begin,
                        "span_end": span.end,
                        "span_label": span.label,
                        "span_data": span.data,
                    }

            if not has_any_span:
                for document_line in case.document.lines:
                    yield {
                        "sample_id": sample_id,
                        "target_idx": target_idx,
                        "line_idx": document_line.idx,
                        "line_text": document_line.text,
                        "line_role_name": document_line.role.name,
                        "line_role_kind": document_line.role.kind.value,
                        "span_begin": None,
                        "span_end": None,
                        "span_label": None,
                        "span_data": None,
                    }
