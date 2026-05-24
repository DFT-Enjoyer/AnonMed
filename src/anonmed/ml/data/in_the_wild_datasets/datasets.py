from __future__ import annotations

from anonmed.ml.data.in_the_wild_datasets.base import InTheWildDataset


class InTheWildComprehensivePIIDataset(InTheWildDataset):
    dataset_directory = "1"
    name = "in_the_wild_russian_pii_speech"


class InTheWildNewsEntityDataset(InTheWildDataset):
    dataset_directory = "2"
    name = "in_the_wild_russian_news_ner"


class InTheWildNamesAddressesDataset(InTheWildDataset):
    dataset_directory = "3"
    name = "in_the_wild_russian_names_addresses"


class InTheWildDialogPIIDataset(InTheWildDataset):
    dataset_directory = "4"
    name = "in_the_wild_dialog_pii"


class InTheWildControlledPIIDataset(InTheWildDataset):
    dataset_directory = "5"
    name = "in_the_wild_controlled_synthetic_pii"


class InTheWildMedicalNotesPIIDataset(InTheWildDataset):
    dataset_directory = "6"
    name = "in_the_wild_medical_notes_pii"


__all__: list[str] = [
    "InTheWildComprehensivePIIDataset",
    "InTheWildControlledPIIDataset",
    "InTheWildDialogPIIDataset",
    "InTheWildMedicalNotesPIIDataset",
    "InTheWildNamesAddressesDataset",
    "InTheWildNewsEntityDataset",
]
