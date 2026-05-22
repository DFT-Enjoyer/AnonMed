from __future__ import annotations

from dataclasses import dataclass
import unittest

from anonmed.preprocessing import IntegerExtractor, IntegerSpan, replace_integer_spans


@dataclass(frozen=True, slots=True)
class ExtractionCase:
    text: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplacementCase:
    text: str
    replaced: str


_SIMPLE_EXTRACTION_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("стоп здесь", ()),
    ExtractionCase("никаких чисел нет", ()),
    ExtractionCase("обычный текст без цифровых сущностей", ()),
    ExtractionCase("пациент 0", ("0",)),
    ExtractionCase("код 7", ("7",)),
    ExtractionCase("значение 0007", ("0007",)),
    ExtractionCase("abc5435453def", ("5435453",)),
    ExtractionCase("до 93243243243 потом 43249329 и текст5435453", ("93243243243", "43249329", "5435453")),
    ExtractionCase("ноль", ("0",)),
    ExtractionCase("нуль", ("0",)),
    ExtractionCase("один", ("1",)),
    ExtractionCase("одна", ("1",)),
    ExtractionCase("одно", ("1",)),
    ExtractionCase("четвертый", ("4",)),
    ExtractionCase("пятый", ("5",)),
    ExtractionCase("шестой", ("6",)),
    ExtractionCase("седьмой", ("7",)),
    ExtractionCase("раз", ("1",)),
    ExtractionCase("два", ("2",)),
    ExtractionCase("две", ("2",)),
    ExtractionCase("три", ("3",)),
    ExtractionCase("четыре", ("4",)),
    ExtractionCase("пять", ("5",)),
    ExtractionCase("шесть", ("6",)),
    ExtractionCase("семь", ("7",)),
    ExtractionCase("восемь", ("8",)),
    ExtractionCase("девять", ("9",)),
    ExtractionCase("десять", ("10",)),
    ExtractionCase("одиннадцать", ("11",)),
    ExtractionCase("двенадцать", ("12",)),
    ExtractionCase("тринадцать", ("13",)),
    ExtractionCase("четырнадцать", ("14",)),
    ExtractionCase("пятнадцать", ("15",)),
    ExtractionCase("шестнадцать", ("16",)),
    ExtractionCase("семнадцать", ("17",)),
    ExtractionCase("восемнадцать", ("18",)),
    ExtractionCase("девятнадцать", ("19",)),
    ExtractionCase("двадцать", ("20",)),
    ExtractionCase("тридцать", ("30",)),
    ExtractionCase("сорок", ("40",)),
    ExtractionCase("пятьдесят", ("50",)),
    ExtractionCase("шестьдесят", ("60",)),
    ExtractionCase("семьдесят", ("70",)),
    ExtractionCase("восемьдесят", ("80",)),
    ExtractionCase("девяносто", ("90",)),
    ExtractionCase("сто", ("100",)),
    ExtractionCase("двести", ("200",)),
    ExtractionCase("триста", ("300",)),
    ExtractionCase("четыреста", ("400",)),
    ExtractionCase("пятьсот", ("500",)),
    ExtractionCase("шестьсот", ("600",)),
    ExtractionCase("семьсот", ("700",)),
    ExtractionCase("восемьсот", ("800",)),
    ExtractionCase("девятьсот", ("900",)),
    ExtractionCase("сорок два", ("42",)),
    ExtractionCase("девяносто девять", ("99",)),
    ExtractionCase("сто пять", ("105",)),
    ExtractionCase("двести тринадцать", ("213",)),
    ExtractionCase("триста сорок семь", ("347",)),
    ExtractionCase("девятьсот девяносто девять", ("999",)),
    ExtractionCase("одна тысяча пять", ("1005",)),
    ExtractionCase("тысяча двадцать три", ("1023",)),
    ExtractionCase("две тысячи десять", ("2010",)),
    ExtractionCase("пять тысяч шестьсот семь", ("5607",)),
    ExtractionCase("девять тысяч девятьсот девяносто девять", ("9999",)),
    ExtractionCase("один миллион", ("1000000",)),
    ExtractionCase("два миллиона три", ("2000003",)),
    ExtractionCase("миллион три тысячи пять", ("1003005",)),
    ExtractionCase("минус один", ("-1",)),
    ExtractionCase("минус семь", ("-7",)),
    ExtractionCase("минус двадцать пять", ("-25",)),
    ExtractionCase("минус сто три", ("-103",)),
    ExtractionCase("минус две тысячи пять", ("-2005",)),
)

_DIGIT_SEQUENCE_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("один два три", ("123",)),
    ExtractionCase("ноль пять шесть", ("056",)),
    ExtractionCase("девять три два четыре", ("9324",)),
    ExtractionCase("девять ноль ноль один", ("9001",)),
    ExtractionCase("одна две три четыре пять", ("12345",)),
    ExtractionCase("раз два три четыре", ("1234",)),
    ExtractionCase("код девять три два четыре", ("9324",)),
    ExtractionCase("номер ноль ноль один", ("001",)),
    ExtractionCase("значение семь восемь девять", ("789",)),
    ExtractionCase("телефон девять один два три четыре пять шесть семь восемь девять", ("9123456789",)),
    ExtractionCase("номер 7 восемь девять", ("789",)),
    ExtractionCase("код девять 8 семь", ("987",)),
    ExtractionCase("один 2 три 4", ("1234",)),
    ExtractionCase("ноль 0 ноль 1", ("0001",)),
    ExtractionCase("минус девять восемь семь", ("-987",)),
)

_MIXED_CARDINAL_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("двадцать 5", ("25",)),
    ExtractionCase("сорок 2", ("42",)),
    ExtractionCase("сто 5", ("105",)),
    ExtractionCase("двести 30", ("230",)),
    ExtractionCase("триста 40 7", ("347",)),
    ExtractionCase("тысяча 5", ("1005",)),
    ExtractionCase("две тысячи 15", ("2015",)),
    ExtractionCase("3 тысячи пять", ("3005",)),
    ExtractionCase("5 миллионов 12", ("5000012",)),
    ExtractionCase("минус 12", ("-12",)),
)

_INFLECTED_UNIT_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до одного", ("1",)),
    ExtractionCase("к одному", ("1",)),
    ExtractionCase("с одним", ("1",)),
    ExtractionCase("для одной", ("1",)),
    ExtractionCase("в одном", ("1",)),
    ExtractionCase("вижу одну", ("1",)),
    ExtractionCase("до двух", ("2",)),
    ExtractionCase("к двум", ("2",)),
    ExtractionCase("с двумя", ("2",)),
    ExtractionCase("до трех", ("3",)),
    ExtractionCase("к трем", ("3",)),
    ExtractionCase("с тремя", ("3",)),
    ExtractionCase("до четырех", ("4",)),
    ExtractionCase("к четырем", ("4",)),
    ExtractionCase("с четырьмя", ("4",)),
    ExtractionCase("до пяти", ("5",)),
    ExtractionCase("с пятью", ("5",)),
    ExtractionCase("до шести", ("6",)),
    ExtractionCase("с шестью", ("6",)),
    ExtractionCase("до семи", ("7",)),
    ExtractionCase("с семью", ("7",)),
    ExtractionCase("до восьми", ("8",)),
    ExtractionCase("с восемью", ("8",)),
    ExtractionCase("до девяти", ("9",)),
    ExtractionCase("с девятью", ("9",)),
)

_ORDINAL_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("первый", ("1",)),
    ExtractionCase("вторую", ("2",)),
    ExtractionCase("третьем", ("3",)),
    ExtractionCase("четвертый", ("4",)),
    ExtractionCase("четвертую", ("4",)),
    ExtractionCase("шестом", ("6",)),
    ExtractionCase("восьмое", ("8",)),
    ExtractionCase("девятых", ("9",)),
    ExtractionCase("десятому", ("10",)),
    ExtractionCase("одиннадцатая", ("11",)),
    ExtractionCase("двадцатый", ("20",)),
    ExtractionCase("сороковой", ("40",)),
    ExtractionCase("сотый", ("100",)),
    ExtractionCase("двухсотый", ("200",)),
    ExtractionCase("тысячном", ("1000",)),
    ExtractionCase("двадцать первый", ("21",)),
    ExtractionCase("к двадцать первому", ("21",)),
    ExtractionCase("до сорокового", ("40",)),
)

_INFLECTED_TENS_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до двадцати", ("20",)),
    ExtractionCase("с двадцатью", ("20",)),
    ExtractionCase("до тридцати", ("30",)),
    ExtractionCase("с тридцатью", ("30",)),
    ExtractionCase("до сорока", ("40",)),
    ExtractionCase("с сорока", ("40",)),
    ExtractionCase("до пятидесяти", ("50",)),
    ExtractionCase("с пятьюдесятью", ("50",)),
    ExtractionCase("до шестидесяти", ("60",)),
    ExtractionCase("с шестьюдесятью", ("60",)),
    ExtractionCase("до семидесяти", ("70",)),
    ExtractionCase("с семьюдесятью", ("70",)),
    ExtractionCase("до восьмидесяти", ("80",)),
    ExtractionCase("с восемьюдесятью", ("80",)),
    ExtractionCase("до девяноста", ("90",)),
    ExtractionCase("с девяноста", ("90",)),
)

_INFLECTED_TENS_WITH_UNITS_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до двадцати одного", ("21",)),
    ExtractionCase("к двадцати одному", ("21",)),
    ExtractionCase("с двадцатью одним", ("21",)),
    ExtractionCase("вижу двадцать одну", ("21",)),
    ExtractionCase("до двадцати двух", ("22",)),
    ExtractionCase("к двадцати двум", ("22",)),
    ExtractionCase("с двадцатью двумя", ("22",)),
    ExtractionCase("до двадцати пяти", ("25",)),
    ExtractionCase("с двадцатью пятью", ("25",)),
    ExtractionCase("до тридцати трех", ("33",)),
    ExtractionCase("к тридцати трем", ("33",)),
    ExtractionCase("с тридцатью тремя", ("33",)),
    ExtractionCase("до сорока двух", ("42",)),
    ExtractionCase("к сорока двум", ("42",)),
    ExtractionCase("с сорока двумя", ("42",)),
    ExtractionCase("до пятидесяти четырех", ("54",)),
    ExtractionCase("к пятидесяти четырем", ("54",)),
    ExtractionCase("с пятьюдесятью четырьмя", ("54",)),
    ExtractionCase("до шестидесяти шести", ("66",)),
    ExtractionCase("с шестьюдесятью шестью", ("66",)),
    ExtractionCase("до семидесяти семи", ("77",)),
    ExtractionCase("с семьюдесятью семью", ("77",)),
    ExtractionCase("до восьмидесяти восьми", ("88",)),
    ExtractionCase("с восемьюдесятью восемью", ("88",)),
    ExtractionCase("до девяноста девяти", ("99",)),
    ExtractionCase("с девяноста девятью", ("99",)),
)

_INFLECTED_HUNDREDS_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до ста", ("100",)),
    ExtractionCase("к ста", ("100",)),
    ExtractionCase("со ста", ("100",)),
    ExtractionCase("до двухсот", ("200",)),
    ExtractionCase("к двумстам", ("200",)),
    ExtractionCase("с двумястами", ("200",)),
    ExtractionCase("о двухстах", ("200",)),
    ExtractionCase("до трехсот", ("300",)),
    ExtractionCase("к тремстам", ("300",)),
    ExtractionCase("с тремястами", ("300",)),
    ExtractionCase("о трехстах", ("300",)),
    ExtractionCase("до четырехсот", ("400",)),
    ExtractionCase("к четыремстам", ("400",)),
    ExtractionCase("с четырьмястами", ("400",)),
    ExtractionCase("о четырехстах", ("400",)),
    ExtractionCase("до пятисот", ("500",)),
    ExtractionCase("к пятистам", ("500",)),
    ExtractionCase("с пятьюстами", ("500",)),
    ExtractionCase("о пятистах", ("500",)),
    ExtractionCase("до шестисот", ("600",)),
    ExtractionCase("к шестистам", ("600",)),
    ExtractionCase("с шестьюстами", ("600",)),
    ExtractionCase("о шестистах", ("600",)),
    ExtractionCase("до семисот", ("700",)),
    ExtractionCase("к семистам", ("700",)),
    ExtractionCase("с семьюстами", ("700",)),
    ExtractionCase("о семистах", ("700",)),
    ExtractionCase("до восьмисот", ("800",)),
    ExtractionCase("к восьмистам", ("800",)),
    ExtractionCase("с восемьюстами", ("800",)),
    ExtractionCase("о восьмистах", ("800",)),
    ExtractionCase("до девятисот", ("900",)),
    ExtractionCase("к девятистам", ("900",)),
    ExtractionCase("с девятьюстами", ("900",)),
    ExtractionCase("о девятистах", ("900",)),
)

_INFLECTED_HUNDREDS_COMPOUND_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до ста пяти", ("105",)),
    ExtractionCase("до двухсот десяти", ("210",)),
    ExtractionCase("до двухсот пятнадцати", ("215",)),
    ExtractionCase("до двухсот двадцати пяти", ("225",)),
    ExtractionCase("к двумстам двадцати пяти", ("225",)),
    ExtractionCase("с двумястами двадцатью пятью", ("225",)),
    ExtractionCase("до трехсот сорока семи", ("347",)),
    ExtractionCase("к тремстам сорока семи", ("347",)),
    ExtractionCase("с тремястами сорока семью", ("347",)),
    ExtractionCase("о трехстах сорока семи", ("347",)),
    ExtractionCase("до четырехсот пятидесяти четырех", ("454",)),
    ExtractionCase("с четырьмястами пятьюдесятью четырьмя", ("454",)),
    ExtractionCase("до пятисот шестидесяти шести", ("566",)),
    ExtractionCase("с пятьюстами шестьюдесятью шестью", ("566",)),
    ExtractionCase("до шестисот семидесяти семи", ("677",)),
    ExtractionCase("с шестьюстами семьюдесятью семью", ("677",)),
    ExtractionCase("до семисот восьмидесяти восьми", ("788",)),
    ExtractionCase("с семьюстами восемьюдесятью восемью", ("788",)),
    ExtractionCase("до восьмисот девяноста девяти", ("899",)),
    ExtractionCase("с восемьюстами девяноста девятью", ("899",)),
)

_INFLECTED_SCALE_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до тысячи", ("1000",)),
    ExtractionCase("к тысяче", ("1000",)),
    ExtractionCase("с тысячей", ("1000",)),
    ExtractionCase("до двух тысяч", ("2000",)),
    ExtractionCase("к двум тысячам", ("2000",)),
    ExtractionCase("с двумя тысячами", ("2000",)),
    ExtractionCase("о двух тысячах", ("2000",)),
    ExtractionCase("до пяти тысяч", ("5000",)),
    ExtractionCase("к пяти тысячам", ("5000",)),
    ExtractionCase("с пятью тысячами", ("5000",)),
    ExtractionCase("до миллиона", ("1000000",)),
    ExtractionCase("к миллиону", ("1000000",)),
    ExtractionCase("с миллионом", ("1000000",)),
    ExtractionCase("о миллионе", ("1000000",)),
    ExtractionCase("до двух миллионов", ("2000000",)),
    ExtractionCase("к двум миллионам", ("2000000",)),
    ExtractionCase("с двумя миллионами", ("2000000",)),
    ExtractionCase("о двух миллионах", ("2000000",)),
)

_INFLECTED_SCALE_COMPOUND_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("до двух тысяч пятисот шести", ("2506",)),
    ExtractionCase("к двум тысячам пятистам шести", ("2506",)),
    ExtractionCase("с двумя тысячами пятьюстами шестью", ("2506",)),
    ExtractionCase("о двух тысячах пятистах шести", ("2506",)),
    ExtractionCase("до трех тысяч четырехсот пятидесяти шести", ("3456",)),
    ExtractionCase("с тремя тысячами четырьмястами пятьюдесятью шестью", ("3456",)),
    ExtractionCase("до двадцати пяти тысяч", ("25000",)),
    ExtractionCase("к двадцати пяти тысячам", ("25000",)),
    ExtractionCase("с двадцатью пятью тысячами", ("25000",)),
    ExtractionCase("до ста двадцати трех тысяч четырехсот пятидесяти шести", ("123456",)),
    ExtractionCase("с двумя миллионами тремястами тысячами сорока пятью", ("2300045",)),
    ExtractionCase("до двух миллионов трехсот тысяч сорока пяти", ("2300045",)),
)

_ASR_AND_SPELLING_NOISE_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("адин", ("1",)),
    ExtractionCase("оден", ("1",)),
    ExtractionCase("трии", ("3",)),
    ExtractionCase("четыр", ("4",)),
    ExtractionCase("пят", ("5",)),
    ExtractionCase("шес", ("6",)),
    ExtractionCase("сем", ("7",)),
    ExtractionCase("восимь", ("8",)),
    ExtractionCase("восем", ("8",)),
    ExtractionCase("дивять", ("9",)),
    ExtractionCase("девят", ("9",)),
    ExtractionCase("одинадцать", ("11",)),
    ExtractionCase("двинадцать", ("12",)),
    ExtractionCase("двенадцат", ("12",)),
    ExtractionCase("тринадцат", ("13",)),
    ExtractionCase("пятнадцат", ("15",)),
    ExtractionCase("шеснадцать", ("16",)),
    ExtractionCase("шестнадцат", ("16",)),
    ExtractionCase("семнадцат", ("17",)),
    ExtractionCase("восемнадцат", ("18",)),
    ExtractionCase("девятнадцат", ("19",)),
    ExtractionCase("двадцат пять", ("25",)),
    ExtractionCase("двадцать пят", ("25",)),
    ExtractionCase("тридцат семь", ("37",)),
    ExtractionCase("сорокк два", ("42",)),
    ExtractionCase("педесят шесть", ("56",)),
    ExtractionCase("пятьдесет семь", ("57",)),
    ExtractionCase("шездесят восемь", ("68",)),
    ExtractionCase("шестьдесет девять", ("69",)),
    ExtractionCase("семдесят один", ("71",)),
    ExtractionCase("семисят два", ("72",)),
    ExtractionCase("восемдесят три", ("83",)),
    ExtractionCase("дивяносто четыре", ("94",)),
    ExtractionCase("двесте пять", ("205",)),
    ExtractionCase("тристаа шесть", ("306",)),
    ExtractionCase("пятсот семь", ("507",)),
    ExtractionCase("пицот восемь", ("508",)),
    ExtractionCase("шесот девять", ("609",)),
    ExtractionCase("шиссот десять", ("610",)),
    ExtractionCase("семсот одиннадцать", ("711",)),
    ExtractionCase("восемсот двенадцать", ("812",)),
    ExtractionCase("девятсот тринадцать", ("913",)),
    ExtractionCase("тысича пять", ("1005",)),
    ExtractionCase("тыща семь", ("1007",)),
    ExtractionCase("милион восемь", ("1000008",)),
    ExtractionCase("милен девять", ("1000009",)),
    ExtractionCase("девят нол нол адин", ("9001",)),
    ExtractionCase("номер нол нол семь", ("007",)),
)

_FALSE_POSITIVE_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("стол стоит", ()),
    ExtractionCase("сорока прилетела", ()),
    ExtractionCase("пятак лежит", ()),
    ExtractionCase("семья пришла", ()),
    ExtractionCase("одиночный случай", ()),
    ExtractionCase("тридцатка это разговорное слово", ()),
    ExtractionCase("пятиэтажный дом", ()),
    ExtractionCase("двухкомнатная квартира", ()),
    ExtractionCase("стоп сигнал", ()),
    ExtractionCase("номерной знак без числа", ()),
)

_REPLACEMENT_CASES: tuple[ReplacementCase, ...] = (
    ReplacementCase("до девять три два после 432 конец", "до 932 после 432 конец"),
    ReplacementCase("код сорока двум пациентам", "код 42 пациентам"),
    ReplacementCase("до двадцати пяти дней", "до 25 дней"),
    ReplacementCase("с двадцатью пятью пациентами", "с 25 пациентами"),
    ReplacementCase("к тремстам сорока семи образцам", "к 347 образцам"),
    ReplacementCase("с тремястами сорока семью образцами", "с 347 образцами"),
    ReplacementCase("до двух тысяч пятисот шести строк", "до 2506 строк"),
    ReplacementCase("с двумя тысячами пятьюстами шестью строками", "с 2506 строками"),
    ReplacementCase("abc5435453def", "abc5435453def"),
    ReplacementCase("ошибок нет", "ошибок нет"),
)

_BOUNDARY_CASES: tuple[tuple[str, str, str], ...] = (
    ("до сорока двум после", "42", "сорока двум"),
    ("к двадцати пяти образцам", "25", "двадцати пяти"),
    ("с двадцатью пятью образцами", "25", "двадцатью пятью"),
    ("у трехсот сорока семи пациентов", "347", "трехсот сорока семи"),
    ("с тремястами сорока семью пациентами", "347", "тремястами сорока семью"),
    ("до двух тысяч пятисот шести записей", "2506", "двух тысяч пятисот шести"),
)

_TAIL_REGRESSION_CASES: tuple[ExtractionCase, ...] = (
    ExtractionCase("сорока двум", ("42",)),
    ExtractionCase("двадцати пяти", ("25",)),
    ExtractionCase("тремстам сорока семи", ("347",)),
    ExtractionCase("пятистам шестидесяти шести", ("566",)),
    ExtractionCase("двух тысяч пятисот шести", ("2506",)),
)


class ExtendedIntegerExtractionSpecificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = IntegerExtractor()

    def values(self, text: str) -> tuple[str, ...]:
        spans: list[IntegerSpan] = self.extractor.extract(text)
        values: tuple[str, ...] = tuple(span.value for span in spans)
        return values

    def assert_cases(self, cases: tuple[ExtractionCase, ...]) -> None:
        for case in cases:
            with self.subTest(text=case.text):
                actual_values: tuple[str, ...] = self.values(case.text)
                self.assertEqual(actual_values, case.values)

    def test_simple_numbers(self) -> None:
        self.assert_cases(_SIMPLE_EXTRACTION_CASES)

    def test_digit_sequences(self) -> None:
        self.assert_cases(_DIGIT_SEQUENCE_CASES)

    def test_mixed_cardinals(self) -> None:
        self.assert_cases(_MIXED_CARDINAL_CASES)

    def test_inflected_units(self) -> None:
        self.assert_cases(_INFLECTED_UNIT_CASES)

    def test_ordinals(self) -> None:
        self.assert_cases(_ORDINAL_CASES)

    def test_inflected_tens(self) -> None:
        self.assert_cases(_INFLECTED_TENS_CASES)

    def test_inflected_tens_with_units(self) -> None:
        self.assert_cases(_INFLECTED_TENS_WITH_UNITS_CASES)

    def test_inflected_hundreds(self) -> None:
        self.assert_cases(_INFLECTED_HUNDREDS_CASES)

    def test_inflected_hundreds_compound(self) -> None:
        self.assert_cases(_INFLECTED_HUNDREDS_COMPOUND_CASES)

    def test_inflected_scales(self) -> None:
        self.assert_cases(_INFLECTED_SCALE_CASES)

    def test_inflected_scale_compounds(self) -> None:
        self.assert_cases(_INFLECTED_SCALE_COMPOUND_CASES)

    def test_asr_and_spelling_noise(self) -> None:
        self.assert_cases(_ASR_AND_SPELLING_NOISE_CASES)

    def test_false_positive_controls(self) -> None:
        self.assert_cases(_FALSE_POSITIVE_CASES)

    def test_inflected_phrases_are_not_reduced_to_tail_digit(self) -> None:
        self.assert_cases(_TAIL_REGRESSION_CASES)

    def test_replace_inflected_spans(self) -> None:
        for case in _REPLACEMENT_CASES:
            with self.subTest(text=case.text):
                replaced: str = replace_integer_spans(case.text)
                self.assertEqual(replaced, case.replaced)

    def test_raw_boundaries_for_inflected_spans(self) -> None:
        for text, expected_value, expected_raw in _BOUNDARY_CASES:
            with self.subTest(text=text):
                spans: list[IntegerSpan] = self.extractor.extract(text)
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].value, expected_value)
                self.assertEqual(spans[0].raw, expected_raw)
                self.assertEqual(text[spans[0].start : spans[0].end], expected_raw)

    def test_case_count_is_large_enough_for_regression_suite(self) -> None:
        all_extraction_cases: tuple[ExtractionCase, ...] = (
            _SIMPLE_EXTRACTION_CASES
            + _DIGIT_SEQUENCE_CASES
            + _MIXED_CARDINAL_CASES
            + _INFLECTED_UNIT_CASES
            + _ORDINAL_CASES
            + _INFLECTED_TENS_CASES
            + _INFLECTED_TENS_WITH_UNITS_CASES
            + _INFLECTED_HUNDREDS_CASES
            + _INFLECTED_HUNDREDS_COMPOUND_CASES
            + _INFLECTED_SCALE_CASES
            + _INFLECTED_SCALE_COMPOUND_CASES
            + _ASR_AND_SPELLING_NOISE_CASES
            + _FALSE_POSITIVE_CASES
            + _TAIL_REGRESSION_CASES
        )
        self.assertGreaterEqual(len(all_extraction_cases), 250)


__all__: list[str] = []
