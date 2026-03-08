#!/usr/bin/env python3
"""
Демонстрация и тестирование системы склонения v2.

Покрывает все три фазы + GPU-компоненты.
Включает ранее падавшие тесты (фамилии, предложные группы).

Запуск: python -m russian_declension.tests.demo
"""
from __future__ import annotations
import sys, json, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

from russian_declension.core.enums import Case, Number
from russian_declension.service import DeclensionService, EntityType

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"

def tc(svc, text, case, expected, etype=EntityType.AUTO, gender=None, number=None):
    r = svc.inflect(text=text, target_case=case, entity_type=etype,
                    gender=gender, target_number=number)
    ok = r["result"].lower() == expected.lower()
    s = f"{G}✓{X}" if ok else f"{R}✗{X}"
    info = f'[{r["engine"]}, {r["confidence"]:.2f}]'
    if ok: print(f"  {s} «{text}» → {case.label} → «{r['result']}» {info}")
    else:  print(f"  {s} «{text}» → {case.label} → «{r['result']}» (ожидалось: «{expected}») {info}")
    return ok

def main():
    print(f"\n{B}{'='*70}\n  СИСТЕМА СКЛОНЕНИЯ v2 — CPU + GPU\n{'='*70}{X}\n")
    svc = DeclensionService()
    p = f = 0

    # ══ ФАЗА 1 ═══════════════════════════════════════════════════
    print(f"{C}{B}ФАЗА 1: Одиночные слова{X}\n")
    print(f"  {B}Обычные существительные:{X}")
    for t,c,e in [("кошка",Case.GENITIVE,"кошки"),("кошка",Case.DATIVE,"кошке"),
                   ("кошка",Case.ACCUSATIVE,"кошку"),("кошка",Case.INSTRUMENTAL,"кошкой"),
                   ("кошка",Case.PREPOSITIONAL,"кошке"),("дом",Case.GENITIVE,"дома"),
                   ("дом",Case.DATIVE,"дому"),("дом",Case.PREPOSITIONAL,"доме"),
                   ("время",Case.GENITIVE,"времени"),("время",Case.INSTRUMENTAL,"временем"),
                   ("путь",Case.GENITIVE,"пути"),("путь",Case.INSTRUMENTAL,"путём")]:
        ok = tc(svc, t, c, e, EntityType.WORD); p += ok; f += not ok

    print(f"\n  {B}Несклоняемые:{X}")
    for t in ("метро","кафе","такси","пальто"):
        ok = tc(svc, t, Case.GENITIVE, t, EntityType.WORD); p += ok; f += not ok

    print(f"\n  {B}Pluralia tantum:{X}")
    for t,c,e in [("ножницы",Case.GENITIVE,"ножниц"),("ножницы",Case.INSTRUMENTAL,"ножницами")]:
        ok = tc(svc, t, c, e, EntityType.WORD); p += ok; f += not ok

    print(f"\n  {B}Множественное число:{X}")
    for t,c,e in [("рубль",Case.GENITIVE,"рублей"),("документ",Case.DATIVE,"документам")]:
        ok = tc(svc, t, c, e, EntityType.WORD, number=Number.PLURAL); p += ok; f += not ok

    # ══ ФАЗА 2 ═══════════════════════════════════════════════════
    print(f"\n{C}{B}ФАЗА 2: Именованные сущности{X}\n")

    print(f"  {B}ФИО мужские (ИСПРАВЛЕНО в v2):{X}")
    for t,c,e in [("Иванов Иван Иванович",Case.GENITIVE,"Иванова Ивана Ивановича"),
                   ("Иванов Иван Иванович",Case.DATIVE,"Иванову Ивану Ивановичу"),
                   ("Петров Пётр Петрович",Case.INSTRUMENTAL,"Петровым Петром Петровичем")]:
        ok = tc(svc, t, c, e, EntityType.NAME, gender="male"); p += ok; f += not ok

    print(f"\n  {B}ФИО женские (ИСПРАВЛЕНО в v2):{X}")
    for t,c,e in [("Иванова Анна Ивановна",Case.GENITIVE,"Ивановой Анны Ивановны"),
                   ("Иванова Анна Ивановна",Case.DATIVE,"Ивановой Анне Ивановне")]:
        ok = tc(svc, t, c, e, EntityType.NAME, gender="female"); p += ok; f += not ok

    print(f"\n  {B}Несклоняемые фамилии:{X}")
    for t,c,e in [("Черных",Case.GENITIVE,"Черных"),("Шевченко",Case.DATIVE,"Шевченко")]:
        ok = tc(svc, t, c, e, EntityType.NAME); p += ok; f += not ok

    print(f"\n  {B}Организации:{X}")
    for t,c,e in [('ООО «Ромашка»',Case.GENITIVE,'ООО «Ромашка»'),
                   ('ООО «Ромашка»',Case.DATIVE,'ООО «Ромашка»'),
                   ('«Газпром»',Case.PREPOSITIONAL,'«Газпроме»')]:
        ok = tc(svc, t, c, e, EntityType.ORGANIZATION); p += ok; f += not ok

    print(f"\n  {B}Числительные:{X}")
    for t,c,e in [("5 рублей",Case.DATIVE,"пяти рублям"),
                   ("21 рубль",Case.DATIVE,"двадцати одному рублю"),
                   ("1000 рублей",Case.GENITIVE,"одной тысячи рублей")]:
        ok = tc(svc, t, c, e, EntityType.NUMERAL); p += ok; f += not ok

    # ══ ФАЗА 3 ═══════════════════════════════════════════════════
    print(f"\n{C}{B}ФАЗА 3: Фразовый движок{X}\n")

    print(f"  {B}Именные группы:{X}")
    for t,c,e in [("большой дом",Case.GENITIVE,"большого дома"),
                   ("красивая девушка",Case.DATIVE,"красивой девушке"),
                   ("новое здание",Case.INSTRUMENTAL,"новым зданием"),
                   ("старый добрый друг",Case.GENITIVE,"старого доброго друга")]:
        ok = tc(svc, t, c, e, EntityType.PHRASE); p += ok; f += not ok

    print(f"\n  {B}Генитивные цепочки:{X}")
    ok = tc(svc, "генеральный директор", Case.DATIVE,
            "генеральному директору", EntityType.PHRASE); p += ok; f += not ok

    print(f"\n  {B}Предложные группы (ИСПРАВЛЕНО в v2):{X}")
    for t,c,e in [
        ("общество с ограниченной ответственностью", Case.DATIVE,
         "обществу с ограниченной ответственностью"),
        ("общество с ограниченной ответственностью", Case.PREPOSITIONAL,
         "обществе с ограниченной ответственностью"),
        ("общество с ограниченной ответственностью", Case.INSTRUMENTAL,
         "обществом с ограниченной ответственностью"),
    ]:
        ok = tc(svc, t, c, e, EntityType.PHRASE); p += ok; f += not ok

    print(f"\n  {B}Авто-определение:{X}")
    for t,c,e in [("документ",Case.GENITIVE,"документа"),
                   ("21 рубль",Case.DATIVE,"двадцати одному рублю"),
                   ('ООО «Ромашка»',Case.GENITIVE,'ООО «Ромашка»')]:
        ok = tc(svc, t, c, e, EntityType.AUTO); p += ok; f += not ok

    # ══ Итоги ════════════════════════════════════════════════════
    total = p + f
    print(f"\n{B}{'='*70}")
    print(f"  ИТОГО: {p}/{total} тестов", end="")
    if f: print(f"  ({R}{f} не пройдено{X}{B})")
    else: print(f"  ({G}все пройдены!{X}{B})")
    print(f"{'='*70}{X}")

    # ── Демо API ─────────────────────────────────────────────────
    print(f"\n{Y}{B}Демо: ответ API{X}\n")
    print(json.dumps(svc.inflect("генеральный директор", Case.DATIVE,
                                  EntityType.PHRASE), ensure_ascii=False, indent=2))

    print(f"\n{Y}{B}Демо: парадигма «документ»{X}\n")
    par = svc.paradigm("документ")
    if par:
        forms = par.get("forms", {})
        names = {"nomn":"Именительный","gent":"Родительный","datv":"Дательный",
                 "accs":"Винительный","ablt":"Творительный","loct":"Предложный"}
        print(f"  {'Падеж':<16} {'Ед.ч.':<16} {'Мн.ч.':<16}")
        print(f"  {'─'*16} {'─'*16} {'─'*16}")
        for cc,cn in names.items():
            sg = forms.get(f"{cc}_sing","—"); pl = forms.get(f"{cc}_plur","—")
            print(f"  {cn:<16} {sg or '—':<16} {pl or '—':<16}")

    print(f"\n{Y}{B}Демо: healthcheck{X}\n")
    print(json.dumps(svc.health(), ensure_ascii=False, indent=2))

    return 0 if f == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
