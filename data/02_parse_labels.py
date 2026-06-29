#!/usr/bin/env python3
"""
Step 2: Parse growth record tables (2024/2025) to extract stress/disease labels.
Map treatment processing numbers to individual plant numbers.
CSV plant numbers have 'L' prefix (e.g., L111, L112), growth records use raw numbers.

Label rules (3-class):  Healthy / Stress (disease or insect) / Other (abnormal)
Also exports 5-class:   Healthy / Pest / Disease / PestDisease / Other
"""
import os, csv, re, openpyxl
from collections import defaultdict, Counter

BASE = r'/Users/rainxu/Downloads/2023-2025 Tomato dataset'

# Treatment → plants (raw numbers, without 'L' prefix)
TREATMENT_MAP = {
    '1':  ['111','112','113','121','122','123','211','212','213','221','222','223'],
    '2':  ['214','215','216','224','225','226','311','312','313','321','322','323'],
    '3':  ['314','315','316','324','325','326','711','712','713','721','722','723'],
    'CK': ['714','715','716','724','725','726'],
    '4':  ['411','412','413','421','422','423','511','512','513','521','522','523'],
    '5':  ['514','515','516','524','525','526','611','612','613','621','622','623'],
    '6':  ['517','518','519','527','528','529','614','615','616','624','625','626'],
}

# Build lookup: raw_number -> treatment
RAW_TO_TREATMENT = {}
for t, plants in TREATMENT_MAP.items():
    for p in plants:
        RAW_TO_TREATMENT[p] = t


TREATMENT_DIGIT = {'1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'3'}

def find_treatment(csv_number):
    """Map CSV plant number to treatment group.
    'L111' -> strip 'L' -> '111' -> treatment '1'
    'R114' -> strip 'R' -> '114' -> first digit '1' -> treatment '1'
    'CK211' -> starts with CK -> treatment 'CK'
    """
    n = csv_number.strip()
    if n.upper().startswith('CK'):
        return 'CK'
    if n.startswith('L'):
        raw = n[1:]
        if raw in RAW_TO_TREATMENT:
            return RAW_TO_TREATMENT[raw]
    if n.startswith('R'):
        raw = n[1:]
        if raw[:1] in TREATMENT_DIGIT:
            return TREATMENT_DIGIT[raw[:1]]
    if n in RAW_TO_TREATMENT:
        return RAW_TO_TREATMENT[n]
    return None


def parse_growth_records(filepath, sheet_name, year):
    """Parse a growth record xlsx, return list of record dicts."""
    wb = openpyxl.load_workbook(filepath)
    ws = wb[sheet_name]
    records, current_date = [], None

    for ri, row in enumerate(ws.iter_rows(values_only=True)):
        if ri < 7:
            continue
        vals = [str(c).strip() if c else '' for c in row]
        # Detect date header: e.g. "4.21\nThe first time"
        if vals[0] and re.search(r'\d+\.\d+', vals[0]):
            m = re.search(r'(\d+)\.(\d+)', vals[0])
            if m:
                current_date = f'{year}{int(m.group(1)):02d}{int(m.group(2)):02d}'
        if not current_date or not vals[1]:
            continue
        records.append({
            'date': current_date,
            'treatment': vals[1].strip(),
            'insect': vals[2] if len(vals) > 2 else '',
            'disease': vals[3] if len(vals) > 3 else '',
            'growth_state': vals[4] if len(vals) > 4 else '',
            'abnormal': vals[12] if len(vals) > 12 else '',
        })
    return records


def label_row(row, recs_by_treatment):
    """Assign 3-class and 5-class labels to an agronomic data row."""
    csv_num = row['Number']
    treatment = find_treatment(csv_num)
    if treatment is None or treatment not in recs_by_treatment:
        row['label_3class'] = 'Unknown'
        row['label_5class'] = 'Unknown'
        row['label_treatment'] = ''
        row['insect'] = ''
        row['disease'] = ''
        row['growth_state'] = ''
        return row

    recs = recs_by_treatment[treatment]
    closest = min(recs, key=lambda r: abs(int(row['Date']) - int(r['date'])))

    hi = closest['insect'] in ('A', 'Yes', 'yes', 'Y')
    hd = closest['disease'] in ('A', 'Yes', 'yes', 'Y')
    ha = bool(closest.get('abnormal', ''))

    row['label_3class'] = 'Stress' if (hi or hd) else ('Other' if ha else 'Healthy')
    row['label_5class'] = 'PestDisease' if (hd and hi) else ('Disease' if hd else ('Pest' if hi else ('Other' if ha else 'Healthy')))
    row['label_treatment'] = treatment
    row['insect'] = closest['insect']
    row['disease'] = closest['disease']
    row['growth_state'] = closest['growth_state']
    return row


def main():
    # Read unified CSVs
    in_path = os.path.join(os.path.dirname(__file__), 'unified_agronomic.csv')
    with open(in_path) as f:
        all_rows = list(csv.DictReader(f))

    rows_by_year = defaultdict(list)
    for r in all_rows:
        rows_by_year[r['Source']].append(r)

    # Parse growth records
    recs_2024 = parse_growth_records(os.path.join(BASE, '2024 Tomato Growth Record Table.xlsx'), 'Sheet2', '2024')
    recs_2025 = parse_growth_records(os.path.join(BASE, '2025 Tomato Growth Record Table.xlsx'), 'Sheet2', '2025')
    print(f'Growth records: 2024={len(recs_2024)}, 2025={len(recs_2025)}')

    # Build treatment→records lookup
    def build_lookup(recs):
        d = defaultdict(list)
        for r in recs:
            d[r['treatment']].append(r)
        for t in d:
            d[t].sort(key=lambda x: x['date'])
        return d

    lookup_2024 = build_lookup(recs_2024)
    lookup_2025 = build_lookup(recs_2025)

    # Label each year
    for row in rows_by_year['2024']:
        label_row(row, lookup_2024)
    for row in rows_by_year['2025']:
        label_row(row, lookup_2025)
    for row in rows_by_year['2023']:
        row['label_3class'] = 'Unknown'
        row['label_5class'] = 'Unknown'
        row['label_treatment'] = ''
        row['insect'] = ''
        row['disease'] = ''
        row['growth_state'] = ''

    # Write output
    out_path = os.path.join(os.path.dirname(__file__), 'labeled_agronomic.csv')
    fieldnames = list(all_rows[0].keys()) + ['label_3class','label_5class','label_treatment','insect','disease','growth_state']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rows in rows_by_year.values():
            w.writerows(rows)

    # Stats
    print(f'\nLabel distribution (3-class, 2024+2025):')
    for year in ['2024','2025']:
        c = Counter(r['label_3class'] for r in rows_by_year[year])
        print(f'  {year}: {dict(c)} (n={sum(c.values())})')
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
