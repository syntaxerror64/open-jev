# Random vs trained checkpoint (holdout)

## Reproduce

```bash
.venv/bin/python -m eval.cli --dataset data/real/holdout.targets.jsonl --out eval/out/random --seed 0
.venv/bin/python -m eval.cli --dataset data/real/holdout.targets.jsonl --checkpoint runs/real/checkpoint.pt --out eval/out/trained --seed 0
.venv/bin/python -m eval.compare --dataset data/real/holdout.targets.jsonl --checkpoint runs/real/checkpoint.pt --out eval/COMPARISON.md --reports-dir eval/out
```

model: `random` vs `checkpoint:runs/real/checkpoint.pt`

## Что измеряют числа (рамка, R3)

ECE / Brier / NLL считаются против дистиллированных таргетов учителя, а не
против ground truth: это ***teacher-consistency*** (согласованность студента
с HFLocalTeacher), **не абсолютная правда** -- честнее uniform-заглушек, но и
это не «качество модели». Consistency -- симметричный KL между предсказаниями
на исходных и `shuffle_keys`-переформулировках одних и тех же состояний.

Число строк и вопросов-групп зафиксировано в таблице (`rows` -- строк
датасета, `questions` -- вопрос-групп; оба прогона считали один и тот же
датасет при одном seed, поэтому колонки совпадают).

Сырые отчёты лежат в `eval/out/random/report.json` + `report.md` и
`eval/out/trained/report.json` + `report.md` -- оба каталога gitignored
(при реальном запуске это `eval/out/{random,trained}`), этот файл --
единственный трекаемый след. Warning «Model weights are random» из
`eval/report.py` остаётся в ОБОИХ отчётах (report.py на этапе не правился):
для checkpoint-прогона случайных весов уже нет, но и сама метрика остаётся
teacher-consistency -- рамка задана здесь, в COMPARISON.md.

## Сравнение

| metric | random | checkpoint |
| --- | --- | --- |
| ece | 0.3072955918808778 | 0.15316430727640787 |
| brier | 0.31334368387858075 | 0.15089838951826096 |
| nll | 0.8849852482477824 | 0.6694825490315756 |
| consistency | 0.006497241566345717 | 0.00010305331185615312 |
| rows | 100 | 100 |
| questions | 3 | 3 |
| seed | 0 | 0 |
| n_bins | 10 | 10 |
| step | - | 200 |
| sha256 | - | 7be05cc3d23be4598e7d41773229d8c8ad32767a504b83ddfff902e908c720aa |
