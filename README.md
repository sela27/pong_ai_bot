# pong_ai_bot

משחק Pong שנבנה לוקלית, ו-Agent מסוג DQN שלומד לשחק בו רק מה-Observation ומה-Reward, ברוח המאמר של DeepMind. הכל רץ על המחשב של המפתח. המונחים מוגדרים ב-[GLOSSARY.md](GLOSSARY.md).

היעד: ‏Score margin ממוצע של 18 לפחות ב-Confirmation. ההחלטות, והקישורים לסיבות שלהן, ב-[spec](https://github.com/sela27/pong_ai_bot/issues/17).

## מצב השלבים

| שלב | מצב |
|---|---|
| 1. הסביבה | גמור ([#34](https://github.com/sela27/pong_ai_bot/issues/34)) |
| 2. הלומד ולולאת האימון | בעבודה |
| 3. ה-Agent מפיקסלים | לא התחיל |

בשלב 1, ‏`pong play` עבר את [`window.md`](docs/checklists/window.md) בבדיקה של המפתח: תשע שורות [בבדיקה הראשונה](https://github.com/sela27/pong_ai_bot/issues/33#issuecomment-7212234760), ושורה 9 [בבדיקה החוזרת](https://github.com/sela27/pong_ai_bot/issues/33#issuecomment-7217579392), ברוחב הנעול.

## מה צריך

- ‏Windows.
- כרטיס NVIDIA. ‏PyTorch נעול ל-build ‏`cu126`, האחרון שתומך ב-Quadro P2200, והדרייבר נשאר על branch ‏R580 ([#2](https://github.com/sela27/pong_ai_bot/issues/2)).
- ‏Python 3.14.
- ‏[uv](https://docs.astral.sh/uv/) 0.12.6.

## התקנה

1. התקנת התלויות, לפי `uv.lock`:

   ```
   uv sync
   ```

2. הפעלת ה-hook, פעם אחת לכל clone. מכאן כל commit עובר קודם formatting, ‏linting, בדיקת טיפוסים והבדיקות המהירות:

   ```
   git config core.hooksPath .githooks
   ```

3. בדיקה שכרטיס המסך נראה ל-PyTorch:

   ```
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
   ```

   צריך לראות `True`, את שם הכרטיס, ואת הארכיטקטורה שלו ברשימה (ב-P2200: ‏`sm_61`).

## פקודות

### לשחק ולצפות

```
uv run pong play
```

‏Human נגד ה-Opponent, בחלון: אתה מימין, במקום של ה-Agent, ורואה בדיוק את מה שה-Agent רואה. החצים למעלה ולמטה מזיזים את ה-Paddle, רווח עוצר וממשיך, ו-Esc יוצא.

### לאמן

```
uv run pong train --player test-env-1 --name tiny --seed 1 --device cpu --budget 1000 --fill-steps 100
```

‏Run חדש של שחקן לומד, מאפס, ולא מהמשקלים של Run אחר. הפקודה יוצרת תיקיית Run חדשה ב-`~/pong_ai_bot_runs/`, או בתיקייה שהמשתנה `PONG_AI_BOT_RUNS_DIR` מציין, וקובץ ההגדרות שבה רושם את השחקן, השם הקצר, ה-seed וכל ההגדרות, ולא משתנה אחר כך. ה-seed קטן ממיליון ([#13](https://github.com/sela27/pong_ai_bot/issues/13)). כל הגדרה של ה-Run היא אפשרות בשם שלה, למשל `--budget` או `--fill-steps`, ומה שלא משנים נשאר בברירת המחדל מהטבלה ב-[spec](https://github.com/sela27/pong_ai_bot/issues/17). ‏`uv run pong train --help` מראה את כולן.

בהתחלה הפקודה מדפיסה על איזה מכשיר היא רצה, ומסרבת לרוץ על ה-CPU בלי `--device cpu` ([#11](https://github.com/sela27/pong_ai_bot/issues/11)). ה-buffer מתמלא לפני הלמידה, וה-Step-ים של המילוי נספרים ב-Budget ([#14](https://github.com/sela27/pong_ai_bot/issues/14)). ‏checkpoint מלא נכתב כל 50,000 Step-ים, ו-Milestone ב-Step 0, כל 500,000 Step-ים ובסוף ה-Run, שנעצר ב-Budget. בזמן שה-Run רץ, Windows לא נרדם ([#10](https://github.com/sela27/pong_ai_bot/issues/10)).

‏`Ctrl+C` אחד מסיים את ה-Step הנוכחי, כותב checkpoint מלא באותו Step ויוצא, בלי Milestone, כי ה-Run לא נגמר. ‏`Ctrl+C` שני יוצא מיד, בלי checkpoint של ה-Step, ואז ה-checkpoint האחרון הוא זה שנכתב לפניו ([#10](https://github.com/sela27/pong_ai_bot/issues/10)). בשני המקרים קוד היציאה הוא 130, כמו של תוכנית שנעצרה ב-`Ctrl+C`.

בינתיים השחקנים היחידים הם של ארבע סביבות הבדיקה, `test-env-1` עד `test-env-4`, כדי שסוכן יוכל להריץ Run-ים זעירים בתיקייה זמנית ([#9](https://github.com/sela27/pong_ai_bot/issues/9)).

### להמשיך

```
uv run pong resume ~/pong_ai_bot_runs/2026-11-27_1430_test-env-1_tiny_seed1 --device cpu
```

המשך של Run שנעצר, ב-`Ctrl+C` או בקריסה, כאותו Run ([#10](https://github.com/sela27/pong_ai_bot/issues/10)). הפקודה מקבלת רק את תיקיית ה-Run: ההגדרות נקראות מקובץ ההגדרות שלה, ואי אפשר לשנות אותן. אם הקוד ב-commit אחר מזה שה-Run נוצר בו, מוצגת אזהרה, וה-Run ממשיך.

ה-Run ממשיך מה-checkpoint המלא האחרון, עם Game חדש. מונה ה-Step-ים ממשיך משם, ו-ε איתו, ו-`run/resumes` עולה ב-1. ה-buffer לא נשמר, ולכן buffer חדש מתמלא במדיניות הנוכחית לפני שהלמידה חוזרת, וה-Step-ים של המילוי נספרים ב-Budget ([#14](https://github.com/sela27/pong_ai_bot/issues/14)). מה שנכתב לגרפים אחרי ה-checkpoint נמחק מ-TensorBoard, כדי שהקווים לא יחזרו אחורה. ‏Run שהגיע ל-Budget לא ממשיך, כי אין מה להמשיך. ‏Run שנעצר לפני ה-checkpoint הראשון לא ממשיך: מתחילים Run חדש.

כמו `pong train`, היא מדפיסה על איזה מכשיר היא רצה ומסרבת לרוץ על ה-CPU בלי `--device cpu`, ‏Windows לא נרדם בזמן שה-Run רץ, ו-`Ctrl+C` עוצר אותו באותה צורה.

### לראות את הגרפים

```
uv run tensorboard --logdir ~/pong_ai_bot_runs
```

‏TensorBoard אחד על התיקייה הראשית, או על התיקייה ש-`PONG_AI_BOT_RUNS_DIR` מציין, מראה את כל ה-Run-ים יחד, כדי להשוות ביניהם ([#10](https://github.com/sela27/pong_ai_bot/issues/10)). הוא נפתח בדפדפן ב-`http://localhost:6006`, ומתעדכן בזמן שה-Run רץ, באיחור של עד שתי דקות. כל Run כותב את הגרפים לתיקייה שלו, וציר ה-X הוא מונה ה-Step-ים. השמות והתדירויות בטבלה "גרפים" ב-[spec](https://github.com/sela27/pong_ai_bot/issues/17). ‏`train/max_q` מחושב על 1,000 קלטים קבועים, שה-Run אוסף במשחק אקראי לפני Step 0 ושומר בתיקייה שלו.

### למדוד

```
uv run pong measure-env
```

כמה Step-ים בשנייה הסביבה עושה: סביבה אחת, ‏Action-ים אקראיים, עם ה-Opponent ועם ציור ה-Observation. השער הוא 5,000 לפחות ([#14](https://github.com/sela27/pong_ai_bot/issues/14)), ומתחתיו הפקודה נכשלת. זה סקריפט ולא בדיקה, כי המהירות תלויה במחשב.

```
uv run pong evaluate --scripted random
```

‏Evaluation של Scripted player: ‏Game אחד נגד ה-Opponent לכל אחד מ-100 ה-seed-ים של הסט הרגיל, ובסוף שורת סיכום ([#13](https://github.com/sela27/pong_ai_bot/issues/13)). עם `--output FILE` הפקודה גם מוסיפה לסוף הקובץ את השורה של ה-Evaluation, בפורמט של `evaluations.jsonl` ב-Run, עם התוצאה של כל seed. אין בה אפשרות לסט ה-Confirmation, כי אליו מגיעים רק בסוף Run ([#16](https://github.com/sela27/pong_ai_bot/issues/16)).

ה-Scripted player-ים:

- `random`: ‏Action אקראית בכל Step, עם seed קבוע.
- `edge-predictor`: מחשב לאן הכדור יגיע, כולל קפיצה מקיר, ומחזיר אותו מקצה ה-Paddle ב-45 מעלות, לצד הרחוק מה-Paddle של ה-Opponent. כשהכדור מתרחק, הוא חוזר למרכז. מחליט כל Step, כמו ה-Agent ([#7](https://github.com/sela27/pong_ai_bot/issues/7)).

### לבחור את רוחב ההיסט

```
uv run pong choose-width
```

‏Evaluation של שני ה-Scripted player-ים על הסט הרגיל, נגד Opponent בכל אחד מחמשת רוחבי ההיסט. אחרי כל Evaluation מודפסת שורת הסיכום שלו, ובסוף טבלה עם הממוצע של כל שחקן בכל רוחב ומה נכשל בכל רוחב, והרוחב שנבחר לפי הכלל של [#7](https://github.com/sela27/pong_ai_bot/issues/7). אם אף רוחב לא עובר, הפקודה אומרת את זה, לא בוחרת, ונכשלת. בתיקו בממוצע של המנבא נבחר הרחב מבין הרוחבים, והפקודה אומרת שהיה תיקו ([#62](https://github.com/sela27/pong_ai_bot/issues/62)). זה סקריפט ולא בדיקה: 10 ה-Evaluation-ים לוקחים כמה דקות. הפקודה לא נועלת את הרוחב.

הרוחב נבחר וננעל על 100% מחצי גובה ה-Paddle, כברירת המחדל של ה-Opponent, בכל מקום שהוא משחק, גם ב-`pong play` וב-`pong evaluate` ([#31](https://github.com/sela27/pong_ai_bot/issues/31), לפי הכלל של [#62](https://github.com/sela27/pong_ai_bot/issues/62)). שתי בדיקות איטיות, `uv run pytest -m slow`, בודקות שברוחב הנעול שני ה-Scripted player-ים עדיין עוברים את הכלל.

## רשימות בדיקה

- [בדיקת החלון בעין](docs/checklists/window.md): אחרי כל שינוי בחלון.
