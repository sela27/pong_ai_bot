# pong_ai_bot

משחק Pong שנבנה לוקלית, ו-Agent מסוג DQN שלומד לשחק בו רק מה-Observation ומה-Reward, ברוח המאמר של DeepMind. הכל רץ על המחשב של המפתח. המונחים מוגדרים ב-[GLOSSARY.md](GLOSSARY.md).

היעד: ‏Score margin ממוצע של 18 לפחות ב-Confirmation. ההחלטות, והקישורים לסיבות שלהן, ב-[spec](https://github.com/sela27/pong_ai_bot/issues/17).

## מצב השלבים

| שלב | מצב |
|---|---|
| 1. הסביבה | בעבודה |
| 2. הלומד ולולאת האימון | לא התחיל |
| 3. ה-Agent מפיקסלים | לא התחיל |

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
