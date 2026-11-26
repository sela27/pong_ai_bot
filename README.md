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

## רשימות בדיקה

- [בדיקת החלון בעין](docs/checklists/window.md): אחרי כל שינוי בחלון.
