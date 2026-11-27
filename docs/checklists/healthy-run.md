# ‏Run בריא

בזמן כל Run, של ה-Position learner ושל ה-Agent, משווים את הגרפים ב-TensorBoard לשורות כאן, וסוכן מאבחן לפיהן Run שנכשל. זה לא שער: ה-Evaluation הוא השופט ([#10](https://github.com/sela27/pong_ai_bot/issues/10)).

```
uv run tensorboard --logdir $HOME\pong_ai_bot_runs
```

אף גרף לא מספיק לבד, וציר ה-X בכולם הוא מונה ה-Step-ים של ה-Run. אין כאן ספים לעצירה: מותר לוותר על Run ביד אחרי מבט בגרפים, ואז הוא נחשב Run שנכשל ([#14](https://github.com/sela27/pong_ai_bot/issues/14)). בכל שורה: איך הגרף נראה כשהכל תקין, ו"אם לא": מה זה כנראה אומר, ואיפה מחפשים קודם.

1. **ה-Evaluation**, ‏`eval/mean`, ‏`eval/min` ו-`eval/cut_off`: ב-Step 0 קרוב ל-21-, כמו ה-Scripted player האקראי ([#9](https://github.com/sela27/pong_ai_bot/issues/9)), ועולה מ-Milestone ל-Milestone, כל 500,000 Step-ים ([#10](https://github.com/sela27/pong_ai_bot/issues/10)), עם מעט Game-ים שנחתכו. ‏CleanRL הגיע ל-18 אחרי 1.6M עד 3.5M Step-ים ([#14](https://github.com/sela27/pong_ai_bot/issues/14)), אבל ב-ALE Pong ולא ב-Pong שלנו: זה הקשר, ולא סף. ה-Run נעצר בעצמו ב-Milestone הראשונה עם 19 לפחות ([#14](https://github.com/sela27/pong_ai_bot/issues/14)).
   - אם לא: ‏`eval/mean` ב-Step 0 הרבה מעל 21- הוא כנראה באג ב-Evaluation או ב-Opponent, ומשווים קודם ל-`uv run pong evaluate --scripted random`. אם הרבה Game-ים נחתכים, רואים ב-`evaluations.jsonl` שבתיקיית ה-Run באילו seed-ים, ובאיזה Score margin.
2. **אורך ה-Game באימון**, ‏`train/game_length` ו-`train/game_length_avg100`: עולה לפני ה-Score margin. בהתחלה, במשחק כמעט אקראי, Point נמשך 20 עד 40 Step-ים ([#12](https://github.com/sela27/pong_ai_bot/issues/12)), וה-Game כמה מאות Step-ים.
   - אם לא: אורך שלא עולה גם אחרי ש-ε ירד ל-0.01, בשורה 6, הוא שחקן שלא לומד להחזיר את הכדור, ומסתכלים קודם על ה-Q ועל ה-loss, בשורות 4 ו-5. אורך שמגיע ל-27,000 Step-ים, ‏108,000 Frame-ים, הוא Game שנחתך ([#4](https://github.com/sela27/pong_ai_bot/issues/4)): השחקן מחזיר בלי לנצח.
3. **ה-Score margin באימון**, ‏`train/score_margin` ו-`train/score_margin_avg100`: עולה, ומפגר אחרי ה-Evaluation בגלל ε. לא שופטים אותו לפני מיליון Step-ים, כש-ε מגיע ל-0.01 ([#12](https://github.com/sela27/pong_ai_bot/issues/12)).
   - אם לא: אם הוא עולה וה-Evaluation לא, מחפשים קודם ב-Evaluation. ‏`uv run pong evaluate --milestone` משחק Milestone שוב, על ה-CPU, ומשווים לשורה שלה ב-`evaluations.jsonl`. אם שניהם לא עולים אחרי מיליון Step-ים, השחקן לא לומד, ומסתכלים על שורות 4 ו-5.
4. **ה-Q על הסט הקבוע**, ‏`train/max_q`: בהתחלה יכול לרדת עד 3- או 5-, ואחר כך מתייצב בערך בין 1.35- ל-1.35+ ([#12](https://github.com/sela27/pong_ai_bot/issues/12)). הוא יכול לעלות גם כשהמשחק לא משתפר.
   - אם לא: ‏Q שעולה הרבה מעבר ל-1.35+ הוא סימן להתבדרות, ובודקים קודם את חישוב ה-target ואת ההעתקה ל-target network. אם הוא מתנדנד או מתפוצץ, התיקון הראשון הוא העתקה איטית יותר ל-target network, ב-Run חדש עם `--target-copy-every` גדול יותר ([#12](https://github.com/sela27/pong_ai_bot/issues/12)).
5. **ה-loss**, ‏`train/loss`: רועש, אבל לא עולה בהתמדה. הוא מתחיל רק אחרי המילוי, 50,000 Step-ים ([#12](https://github.com/sela27/pong_ai_bot/issues/12)), ואחרי כל המשך אין בו נקודות לאורך המילוי.
   - אם לא: אם הוא מתפוצץ, התיקון הראשון הוא Huber עם חיתוך של נורמת הגרדיאנט ל-10, ב-Run חדש ([#12](https://github.com/sela27/pong_ai_bot/issues/12)). ‏`--loss` מקבל היום רק `mse`, ואין חיתוך של הגרדיאנט ([#17](https://github.com/sela27/pong_ai_bot/issues/17)), ולכן שניהם קודם שינוי בקוד.
6. ‏**ε**, ‏`train/epsilon`: קו ישר מ-1 ל-0.01 לאורך מיליון ה-Step-ים הראשונים, ואז 0.01, ובערך 0.95 בסוף המילוי הראשון, ב-Step 50,000 ([#12](https://github.com/sela27/pong_ai_bot/issues/12)). בהמשך Run הוא ממשיך מאותו ערך.
   - אם לא: אם הוא קופץ חזרה ל-1 אחרי המשך, מונה ה-Step-ים לא נטען מה-checkpoint.
7. ‏**Step-ים בשנייה, רק ל-Agent**, ‏`train/steps_per_second`: ‏116 לפחות ([#14](https://github.com/sela27/pong_ai_bot/issues/14)), ויציב לאורך ה-Run. קפיצה למעלה בזמן המילוי, בהתחלה ואחרי כל המשך, צפויה. הזמן של ה-Evaluation לא נספר בו. ל-Position learner אין יעד מהירות.
   - אם לא: אם הוא יורד לאט לאורך שעות, משהו דולף, ועושים profiling.
8. **מספר ההמשכים**, ‏`run/resumes`: ‏0 ב-Step 0, וקופץ ב-1 בכל המשך. ירידה קטנה בגרפים האחרים מיד אחרי המשך צפויה, כי ה-buffer מתחיל ריק.
   - אם לא: אם יש ירידה אחרי כל המשך, התיקון הראשון הוא לשמור את ה-buffer בעצירה מתוכננת, ב-Run חדש ([#10](https://github.com/sela27/pong_ai_bot/issues/10)).

## ‏Run-ים שהסתיימו

שורה לכל Run שהסתיים, גם לכזה שנכשל: שם התיקייה, הלומד, השלב בסולם ([#14](https://github.com/sela27/pong_ai_bot/issues/14)), ה-Step שבו ה-Evaluation עבר 18 בפעם הראשונה ([#13](https://github.com/sela27/pong_ai_bot/issues/13)) ואז 19 ([#14](https://github.com/sela27/pong_ai_bot/issues/14)), איך הוא נגמר (ב-19+, ב-Budget או בוויתור ביד), תוצאת ה-Confirmation אם הייתה, ומשפט אחד על מה שנראה אחרת מהשורות למעלה. ב-Run שנכשל, המשפט הוא תמצית האבחון. אחרי כל Run מתקנים גם את השורות למעלה לפי מה שנראה בו.

עדיין אין Run שהסתיים.
