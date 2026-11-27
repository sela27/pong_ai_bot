# לפני Run

לפני כל Run חדש, של ה-Position learner או של ה-Agent, עוברים על השורות לפי הסדר, בתיקייה של ה-repo שממנה מתחילים את ה-Run. זה שער: לא מתחילים Run לפני שכל השורות עברו. בהמשך של Run לא צריך את הרשימה, כי `pong resume` כבר מזהיר כשה-commit השתנה.

1. אין שינויים בלי commit:

   ```
   git status
   ```

   עבר: ‏`nothing to commit, working tree clean`. גם קובץ חדש שלא נכנס ל-git הוא שינוי.

2. הבדיקות האיטיות עוברות, כמו שנקבע ב-[#11](https://github.com/sela27/pong_ai_bot/issues/11):

   ```
   uv run pytest -m slow
   ```

   עבר: בשורה האחרונה רק `passed` ו-`deselected`, בלי `failed`, ‏`error` ו-`skipped`.

3. אין Run אחר שרץ על המחשב, גם לא מ-worktree אחר. ב-PowerShell:

   ```
   Get-CimInstance Win32_Process -Filter "Name = 'pong.exe'" | Where-Object CommandLine -Match '" (train|resume) ' | Select-Object ProcessId, CommandLine
   ```

   עבר: הפקודה לא מדפיסה כלום. כל שורה שהיא כן מדפיסה היא `pong train` או `pong resume` שעדיין רץ.
