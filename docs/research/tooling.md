# בחירת כלים: שפה, ספריית רשתות ורינדור

מחקר לטיקט [#2](https://github.com/sela27/pong_ai_bot/issues/2), חלק מהמפה [#1](https://github.com/sela27/pong_ai_bot/issues/1). נכתב ב-2026-11-25. כל הגרסאות נבדקו מול המקורות ביום הזה, ומספרי המקורות בסוגריים מרובעים מפנים לרשימה בסוף.

## Question

באילו כלים נבנה את הסביבה ואת ה-Agent, כך שהכול ירוץ לוקלית על המחשב הזה: Windows 11, ‏Quadro P2200 עם 5 GB, ו-Python 3.14.3?

## Summary

- **ב-PyTorch העדכני (2.15.0) אין יותר תמיכה בכרטיס.** הגרסה האחרונה שתומכת ב-sm_61 היא **PyTorch 2.14.1 ב-build ‏`cu126`**, ויש לה wheel ל-Python 3.14 על Windows [4][5][6].
- **אין צורך ב-Python ישן יותר.** כל החבילות שנדרשות מפרסמות wheels ל-cp314 על Windows [6][13].
- **לרינדור של ה-Observation מומלץ ציור ישיר במערך NumPy**, ו-pygame-ce רק לחלון שבו בן אדם צופה במשחק. ‏`pygame` המקורי לא מפרסם wheels ל-Python 3.14 [13].
- **כדאי לממש את הסביבה לפי ה-API של Gymnasium 1.4.0.** המחיר הוא תלות קטנה ב-pure Python, והתמורה היא `check_env`, ‏wrappers מוכנים ואפשרות להשוות מול ספרייה חיצונית [13][17][20].

## Local machine

נתונים שנמדדו על המחשב עצמו (`nvidia-smi`, ‏`python --version`, ‏`uv python list`, ‏`py --list`):

| פריט | ערך |
|---|---|
| ‏GPU | ‏Quadro P2200, ‏5120 MiB, ‏compute capability 6.1 (sm_61) |
| ‏Driver | ‏582.70, ומדווח "CUDA Version: 13.0" (המקסימום שה-driver תומך בו) |
| ‏CPU / RAM | ‏Intel Core i9-10900K (10 ליבות, 20 threads), ‏31.7 GB RAM |
| ‏Python מערכת | ‏3.14.3 ב-`C:\Python314\python.exe` (וגם ה-`py` launcher, עם 3.14 כברירת מחדל) |
| Python נוספים | ‏3.12.14 שמנוהל על ידי uv |
| ‏uv | ‏0.12.6 (הותקן דרך winget) |
| חבילות ב-Python 3.14 | אין `numpy` ואין `pygame`. יש ‏Pillow 12.2.0 |

## PyTorch and CUDA on sm_61

### ‏PyTorch 2.15 הסיר את התמיכה ב-Pascal

- ב-release notes של PyTorch 2.15.0 כתוב שה-wheels של 2.15 הם רק `cu132` ו-`cu134`, ש-`cu126` ו-`cu130` כבר לא נבנים, ושגרסאות CUDA 13 דורשות GPU מדור Turing ‏(`sm_75`) ומעלה. ההנחיה המפורשת שם היא שבעלי Maxwell, ‏Pascal ו-Volta צריכים להישאר על 2.14 [4].
- טבלת הארכיטקטורות של 2.15 ב-`RELEASE.md` מפרטת ל-CUDA 13.2.2 ול-13.4.3 רק את Turing ‏(7.5) ומעלה, גם ל-Linux וגם ל-Windows [1].
- הודעה מוקדמת בפורום המפתחים של PyTorch ‏(4 בספטמבר 2026) אמרה ש-2.14 הוא ה-release האחרון עם wheels של CUDA 12.x, ושמי שצריך 12.6 מגרסה 2.15 והלאה צריך לבנות מקוד מקור [5].
- הסיבה היא ב-NVIDIA: ב-CUDA Toolkit 13.0 הוסרו offline compilation ותמיכת הספריות ל-Maxwell, ‏Pascal ו-Volta, בעוד ש-toolkits מסדרה 12.x ממשיכים לתמוך בבנייה לארכיטקטורות האלה [9].
- בפועל, באינדקס ההורדות של PyTorch יש ל-2.15.0 על Windows רק את הגרסאות `cpu`, ‏`cu132`, ‏`cu134` ו-`xpu` [7].

### ‏PyTorch 2.14.1 עם `cu126` כן תומך ב-sm_61

- סקריפט הבנייה של ה-wheels ל-Windows בתג `v2.14.1` מגדיר ל-CUDA 12.6 את רשימת הארכיטקטורות `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0`, כלומר sm_61 נבנה במפורש. לעומת זאת, ל-`128`, ‏`129`, ‏`130` ו-`132` הרשימה מתחילה ב-7.5. הערה באותו קובץ מסבירה ש-"CUDA 13.x dropped sm_50/60/70" [2].
- הטבלה הקריאה ב-`RELEASE.md` של אותו תג מציגה ל-CUDA 12.6.3 את "Pascal(6.0)" בלבד [1b]. רשימת הבנייה [2] היא המקור המדויק יותר, והיא כוללת את 6.1.
- ב-`cu126` של 2.14.1 יש wheel ל-`cp314-cp314-win_amd64` (וגם ל-3.10 עד 3.15) [6]. גודל הקובץ כ-2.5 GB (נמדד ב-HEAD request על הקובץ ב-[6]).
- ‏2.14.1 היא גרסת ה-patch האחרונה של 2.14: לפי לוח הזמנים ב-`RELEASE.md`, ל-2.14 לא מתוכנן patch שני [1].
- ל-2.14.1 יש גם builds של `cu130` ו-`cu132` ל-Python 3.14 [7], אבל לפי [2] הם לא כוללים sm_61. **צריך לבחור במפורש ב-`cu126`.**
- ה-wheels של Windows כוללים את ה-DLL-ים של CUDA בתוך `torch/lib/`, ולכן לא צריך להתקין CUDA Toolkit נפרד [3].
- ה-wheel של torch ב-PyPI עבור Windows הוא CPU בלבד: ‏`torch-2.14.1-cp314-cp314-win_amd64.whl` שוקל כ-118 MB [8], ו-uv מתעד ש-PyPI "hosts CPU-only wheels for Windows" [11]. לכן חייבים את האינדקס של PyTorch.

### ‏Driver

- ה-driver המותקן (582.70) מדווח CUDA 13.0, ולכן הוא חדש מספיק ל-CUDA 12.6: בטבלת הגרסאות של NVIDIA, ‏CUDA 12.6 Update 3 על Windows דורש driver ‏`>=561.17` [9].
- לפי NVIDIA, ‏driver branch ‏R580 הוא "the final branch that supports GPUs before CC 7.5", והוא LTS שנתמך עד אמצע 2028 [10]. **מסקנה (היסק, לא נבדק מול רשימת ה-branches של NVIDIA):** כנראה ש-582.70 שייך ל-R580, כי הוא מדווח CUDA 13.0, הגרסה ש-R580 יצא איתה [9][10]. לכן לא כדאי לשדרג ל-branch חדש יותר, כי branch חדש לא יתמוך בכרטיס.

### התקנה עם uv: מה לא לעשות

- ‏uv מתעד את `uv pip install --torch-backend=auto`, שבוחר אינדקס לפי גרסת ה-CUDA driver המותקן [11]. **היסק:** כאן ה-driver מדווח CUDA 13.0, ולכן `auto` כנראה יבחר build של CUDA 13, שאין בו sm_61. צריך לציין את `cu126` במפורש.
- הדרך המתועדת לנעול אינדקס ב-`pyproject.toml` היא `[[tool.uv.index]]` עם `explicit = true` ו-`[tool.uv.sources]` [11]. בפרויקט הזה זה ייראה כך:

```toml
[project]
requires-python = ">=3.14,<3.15"
dependencies = ["torch==2.14.1", "numpy>=2.5", "gymnasium>=1.4,<2", "pygame-ce>=2.5.8"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

- בדיקה אחרי ההתקנה: ‏`torch.cuda.get_arch_list()` מחזירה את רשימת ארכיטקטורות ה-CUDA שהספרייה נבנתה עבורן [12], וצריך לראות בה `sm_61`.

## Python version

- ‏PyTorch 2.14 תומך ב-Python ‏`>=3.10, <=(3.15, 3.15t experimental)` [1], כלומר 3.14 נתמך באופן מלא, ולא כ-experimental.
- wheels של cp314 ל-Windows קיימים עבור: ‏torch 2.14.1+cu126 [6], ‏numpy 2.5.5, ‏pygame-ce 2.5.8 ו-Pillow 13.0.0. ‏Gymnasium 1.4.0 הוא wheel של pure Python ‏(`py3-none-any`) ומצהיר על תמיכה ב-3.10 עד 3.14 [13].
- ‏`pygame` המקורי: הגרסה האחרונה היא 2.6.1 מספטמבר 2024, וה-wheels שלה ל-Windows מגיעים רק עד cp313 [13]. על Python 3.14, ‏pip יצטרך לבנות אותו מקוד מקור. ‏pygame-ce, לעומת זאת, תומך ב-3.10 עד 3.15 בגרסה 2.5.8 [16].
- **מסקנה:** ה-venv יכול להשתמש ב-Python 3.14 שכבר מותקן. אין סיבה לרדת ל-3.12 או ל-3.13.

## Rendering the Observation headless

### האפשרויות

1. **ציור ישיר במערך NumPy.** המשחק מצייר מלבנים (מחבטים וכדור) כ-slice assignment על מערך `uint8`. אין תלות ב-SDL ואין חלון, והתוצאה כבר מערך שאפשר להעביר ל-PyTorch.
2. **‏pygame-ce עם surface בלי חלון.** ‏pygame-ce מגדיר את הקבוע `pygame.NULL_VIDEODRIVER = "dummy"` [15], והדוגמה הרשמית לעבודה בלי מערכת חלונות מציבה אותו ב-`SDL_VIDEODRIVER` לפני `pygame.display.init()` [15]. כדי לקבל מערך: ‏`surfarray.pixels3d` מחזיר view בלי העתקה, אבל נועל את ה-Surface כל עוד המערך קיים. ‏`array3d` מעתיק [14]. ‏`surfarray` דורש NumPy, והמערכים בו מאונדקסים "by the X axis first", כלומר `[x, y]` ולא `[y, x]` כמו במערך תמונה רגיל [14]. נוסף על זה, ‏surface הוא RGB, וכדי להגיע ל-Observation בגווני אפור ברזולוציה קטנה צריך עוד המרה ועוד הקטנה.
3. **‏Pillow ‏(`ImageDraw`).** מצייר על תמונה במצב `"L"` (גווני אפור) ומחזיר bytes. הוא כבר מותקן, ולכן אפשר היה למדוד אותו.

### מדידה מקומית

NumPy ו-pygame-ce לא מותקנים, ולפי ההנחיה לא התקנתי אותם. לכן מדדתי רק את מה שכבר קיים: Python 3.14.3 ו-Pillow 12.2.0. המדידה היא 20,000 צעדים של משחק דמוי Pong ‏(כדור, מחבט של ה-Agent ו-Opponent שעוקב אחרי הכדור), עם frame בגווני אפור בגודל 160×210, על ה-i9-10900K. שתי ריצות:

| מה נמדד | צעדים בשנייה (ריצה 1 / ריצה 2) |
|---|---|
| לוגיקת המשחק בלבד, ב-pure Python | ‏1,426,055 / 1,419,890 |
| לוגיקה + ציור ב-`bytearray` של pure Python | ‏85,216 / 82,280 |
| לוגיקה + ‏Pillow `ImageDraw` + `tobytes()` | ‏93,159 / 97,075 |
| לוגיקה + Pillow + הקטנה ל-84×84 ‏(`NEAREST`) | ‏45,092 / 44,731 |

<details>
<summary>הסקריפט (גרסה מקוצרת של זה שנמדד. הרצה שלה נתנה 1,445,546 / 77,839 / 97,813 / 46,416, באותו סדר גודל)</summary>

```python
import sys, time
import PIL
from PIL import Image, ImageDraw

W, H, PADDLE_W, PADDLE_H, BALL, N = 160, 210, 4, 16, 2, 20000

class Game:
    def __init__(self):
        self.bx, self.by, self.vx, self.vy = 80.0, 105.0, 2.0, 1.5
        self.ly = self.ry = 97.0
    def step(self, action):
        self.ry = min(max(self.ry + (action - 1) * 3.0, 0.0), H - PADDLE_H)
        self.ly += max(-2.5, min(2.5, self.by - (self.ly + PADDLE_H / 2)))
        self.bx += self.vx; self.by += self.vy
        if self.by < 0 or self.by > H - BALL: self.vy = -self.vy
        if self.bx < 8 + PADDLE_W and self.ly <= self.by <= self.ly + PADDLE_H: self.vx = abs(self.vx)
        if self.bx > W - 8 - PADDLE_W - BALL and self.ry <= self.by <= self.ry + PADDLE_H: self.vx = -abs(self.vx)
        if self.bx < 0: self.bx, self.by = 80.0, 105.0; return 1
        if self.bx > W: self.bx, self.by = 80.0, 105.0; return -1
        return 0

def bench(name, fn):
    g = Game(); t0 = time.perf_counter()
    for i in range(N): g.step(i % 3); fn(g)
    print(f"{name:45s} {N / (time.perf_counter() - t0):10.0f} steps/s")

img = Image.new("L", (W, H), 0); draw = ImageDraw.Draw(img)
def pillow_render(g):
    draw.rectangle((0, 0, W - 1, H - 1), fill=0)
    draw.rectangle((8, int(g.ly), 8 + PADDLE_W - 1, int(g.ly) + PADDLE_H - 1), fill=255)
    draw.rectangle((W - 8 - PADDLE_W, int(g.ry), W - 9, int(g.ry) + PADDLE_H - 1), fill=255)
    draw.rectangle((int(g.bx), int(g.by), int(g.bx) + BALL - 1, int(g.by) + BALL - 1), fill=255)
    return img.tobytes()

buf, ZERO = bytearray(W * H), bytes(W * H)
def fill_rect(x, y, w, h):
    x, y, row = max(0, min(W - w, x)), max(0, min(H - h, y)), b"\xff" * w
    for r in range(y, y + h): buf[r * W + x:r * W + x + w] = row
def bytearray_render(g):
    buf[:] = ZERO
    fill_rect(8, int(g.ly), PADDLE_W, PADDLE_H)
    fill_rect(W - 8 - PADDLE_W, int(g.ry), PADDLE_W, PADDLE_H)
    fill_rect(int(g.bx), int(g.by), BALL, BALL)
    return bytes(buf)

print("python", sys.version.split()[0], "| Pillow", PIL.__version__)
bench("game logic only (pure Python)", lambda g: None)
bench("logic + pure-Python bytearray render", bytearray_render)
bench("logic + Pillow ImageDraw render + tobytes", pillow_render)
bench("logic + Pillow render + resize to 84x84",
      lambda g: (pillow_render(g), img.resize((84, 84), Image.Resampling.NEAREST).tobytes()))
```

</details>

### מה אפשר להסיק מזה

- **הרינדור הוא העלות העיקרית, לא הפיזיקה:** לוגיקת המשחק עולה פחות מ-1 µs לצעד, והרינדור כ-10 עד 20 µs לצעד (חישוב מהטבלה).
- **הסביבה לא תהיה צוואר הבקבוק.** ה-DQN המקורי אומן על "a total of 10 million frames" [21]. גם בקצב הנמוך שנמדד (כ-45,000 צעדים בשנייה), ‏10 מיליון צעדים של סביבה הם כ-4 דקות של CPU (חישוב: 10,000,000 / 45,000 ≈ 222 שניות). את רוב הזמן יבלה אימון הרשת. לא מדדתי את קצב האימון על ה-P2200.
- **‏NumPy ו-pygame-ce לא נמדדו.** הערכה בלי מדידה: ציור ב-NumPy הוא כמה slice assignments של C על מערך קטן, ולכן הוא צפוי להיות באותו סדר גודל כמו `bytearray` או מהיר ממנו. ב-pygame-ce הציור עצמו ב-C, אבל העתקה של surface בצבע ל-NumPy, המרה לגווני אפור והקטנה מוסיפות עבודה. **צריך לאמת את זה במדידה קצרה אחרי שה-venv יהיה מוכן.**

### ההמלצה לרינדור

- **ה-Observation ייווצר בציור ישיר במערך NumPy ‏`uint8` בגווני אפור, ישר ברזולוציה הסופית.** ה-DQN המקורי המיר את ה-frame לגווני אפור, הקטין וחתך אותו ל-84×84, וערם את 4 ה-frames האחרונים לקלט אחד של 84×84×4 [21]. ציור ישיר ברזולוציה הסופית חוסך את שלבי ההמרה וההקטנה, וגם את התלות ב-OpenCV ש-`ResizeObservation` של Gymnasium דורש [19]. הרזולוציה עצמה היא החלטה ל-spec.
- **‏pygame-ce 2.5.8 משמש רק כחלון לבן אדם**: לצפות ב-Agent, או לשחק בעצמך נגד ה-Opponent. החלון יציג את אותו מערך NumPy, מוגדל, דרך `surfarray` [14], כך שמה שרואים בחלון הוא בדיוק ה-Observation.
- לא `pygame` המקורי, כי אין לו wheels ל-Python 3.14 [13].

## Gymnasium API

### מה ה-API דורש

- מחלקה שיורשת מ-`gymnasium.Env`, ומגדירה `observation_space` ו-`action_space` [17].
- ‏`reset(seed=None, options=None)` מחזיר `(observation, info)`, ואת הקריאה `super().reset(seed=seed)` צריך לבצע ראשונה כדי לזרוע את `self.np_random` [17][18].
- ‏`step(action)` מחזיר `(observation, reward, terminated, truncated, info)` [17]. ‏`terminated` מסמן מצב סופי של המשימה, ו-`truncated` מסמן תנאי מחוץ למשימה, כמו מגבלת זמן [18].
- ‏`metadata` עם `render_modes` ו-`render_fps`. במצב `"rgb_array"`, ‏`render()` מחזיר frame כמערך NumPy [18].

### מיפוי למונחים של הפרויקט

- **‏Game** הוא episode ב-Gymnasium: ‏`terminated=True` כשאחד הצדדים מגיע ל-21 **Points**.
- **‏Reward** הוא ה-`reward` שמוחזר מ-`step`: חיובי כשה-Agent זוכה ב-Point ושלילי כשהוא מפסיד אחת.
- **‏Observation** הוא ה-`observation`. ה-`render()` של Gymnasium נשאר לצפייה בלבד.
- ‏`truncated` רלוונטי רק אם נחליט על מגבלת זמן ל-Game. זו החלטה ל-spec.

### מה זה עולה

- תלות אחת נוספת: ‏Gymnasium 1.4.0 (יצא ב-5 באוקטובר 2026), ‏pure Python, ותלוי ב-`numpy`, ‏`cloudpickle`, ‏`typing-extensions` ו-`farama-notifications`, שכולן pure Python חוץ מ-NumPy [13][20].
- צריך להגדיר spaces (למשל `Box(0, 255, (84, 84), uint8)` ל-Observation ו-`Discrete(3)` לפעולות), ולהחזיר את ה-tuple בן חמשת האיברים.
- **מלכודת:** החל מ-1.4.0, ‏`check_env` בודק ש-`reset` ו-`step` מחזירים אובייקטים נפרדים של observation בכל צעד, כדי למנוע דריסה שקטה של buffers ב-replay buffer [20]. אם מציירים לתוך מערך אחד שחוזר על עצמו, צריך להחזיר עותק.

### מה זה נותן

- ‏`check_env` "will catch many common issues" בסביבה מותאמת אישית [17].
- wrappers מוכנים לעיבוד פיקסלים: ‏`FrameStackObservation` (ערימת N ה-Observations האחרונות), ‏`MaxAndSkipObservation` (דילוג על frames ומקסימום על השניים האחרונים), ו-`GrayscaleObservation` [19].
- אפשרות להשוות מול מימוש DQN מוכן: ‏Stable-Baselines3 2.9.2 דורש `gymnasium>=0.29.1,<2.0` ו-`torch>=2.8` [13], כך שאפשר להריץ אותו על אותה סביבה בדיוק, כ-baseline או לדיבוג.
- **אין חובה להשתמש ב-wrappers.** אפשר לממש את ה-API ועדיין לכתוב את ה-frame stacking בעצמנו, כחלק מהלמידה. זו החלטה ל-spec.

## Recommendation

| רכיב | בחירה | למה |
|---|---|---|
| שפה | ‏**Python 3.14** (ה-3.14.3 המותקן), ב-venv שמנוהל על ידי uv 0.12.6 | לכל החבילות יש wheels ל-cp314 על Windows [6][13] |
| ספריית רשתות | ‏**PyTorch 2.14.1, build ‏`cu126`**, מ-`https://download.pytorch.org/whl/cu126` ‏(`explicit = true`) | הגרסה האחרונה שנבנתה עם sm_61 [2][4][5]. בלי `--torch-backend=auto` ובלי `cu130`/`cu132` |
| ‏Driver | להישאר על branch ‏R580 (כרגע 582.70) | הוא האחרון שתומך ב-Pascal, ‏LTS עד אמצע 2028 [10] |
| מערכים | ‏**NumPy 2.5.5** | ‏wheel ל-cp314 [13] |
| רינדור ה-Observation | **ציור ישיר במערך NumPy ‏`uint8` בגווני אפור, ברזולוציה הסופית** | בלי SDL, בלי המרה, בלי OpenCV [19][21]. מהיר מספיק בהרבה (מדידה למעלה) |
| חלון לצפייה ולמשחק אנושי | ‏**pygame-ce 2.5.8**, שמציג את אותו מערך דרך `surfarray` | ל-`pygame` המקורי אין wheels ל-3.14 [13][14][16] |
| ‏API של הסביבה | ‏**Gymnasium 1.4.0** ‏(`reset` / `step`) | ‏`check_env`, ‏wrappers ו-baseline אפשרי, במחיר של תלות קטנה [13][17][19][20] |
| ‏torchvision | לא צריך | אין בו צורך ל-DQN מפיקסלים כשהרינדור כבר מוכן ברזולוציה הסופית |

## Uncertain

- **הקצב של NumPy ושל pygame-ce לא נמדד**, כי הן לא מותקנות. נמדדו רק pure Python ו-Pillow. צריך לחזור על המדידה אחרי שה-venv מוכן.
- **‏PyTorch 2.14.1+cu126 על Python 3.14 עם NumPy 2.5.5 לא נבדק מקצה לקצה.** התאימות מבוססת על metadata ועל קבצי הבנייה בלבד [1][2][6]. צריך לאמת אחרי ההתקנה (`torch.cuda.is_available()` ו-`sm_61` ב-`get_arch_list()` [12]).
- **קצב האימון של הרשת על ה-P2200 לא ידוע.** הוא ישפיע על טיקט תקציב האימון יותר מקצב הסביבה.
- **ההתנהגות של `--torch-backend=auto` היא היסק** מהתיעוד של uv [11], ולא נבדקה.
- **השיוך של 582.70 ל-R580 הוא היסק** מכך שהוא מדווח CUDA 13.0 [9][10], ולא נבדק מול רשימת ה-branches של NVIDIA.
- **‏2.14.1 הוא תקרה קבועה:** פיצ'רים של 2.15 ומעלה לא יהיו זמינים על הכרטיס הזה. החלופות הן בנייה מקוד מקור עם CUDA 12.x [5], או אימון על CPU עם הגרסה העדכנית. אף אחת מהן לא נבדקה.
- **‏PyTorch לא התחייב עד מתי ה-wheels של 2.14 יישארו להורדה.** ההודעה אומרת שהם "remain available" בלי תאריך [5].

## Sources

- [1] PyTorch, `RELEASE.md` בענף `main` ‏(commit `8d5a301`): טבלת התאימות (שורות 54-55), ארכיטקטורות של 2.15 (שורות 75-84), לוח זמנים (שורות 108-109). https://github.com/pytorch/pytorch/blob/8d5a3010668c665b4f8cc44e07c5a777a0d9abc2/RELEASE.md
- [1b] PyTorch, `RELEASE.md` בתג `v2.14.1`: טבלת הארכיטקטורות של 2.12 ו-2.13, שורה 77 ‏(CUDA 12.6.3). https://github.com/pytorch/pytorch/blob/v2.14.1/RELEASE.md?plain=1#L73-L77
- [2] PyTorch, `.ci/pytorch/windows/build_env_setup.py` בתג `v2.14.1`, שורות 65-75 ‏(`TORCH_CUDA_ARCH_LIST_TABLE`). https://github.com/pytorch/pytorch/blob/v2.14.1/.ci/pytorch/windows/build_env_setup.py#L65-L75
- [3] PyTorch, `.ci/pytorch/windows/build_wheel.py` בתג `v2.14.1`, שורות 10-14. https://github.com/pytorch/pytorch/blob/v2.14.1/.ci/pytorch/windows/build_wheel.py#L10-L14
- [4] PyTorch v2.15.0 release notes, הסעיף "PyPI wheels now ship with CUDA 13.2, and CUDA 12.6 and CUDA 13.0 wheels are no longer published". https://github.com/pytorch/pytorch/releases/tag/v2.15.0
- [5] Andrey Talman, "Notice: CUDA 12.6 wheels will no longer be published from PyTorch 2.15 (drops Maxwell / Pascal / Volta)", ‏dev-discuss.pytorch.org, 4 בספטמבר 2026. https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
- [6] אינדקס ה-wheels של PyTorch ל-`cu126` (נבדק ב-2026-11-25). https://download.pytorch.org/whl/cu126/torch/
- [7] אינדקס ה-wheels הכללי של PyTorch (נבדק ב-2026-11-25). https://download.pytorch.org/whl/torch/
- [8] PyPI JSON של torch: ‏latest 2.15.0, וה-wheels של 2.14.1. https://pypi.org/pypi/torch/json ‏, https://pypi.org/pypi/torch/2.14.1/json
- [9] NVIDIA, CUDA Toolkit 13.0 Release Notes: סעיף 2.6.2 (Deprecated Architectures) וטבלאות גרסאות ה-driver. https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- [10] Jonathan Bentz ו-Tony Scudiero, "Navigating GPU Architecture Support: A Guide for NVIDIA CUDA Developers", ‏NVIDIA Technical Blog, 4 באוגוסט 2025. https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/
- [11] Astral, "Using uv with PyTorch". https://docs.astral.sh/uv/guides/integration/pytorch/
- [12] PyTorch 2.14 docs, `torch.cuda.get_arch_list`. https://docs.pytorch.org/docs/2.14/generated/torch.cuda.get_arch_list.html
- [13] PyPI JSON (נבדק ב-2026-11-25) של: numpy ‏(2.5.5), pygame ‏(2.6.1), pygame-ce ‏(2.5.8), gymnasium ‏(1.4.0), pillow ‏(13.0.0), opencv-python, stable-baselines3 ‏(2.9.2), ואת התלויות של Gymnasium ‏(cloudpickle, ‏typing-extensions ו-farama-notifications, שלכולן יש רק wheel מסוג `py3-none-any`). https://pypi.org/pypi/numpy/json ‏, https://pypi.org/pypi/pygame/json ‏, https://pypi.org/pypi/pygame-ce/json ‏, https://pypi.org/pypi/gymnasium/1.4.0/json ‏, https://pypi.org/pypi/pillow/json ‏, https://pypi.org/pypi/stable-baselines3/json
- [14] pygame-ce docs, `pygame.surfarray`. https://pyga.me/docs/ref/surfarray.html
- [15] pygame-ce בתג `2.5.8`: הקבוע `NULL_VIDEODRIVER` ב-`src_c/constants.c` שורה 659, והדוגמה `examples/headless_no_windows_needed.py` שורות 21-28. https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/constants.c#L659 ‏, https://github.com/pygame-community/pygame-ce/blob/2.5.8/examples/headless_no_windows_needed.py#L21-L28
- [16] pygame-ce 2.5.8 release notes ‏(9 באוגוסט 2026): "supported range to Python 3.10 - 3.15". https://github.com/pygame-community/pygame-ce/releases/tag/2.5.8
- [17] Gymnasium docs, "Create a Custom Environment". https://gymnasium.farama.org/introduction/create_custom_env/
- [18] Gymnasium docs, `Env` API. https://gymnasium.farama.org/api/env/
- [19] Gymnasium docs, Observation Wrappers. https://gymnasium.farama.org/api/wrappers/observation_wrappers/
- [20] Gymnasium v1.4.0 release notes ‏(5 באוקטובר 2026). https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.4.0
- [21] Mnih et al., "Playing Atari with Deep Reinforcement Learning", ‏arXiv:1312.5602 (2013), סעיפים 4.1 ו-5. https://arxiv.org/abs/1312.5602
- מדידות מקומיות: ‏`nvidia-smi`, ‏`nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total`, ‏`python --version`, ‏`py --list`, ‏`uv python list`, ‏`python -m pip list`, והסקריפט שלמעלה. כולן הורצו על המחשב ב-2026-11-25.
